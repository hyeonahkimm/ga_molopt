from __future__ import print_function

import torch
import random
from typing import List

import numpy as np
from joblib import delayed
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol
rdBase.DisableLog('rdApp.error')

from utils import seq_to_smiles
# import main.genetic_gfn.genetic_operator.string_crossover as co
# import main.genetic_gfn.genetic_operator.string_mutate as mu

import gc


MINIMUM = 1e-10


def crossover_and_mutate(parent_a, parent_b, model, mutation_rate=0.05, temp=1.0):
    seq_a = torch.tensor(model.voc.encode(model.voc.tokenize(parent_a)))
    seq_b = torch.tensor(model.voc.encode(model.voc.tokenize(parent_b)))

    if torch.cuda.is_available():
        seq_a = seq_a.cuda()
        seq_b = seq_b.cuda()

    tokens = torch.unique(torch.concat([seq_a, seq_b], axis=0))

    mask = torch.zeros((1, model.voc.vocab_size)).to(seq_a.device)
    mask[0, tokens.long()] = 1
    mask[0, model.voc.vocab['EOS']] = 1

    child, _, _ = model.regenerate(1, mask=mask, mutation_rate=mutation_rate, temp=temp) # 0.1)
    child_smiles = seq_to_smiles(child, model.voc)[0]

    return child_smiles


def select_next(population_smi, population_scores, novelty, population_size: int, rank_coefficient=0.01, replace=False):
    scores_np = np.array(population_scores) + 1e-4  # including invalid molecules
    ranks = np.argsort(np.argsort(-1 * scores_np))
    if novelty is not None:
        ranks = 0.5 * ranks + 0.5 * np.argsort(np.argsort(-1 * novelty))
    weights = 1.0 / (rank_coefficient * len(scores_np) + ranks)
    
    indices = list(torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=population_size, replacement=replace
        ))
    next_pop = [population_smi[i] for i in indices]
    next_pop_score = [population_scores[i] for i in indices]
    
    if novelty is not None:
        next_pop_novelty = np.array([novelty[i] for i in indices])
        return next_pop, next_pop_score, next_pop_novelty
    
    return next_pop, next_pop_score, None


def make_mating_pool(population_smi, population_mol: List[Mol], population_scores, population_size: int, rank_coefficient=0.01):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights
    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return
    Returns: a list of RDKit Mol (probably not unique)
    """
    # scores -> probs 
    if rank_coefficient > 0:
        scores_np = np.array(population_scores) + 1e-4  # including invalid molecules
        ranks = np.argsort(np.argsort(-1 * scores_np))
        weights = 1.0 / (rank_coefficient * len(scores_np) + ranks)
        
        indices = list(torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=population_size, replacement=True
            ))
        mating_pool = [population_smi[i] for i in indices]
        mating_pool_score = [population_scores[i] for i in indices]
        # mating_pool = [population_mol[i] for i in indices if population_mol[i] is not None]
        # mating_pool_score = [population_scores[i] for i in indices if population_mol[i] is not None]
        # print(mating_pool)
    else:
        population_scores = [s + MINIMUM for s in population_scores]
        sum_scores = sum(population_scores)
        population_probs = [p / sum_scores for p in population_scores]
        # mating_pool = np.random.choice(population_mol, p=population_probs, size=offspring_size, replace=True)
        indices = np.random.choice(np.arange(len(population_mol)), p=population_probs, size=population_size, replace=True)
        mating_pool = [population_mol[i] for i in indices if population_mol[i] is not None]
        mating_pool_score = [population_scores[i] for i in indices if population_mol[i] is not None]

    return mating_pool, mating_pool_score


def reproduce(mating_pool, mutation_rate, model):
    """
    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation
    Returns:
    """
    for _ in range(10):
        try:
            parent_a = random.choice(mating_pool)
            parent_b = random.choice(mating_pool)
            new_child = crossover_and_mutate(parent_a, parent_b, model, mutation_rate)  # smiles
            if new_child is not None:
                return new_child
        except:
            new_child = None
    if new_child is None:
        new_child = random.choice(mating_pool)
    # if new_child is not None:
    #     new_child = mu.mutate(new_child, mutation_rate)
    return new_child


class GeneticOperatorHandler:
    def __init__(self, mutation_rate: float=0.067, population_size=200):
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.temp = 2.0

    # def get_final_population(self, mating_pool, rank_coefficient=0.):
    #     new_mating_pool, new_mating_scores, _, _ = make_mating_pool(mating_pool[0], mating_pool[1], self.population_size, rank_coefficient)
    #     return (new_mating_pool, new_mating_scores)
    def select(self, population_smi, population_scores, novelty=None, rank_coefficient=0.01, replace=False):

        return select_next(population_smi, population_scores, novelty, min(self.population_size, len(population_scores)), rank_coefficient, replace=replace)

    def query(self, query_size, mating_pool, pool, model=None, rank_coefficient=0.01, mutation_rate=None):
        # print(mating_pool)
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
        population_smi = mating_pool[0]
        population_mol = [Chem.MolFromSmiles(s) for s in population_smi]
        population_scores = mating_pool[1]

        # mating pool: List[smiles]
        cross_mating_pool, cross_mating_scores = make_mating_pool(population_smi, population_mol, population_scores, self.population_size, rank_coefficient)

        # print(model.rnn.device)

        offspring_smi = pool(delayed(reproduce)(cross_mating_pool, mutation_rate, model) for _ in range(query_size))
        # new_mating_pool = cross_mating_pool
        # new_mating_scores = cross_mating_scores

        smis = []
        for smi in offspring_smi:
            try:
                # smis.append(Chem.MolToSmiles(m))
                # smi = Chem.MolToSmiles(m)
                if smi not in smis:  # unique
                    smis.append(smi)
                    # print(smi)
                    # n_atoms.append(m.GetNumAtoms())
            except:
                pass

        gc.collect()

        # pop_valid_smis, pop_valid_scores = [], []
        # for m, s in zip(new_mating_pool, new_mating_scores):
        #     try:
        #         # pop_valid_smis.append(Chem.MolToSmiles(m))
        #         pop_valid_smis.append(Chem.MolToSmiles(m))
        #         pop_valid_scores.append(s)
        #     except:
        #         pass

        return smis, None, cross_mating_pool, cross_mating_scores
