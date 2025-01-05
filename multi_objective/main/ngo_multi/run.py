from __future__ import print_function

import random
from typing import List

import joblib
import numpy as np
# from joblib import delayed
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol
rdBase.DisableLog('rdApp.error')

# import main.molleo_multi.crossover as co, main.molleo_multi.mutate as mu
from main.optimizer import BaseOptimizer

#from main.graph_ga.mol_lm import MolCLIP
# from main.molleo_multi.biot5 import BioT5
# from main.molleo_multi.GPT4 import GPT4
from utils import get_fp_scores
# from network import create_and_train_network, obtain_model_pred
import torch
import os

from tdc import Evaluator
from data_structs import Vocabulary, Experience, MolData
from model import RNN
from rnn_utils import Variable, seq_to_smiles, unique


MINIMUM = 1e-10

def make_mating_pool(population_mol: List[Mol], population_smiles, population_scores, offspring_size: int, rank_coefficient=0.01):
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
    all_tuples = list(zip(population_scores, population_mol, population_smiles))
    population_scores = [s + MINIMUM for s in population_scores]
    if rank_coefficient > 0:
        scores_np = np.array(population_scores) + 1e-4  # including invalid molecules
        ranks = np.argsort(np.argsort(-1 * scores_np))
        weights = 1.0 / (rank_coefficient * len(scores_np) + ranks)
        
        mating_indices = list(torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=offspring_size, replacement=True
            ))
    else:
        sum_scores = sum(population_scores)
        population_probs = [p / sum_scores for p in population_scores]
        mating_indices = np.random.choice(len(all_tuples), p=population_probs, size=offspring_size, replace=True)
        
    mating_tuples = [all_tuples[indices] for indices in mating_indices]
    
    return mating_tuples

def reproduce(mating_tuples, mutation_rate, model=None, net=None):
    """
    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation
    Returns:
    """
    for _ in range(100):
        new_child = None
        try:
            parent = []
            parent.append(random.choice(mating_tuples))
            parent.append(random.choice(mating_tuples))

            parent_mol = [t[2] for t in parent]  # we use smiles
            if parent_mol[0] == parent_mol[1]:
                continue
            new_child = crossover_and_mutate(parent_mol[0], parent_mol[1], model, mutation_rate)  # smiles
        except:
            pass
        if new_child is not None:
            break

    # new_child = co.crossover(parent_mol[0], parent_mol[1])
    # new_child_mutation = None
    # if new_child is not None:
    #     new_child_mutation = mu.mutate(new_child, mutation_rate, mol_lm)
    return new_child


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


def get_best_mol(population_scores, population_mol):
    top_mol = population_mol[np.argmax(population_scores)]
    top_smi = Chem.MolToSmiles(top_mol)
    return top_smi


def smiles_to_seqs(smiles, scores, voc, unique=False):
    valid_seqs, valid_scores, valid_smis = [], [], []
    for i, smi in enumerate(smiles):
        if unique and smi in valid_smis:
            continue
        try:
            tokenized = voc.tokenize(smi)
            valid_seqs.append(Variable(voc.encode(tokenized)))
            valid_scores.append(scores[i])
            valid_smis.append(smi)
        except:
            pass
    valid_seqs = MolData.collate_fn(valid_seqs)
    
    return valid_smis, valid_scores, valid_seqs


def get_seq_distances(seqs, ref_seqs, voc):
    cnt = torch.zeros(voc.vocab_size).to(seqs.device)
    for ref in ref_seqs:
        len_ref = torch.where(ref == voc.vocab['EOS'])[0] + 1
        cnt[ref[:len_ref]] += 1
    freq = cnt.view(1, -1) / ref_seqs.size(0)
    return freq[:, seqs].sum(-1).view(-1)

def get_mol_distances(mols, ref_mols):
    evaluator = Evaluator(name = 'Diversity')
    distances = []
    for mol in mols:
        try:
            dist = [evaluator([mol, ref]) for ref in ref_mols]
        except:
            import pdb; pdb.set_trace()
        distances.append(np.mean(dist))
    return distances

def select(scores, distances, num_population, replace=False, rank_coefficient=0.01):
    if len(scores) < num_population:
        return list(range(len(scores)))
    
    if isinstance(scores, list):
        scores = torch.tensor(scores).to(distances.device)
    # scores_np = np.array(scores) + 1e-4  # including invalid molecules
    ranks = torch.argsort(torch.argsort(-1 * scores).view(-1))  # np.argsort(np.argsort(-1 * scores_np))
    
    ranks = 0.5 * ranks + 0.5 * torch.argsort(torch.argsort(-1 * distances))  # np.argsort(np.argsort(-1 * distances))
    weights = 1.0 / (rank_coefficient * len(scores) + ranks)

    indices = list(torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=num_population, replacement=replace
        ))

    return indices


class GA_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "neural_ga"

        # self.mol_lm = None
        # if args.mol_lm == "GPT-4":
        #     self.mol_lm = GPT4()
        # elif args.mol_lm == "BioT5":
        #     self.mol_lm = BioT5()
        
        self.args = args
        # lm_name = "baseline"
        # if args.mol_lm != None:
        #     lm_name = args.mol_lm
        #     self.mol_lm.task = self.args.task_mode

    def _optimize(self, config):

        self.oracle.assign_evaluator(self.args)
        
        path_here = os.path.dirname(os.path.realpath(__file__))
        restore_prior_from=os.path.join(path_here, 'data/Prior.ckpt')
        restore_agent_from=restore_prior_from 
        voc = Vocabulary(init_from_file=os.path.join(path_here, "data/Voc"))
        
        Prior = RNN(voc)
        Agent = RNN(voc)

        # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
        # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
        # to the CPU.
        if torch.cuda.is_available():
            Prior.rnn.load_state_dict(torch.load(os.path.join(path_here,'data/Prior.ckpt')))
            Agent.rnn.load_state_dict(torch.load(restore_agent_from))
        else:
            Prior.rnn.load_state_dict(torch.load(os.path.join(path_here, 'data/Prior.ckpt'), map_location=lambda storage, loc: storage))
            Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

        # We dont need gradients with respect to Prior
        for param in Prior.rnn.parameters():
            param.requires_grad = False

        # optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=config['learning_rate'])
        log_z = torch.nn.Parameter(torch.tensor([5.]).cuda()) if torch.cuda.is_available() else torch.nn.Parameter(torch.tensor([5.]))
        optimizer = torch.optim.Adam([{'params': Agent.rnn.parameters(), 
                                        'lr': config['learning_rate']},
                                    {'params': log_z, 
                                        'lr': config['lr_z']}])

        experience = Experience(voc, max_size=config['num_keep'])

        # pool = joblib.Parallel(n_jobs=self.n_jobs)
        
        if self.smi_file is not None:
            # Exploitation run
            starting_population = self.all_smiles[:config["population_size"]]
        else:
            # Exploration run
            starting_population = np.random.choice(self.all_smiles, config["population_size"])

        # select initial population
        population_smiles = starting_population
        population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
        population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])
        
        # experience.add_experience(zip(population_smiles, population_scores))

        patience = 0
        step = 0
        
        while True:

            if len(self.oracle) > 100:
                self.sort_buffer()
                old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
            else:
                old_scores = 0
            
            # Sample from Agent
            seqs, agent_likelihood, entropy = Agent.sample(config['batch_size'])

            # Remove duplicates, ie only consider unique seqs
            unique_idxs = unique(seqs)
            seqs = seqs[unique_idxs]
            agent_likelihood = agent_likelihood[unique_idxs]
            entropy = entropy[unique_idxs]

            # Get prior likelihood and score
            prior_likelihood, _ = Prior.likelihood(Variable(seqs))
            smiles = seq_to_smiles(seqs, voc)
            score = np.array(self.oracle(smiles))
            

            # Calculate augmented likelihood
            augmented_likelihood = prior_likelihood.float() + config['sigma'] * Variable(score).float()
            loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

            # Experience Replay
            # First sample
            if config['experience_replay'] and len(experience)>config['experience_replay']:
                exp_seqs, exp_score = experience.sample(config['experience_replay'])
                exp_prior_likelihood, _ = Prior.likelihood(exp_seqs.long())
                exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
                exp_augmented_likelihood = exp_prior_likelihood + config['sigma'] * torch.from_numpy(exp_score).to(exp_prior_likelihood.device)
                exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
                loss = torch.cat((loss, exp_loss), 0)
                agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

            # Then add new experience
            prior_likelihood = prior_likelihood.data.cpu().numpy()
            new_experience = zip(smiles, score, prior_likelihood)
            experience.add_experience(new_experience)

            # Calculate loss
            loss = loss.mean()

            # Add regularizer that penalizes high likelihood for the entire sequence
            loss_p = - (1 / agent_likelihood).mean()
            loss += 5 * 1e3 * loss_p

            # Calculate gradients and make an update to the network weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Convert to numpy arrays so that we can print them
            augmented_likelihood = augmented_likelihood.data.cpu().numpy()
            agent_likelihood = agent_likelihood.data.cpu().numpy()

            step += 1
            
            if len(self.oracle) > config['max_training']:
                print('max oracle hit')
                break 

            # early stopping
            if len(self.oracle) > 1000:
                self.sort_buffer()
                new_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
                if new_scores == old_scores:
                    patience += 1
                    if patience >= self.args.patience:
                        # self.log_intermediate(finish=True)
                        # print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0


        print("Starting GA")
        patience = 0

        if len(self.oracle) > 100:
            self.sort_buffer()
            old_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())[:100]])
        else:
            old_score = 0
            
        all_smiles, all_scores = tuple(map(list, zip(*[(smi, elem[0]) for (smi, elem) in self.oracle.mol_buffer.items()])))
        all_smiles, all_scores, all_seqs = smiles_to_seqs(all_smiles, all_scores, voc)
        all_dist = get_seq_distances(all_seqs.long(), all_seqs.long(), voc)
        indices = select(all_scores, all_dist, config["population_size"])  #np.random.choice(np.arange(len(all_smiles)), config["population_size"], replace=False)
        # valid_mols, valid_scores, valid_smiles = [], [], []
        # for i, smi in enumerate(all_smiles):
        #     try:
        #         mol = Chem.MolFromSmiles(smi)
        #         if mol is not None:
        #             valid_smiles.append(smi)
        #             valid_scores.append(all_scores[i])
        #             valid_mols.append(mol)
        #     except:
        #         pass
        # all_dist = get_mol_distances(all_smiles, all_smiles)
        # all_dist = Variable(torch.tensor(all_dist))
        # indices = select(all_scores, all_dist, config["population_size"])  #np.random.choice(np.arange(len(all_smiles)), config["population_size"], replace=False)

        # select initial population
        population_smiles = [all_smiles[i] for i in indices]
        population_scores = [all_scores[i] for i in indices]
        population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
        
        population_smiles, population_scores, population_seqs = smiles_to_seqs(population_smiles, population_scores, voc)
        population_dist = get_seq_distances(population_seqs.long(), population_seqs.long(), voc)
        
        while True:

            # new_population
            # import pdb; pdb.set_trace()
            mating_tuples = make_mating_pool(population_mol, population_smiles, population_scores, config["population_size"])
            
            reproduced_smis = [reproduce(mating_tuples, config["mutation_rate"], Agent) for _ in range(config["offspring_size"])]
            
            offspring_smis, offspring_mol = [], []
            for smis in reproduced_smis:
                try:
                    mol = Chem.MolFromSmiles(smis)
                    if mol != None:
                        offspring_smis.append(smis)
                        offspring_mol.append(mol)
                except:
                    pass
            
            offspring_score = self.oracle(offspring_smis)
            new_experience = zip(offspring_smis, offspring_score)
            print(len(self.oracle), np.max(offspring_score), np.mean(offspring_score))
            experience.add_experience(new_experience)

            # add new_population
            population_mol += offspring_mol
            population_mol = self.sanitize(population_mol)

            # stats
            old_scores = population_scores
            population_smiles = [Chem.MolToSmiles(mol) for mol in population_mol]
            population_scores = self.oracle(population_smiles)
            # population_smiles += offspring_smis  # = [Chem.MolToSmiles(mol) for mol in population_mol]
            # population_scores += offspring_score #self.oracle(population_smiles)
            # population_tuples = list(zip(population_scores, population_mol, population_smiles))
            # population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:config["population_size"]]
            # _, _, offspring_seqs = smiles_to_seqs(offspring_smis, offspring_score, voc)
            # offspring_dist = get_seq_distances(offspring_seqs.long(), offspring_seqs.long(), voc)
            # population_dist = torch.cat([population_dist, offspring_dist])
            population_smiles, population_scores, population_seqs = smiles_to_seqs(population_smiles, population_scores, voc)
            population_dist = get_seq_distances(population_seqs.long(), population_seqs.long(), voc)
            # population_dist = get_mol_distances(population_smiles, population_smiles)
            # population_dist = Variable(torch.tensor(population_dist))
            next_indices = select(population_scores, population_dist, config["population_size"])
            population_mol = [population_mol[t] for t in next_indices]
            population_smiles = [population_smiles[t] for t in next_indices]
            population_scores = [population_scores[t] for t in next_indices]
            
            # Experience Replay
            # First sample
            avg_loss = 0.
            if config['experience_replay'] and len(experience) > config['experience_replay']:
                for _ in range(config['experience_loop']):
                    if config['rank_coefficient'] > 0:
                        exp_seqs, exp_score = experience.rank_based_sample(config['experience_replay'], config['rank_coefficient'])
                    else:
                        exp_seqs, exp_score = experience.sample(config['experience_replay'])

                    exp_agent_likelihood, _ = Agent.likelihood(exp_seqs.long())
                    prior_agent_likelihood, _ = Prior.likelihood(exp_seqs.long())

                    reward = torch.tensor(exp_score).cuda()

                    exp_forward_flow = exp_agent_likelihood + log_z
                    exp_backward_flow = reward * config['beta']
                    
                    exp_backward_flow += prior_agent_likelihood.detach()  # rtb-style
                    loss = torch.pow(exp_forward_flow - exp_backward_flow, 2).mean()

                    # KL penalty
                    # if config['penalty'] == 'prior_kl':
                    # loss_p = (exp_agent_likelihood - prior_agent_likelihood).mean()
                    # loss += 0.01*loss_p

                    # print(loss.item())
                    avg_loss += loss.item()/config['experience_loop']

                    optimizer.zero_grad()
                    loss.backward()
                    # grad_norms = torch.nn.utils.clip_grad_norm_(Agent.rnn.parameters(), 1.0)
                    optimizer.step()
            # print(avg_loss)

            ### early stopping
            if len(self.oracle) > 100:
                self.sort_buffer()
                new_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())[:100]])
                # import ipdb; ipdb.set_trace()
                if (new_score - old_score) < 1e-3:
                    patience += 1
                    if patience >= self.args.patience:
                        self.log_intermediate(finish=True)
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0

                old_score = new_score
                
            if self.finish:
                break

