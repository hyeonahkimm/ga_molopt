import os
import sys
import numpy as np
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('/'.join(path_here.rstrip('/').split('/')[:-2]))
from main.optimizer import BaseOptimizer
from utils import Variable, seq_to_smiles, unique
from model import RNN
from data_structs import Vocabulary, Experience, MolData, Variable
from priority_queue import MaxRewardPriorityQueue
import torch
from rdkit import Chem
from tdc import Evaluator

from joblib import Parallel
# from graph_ga_expert import GeneticOperatorHandler
from amortized_ga import GeneticOperatorHandler


def sanitize(smiles):
    canonicalized = []
    for s in smiles:
        try:
            canonicalized.append(Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True))
        except:
            pass
    return canonicalized


class Amortized_GA_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "ngo_reinvent"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)

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
        # log_z = torch.nn.Parameter(torch.tensor([config['log_z']]).cuda()) if torch.cuda.is_available() else torch.nn.Parameter(torch.tensor([config['log_z']]))
        optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=config['learning_rate'])

        # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
        # occur more often (which means the agent can get biased towards them). Using experience replay is
        # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
        # experience = Experience(voc)
        experience = Experience(voc, max_size=config['num_keep'])

        ga_handler = GeneticOperatorHandler(mutation_rate=config['mutation_rate'], 
                                            population_size=config['population_size'])
        pool = Parallel(n_jobs=config['num_jobs'])

        print("Model initialized, starting training...")

        step = 0
        patience = 0
        prev_n_oracles = 0
        stuck_cnt = 0
        prev_updated = 0
        
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
            if config['valid_only']:
                smiles = sanitize(smiles)
            
            score = np.array(self.oracle(smiles))

            # early stopping
            if len(self.oracle) > 1000:
                self.sort_buffer()
                new_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
                if new_scores == old_scores:
                    patience += 1
                    if patience >= self.args.patience:
                        # self.log_intermediate(finish=True)
                        if config['starting_ga_from'] == 10000:
                            self.log_intermediate(finish=True)
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0

            # early stopping
            if prev_n_oracles < len(self.oracle):
                stuck_cnt = 0
            else:
                stuck_cnt += 1
                if stuck_cnt >= 10:
                    # self.log_intermediate(finish=True)
                    if config['starting_ga_from'] == 10000:
                        self.log_intermediate(finish=True)
                    print('cannot find new molecules, abort ...... ')
                    break
            
            prev_n_oracles = len(self.oracle)
            prev_updated = 0

            # Calculate augmented likelihood
            augmented_likelihood = prior_likelihood.float() + 500 * Variable(score).float()
            loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
            # print('REINVENT:', reinvent_loss.mean().item())

            # Then add new experience
            new_experience = zip(smiles, score)
            experience.add_experience(new_experience)

            if len(self.oracle) >= config['starting_ga_from']:
                break

            # Experience Replay
            # First sample
            if config['experience_replay'] and len(experience)>config['experience_replay']:
                exp_seqs, exp_score = experience.sample(config['experience_replay'])
                exp_prior_likelihood, _ = Prior.likelihood(exp_seqs.long())
                exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
                exp_augmented_likelihood = exp_prior_likelihood + 500 * torch.from_numpy(exp_score).to(exp_prior_likelihood.device)
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
            
            
        ############## GA #################
        if config['starting_ga_from'] < 10000:
            print("Starting GA ...")
            stuck_cnt, patience = 0, 0
            
            while True:
                if len(self.oracle) == 0:
                    # Exploration run
                    starting_population = np.random.choice(self.all_smiles, config["population_size"])

                    # select initial population
                    # population_smiles = heapq.nlargest(config["population_size"], starting_population, key=oracle)
                    population_smiles = starting_population
                    population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
                    population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])
                    all_smis, all_scores = population_smiles, population_scores
                else:
                    
                    self.oracle.sort_buffer()
                    all_smis, all_scores = tuple(map(list, zip(*[(smi, elem[0]) for (smi, elem) in self.oracle.mol_buffer.items()])))
                
                # log likelihood to measure novelty
                if config['use_novelty']:
                    all_smis, all_scores, all_seqs = smiles_to_seqs(all_smis, all_scores, voc)
                    
                    # with torch.no_grad():
                    #     pop_likelihood, _ = Agent.likelihood(all_seqs.long())
                    # novelty = (-1) * pop_likelihood.cpu().numpy()
                    all_novelty = None  # novelty(all_smis, all_smis)

                else:
                    all_novelty = None

                # mating_pool = (pop_smis[:config['num_keep']], pop_scores[:config['num_keep']])
                pop_smis, pop_scores, _ = ga_handler.select(all_smis, all_scores, all_novelty, rank_coefficient=config['rank_coefficient'], replace=False)
                # population = (pop_smis, pop_scores)
                if config['novelty_measure'] == 'edge':
                    pop_smis, pop_scores, valid_pop_seqs = smiles_to_seqs(pop_smis, pop_scores, voc)
                
                termination = False
                for ga_i in range(config['reinitiation_interval']):
                    if len(self.oracle) > 100:
                        self.sort_buffer()
                        old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
                    else:
                        old_scores = 0

                    if config['novelty_measure'] == 'edge':
                        pop_distances = get_pw_seq_distances(valid_pop_seqs, valid_pop_seqs, voc.vocab_size) if 'novelty' in config['mating_rule'] else None
                    else:
                        pop_distances = get_pw_distances(pop_smis, pop_smis) if 'novelty' in config['mating_rule'] else None
                    
                    all_child_smis, child_n_atoms, _, _ = ga_handler.query(
                            query_size=config['offspring_size'], mating_pool=(pop_smis, pop_scores), pool=pool, model=Agent,
                            rank_coefficient=config['rank_coefficient'], mating_rule=config['mating_rule'], pw_distances=pop_distances,
                            top_p = config['top_p'], dist_rank=0.2, chromosome=config['chromosome']  #len(self.oracle)/10000
                        )

                    all_child_score = np.array(self.oracle(all_child_smis))
                    # print(len(self.oracle), '| child_score:', child_score.mean(), child_score.max())
                    
                    # log likelihood to measure novelty
                    # valid_child_smis, valid_child_score = child_smis, child_score.tolist() #
                    # valid_child_smis, valid_child_score, valid_child_seqs = smiles_to_seqs(child_smis, child_score, voc)

                    
                    # if config['use_novelty']:
                    #     # with torch.no_grad():
                    #     #     child_likelihood, _ = Agent.likelihood(valid_child_seqs.long())
                    #     # child_novelty = (-1) * child_likelihood.cpu().numpy()
                    #     child_novelty = novelty(valid_child_smis, pop_smis)
                    # else:
                    #     child_novelty = None
                    child_smis, child_score= [], []
                    for c, s in zip(all_child_smis, all_child_score):
                        if c not in pop_smis:
                            child_smis.append(c)
                            child_score.append(s)
                
                    new_experience = zip(child_smis, child_score)
                    experience.add_experience(new_experience)
                    
                    if config['novelty_measure'] == 'edge':
                        valid_pop_smis, valid_pop_score, valid_pop_seqs = smiles_to_seqs(pop_smis+child_smis, pop_scores+child_score, voc)
                    
                        pw_distances = get_pw_seq_distances(valid_pop_seqs, valid_pop_seqs, voc.vocab_size) if config['use_novelty'] else None
                        pop_smis, pop_scores, _ = ga_handler.select(valid_pop_smis, 
                                                                                valid_pop_score, 
                                                                                # np.concatenate([pop_novelty, child_novelty]) if config['use_novelty'] else None, 
                                                                                pw_distances, 
                                                                                rank_coefficient=config['rank_coefficient'], 
                                                                                replace=False,
                                                                                dist_rank=0.2  #len(self.oracle)/10000
                                                                                )
                        
                        pop_smis, pop_scores, valid_pop_seqs = smiles_to_seqs(pop_smis+child_smis, pop_scores+child_score, voc)
                    
                    else:
                        valid_child_smis, valid_child_score, valid_child_seqs = smiles_to_seqs(child_smis, child_score, voc)
                        pop_smis, pop_scores, _ = ga_handler.select(pop_smis+valid_child_smis, 
                                                                                pop_scores+valid_child_score, 
                                                                                # np.concatenate([pop_novelty, child_novelty]) if config['use_novelty'] else None, 
                                                                                get_pw_distances(pop_smis+valid_child_smis, pop_smis+valid_child_smis) if config['use_novelty'] else None, 
                                                                                rank_coefficient=config['rank_coefficient'], 
                                                                                replace=False,
                                                                                dist_rank=0.2  #len(self.oracle)/10000
                                                                                )
                    
                    # pop_novelty = []
                    # for smi in pop_smis:
                    #     ref = list(filter(lambda x: x != smi, pop_smis))
                    #     novelty_score = novelty([smi], ref)
                    #     pop_novelty.append(novelty_score[0])
                    # pop_novelty = novelty(pop_smis, pop_smis)  # np.array(pop_novelty)
                    print(len(self.oracle), np.max(pop_scores), np.mean(pop_scores))
                    # population = (tuple(map(list, zip(*[(smi, elem[0]) for (smi, elem) in self.oracle.mol_buffer.items()]))))

                    if self.finish:
                        print('max oracle hit')
                        break
                    
                    if config['update_during_ga'] and len(self.oracle) > prev_updated + config['update_thr']:
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

                                reward = torch.tensor(exp_score).cuda() if torch.cuda.is_available() else torch.tensor(exp_score)

                                exp_forward_flow = exp_agent_likelihood + log_z
                                exp_backward_flow = (reward * min(500, config['beta'] * (len(self.oracle) // 1000 + 1))) if config['beta_annealing'] else (reward * config['beta'])
                                # exp_backward_flow = (reward * min(100, config['beta'] * (len(self.oracle) // 200 + 1))) if config['beta_annealing'] else (reward * config['beta'])
                                
                                if config['penalty'] == 'rtb':
                                    exp_backward_flow += prior_agent_likelihood.detach()  # rtb-style
                                loss = torch.pow(exp_forward_flow - exp_backward_flow, 2).mean()

                                # KL penalty
                                if config['penalty'] == 'prior_kl':
                                    loss_p = (exp_agent_likelihood - prior_agent_likelihood).mean()
                                    loss += config['kl_coefficient']*loss_p

                                # print(loss.item())
                                avg_loss += loss.item()/config['experience_loop']

                                optimizer.zero_grad()
                                loss.backward()
                                # grad_norms = torch.nn.utils.clip_grad_norm_(Agent.rnn.parameters(), 1.0)
                                optimizer.step()
                        print(avg_loss)
                        prev_updated = len(self.oracle)
                        
                    # early stopping
                    if len(self.oracle) > 1000:
                        self.sort_buffer()
                        new_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
                        if new_scores == old_scores:
                            patience += 1
                            # ga_handler.temp = min(0.2 + ga_handler.temp, 2.0)
                            if patience >= self.args.patience:
                                # self.log_intermediate(finish=True)
                                print('convergence criteria met (max patience), abort ...... ')
                                termination = True
                                break
                        else:
                            patience = 0
                            ga_handler.temp = 1

                    # early stopping
                    if prev_n_oracles < len(self.oracle):
                        stuck_cnt = 0
                    else:
                        stuck_cnt += 1
                        if stuck_cnt >= 10 and len(self.oracle) > 1000:
                            # self.log_intermediate(finish=True)
                            print('cannot find new molecules (max stuck counts), abort ...... ')
                            termination = True
                            break
                    prev_n_oracles = len(self.oracle)

                if self.finish:
                    print('max oracle hit')
                    break
                
                if termination:
                    self.log_intermediate(finish=True)
                    print('cannot find new molecules, abort ...... ')
                    break
                
                step += 1
                

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


def get_pw_distances(new_smiles, ref_smiles):
    evaluator = Evaluator(name = 'Diversity')  # pairwise
    novelty_scores = []
    valid_ref_smiles = sanitize(ref_smiles)
    for d in new_smiles:
        dist = np.array([evaluator([d, od]) for od in ref_smiles])
        novelty_scores.append(np.nan_to_num(dist).mean())
    # novelty_scores = np.nan_to_num(np.array(novelty_scores))
    return np.array(novelty_scores)


def get_pw_seq_distances(new_seqs, ref_seqs, vocab_size):
    ref_masks = []
    
    for ref in ref_seqs:
        edge_mask = torch.zeros((vocab_size, vocab_size)).to(ref.device)
        edge_mask[ref[:-1].long(), ref[1:].long()] = 1
        ref_masks.append(edge_mask)
    ref_masks = torch.stack(ref_masks)
    
    dist = []
    # import pdb; pdb.set_trace()
    for new in new_seqs:
        edge_mask = torch.zeros((vocab_size, vocab_size)).to(new.device)
        edge_mask[new[:-1].long(), new[1:].long()] = 1
        dist.append((edge_mask[None, :, :]*ref_masks).sum(-1).sum(-1).mean().item())
        
    return np.array(dist) * (-1)