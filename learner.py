from multiprocessing import Process
import time
import numpy as np
import torch
from torch.nn import functional as F
import gc
from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer, ModelPoolClient
from model import get_model, get_perfect_model
from torch.utils.data import DataLoader, TensorDataset, BatchSampler, SubsetRandomSampler
from mcts import MCTS

class Learner(Process):
    
    def __init__(self, config, replay_buffer):
        super(Learner, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.best_score = -float('inf')
        self.best_model_id = None
    
    def run(self):
        # create model pool
        model_pool = ModelPoolServer(self.config['model_pool_size'], self.config['model_pool_name'])
        model_pool_value = ModelPoolServer(self.config['model_pool_size'], 'model_pool_value')
        
        # initialize model params
        device = torch.device(self.config['device'])
        model = get_model()
        value_model = get_perfect_model()
        
        # send to model pool
        model_pool.push(model.state_dict()) # push cpu-only tensor to model_pool
        model_pool_value.push(value_model.state_dict())
        model = model.to(device)
        value_model = value_model.to(device)

        # training
        optimizer = torch.optim.Adam(model.parameters(), lr = self.config['lr'])
        optimizer_value = torch.optim.Adam(value_model.parameters(), lr = self.config['lr'])
        
        # wait for initial samples
        while self.replay_buffer.size() < self.config['min_sample']:
            time.sleep(0.1)
        
        cur_time = time.time()
        iterations = 0
        while True:
            optimizer.zero_grad()
            optimizer_value.zero_grad()
            # sample batch
            batch = self.replay_buffer.sample(self.config['batch_size'])
            per_obs = torch.tensor(batch['perfect_state']['perfect_observation']).to(device)
            obs = torch.tensor(batch['state']['observation']).to(device)
            mask = torch.tensor(batch['state']['action_mask']).to(device)
            seq = torch.tensor(batch['state']['seq_mat']).to(device)
            states = {
                    'observation': obs,
                    'action_mask': mask,
                    'seq_mat': seq,
                }
            per_states = {
                    'observation': per_obs,
                    'action_mask': mask,
                    'seq_mat': seq,
                }
            actions = torch.tensor(batch['action']).unsqueeze(-1).to(device)
            advs = torch.tensor(batch['adv']).to(device)
            targets = torch.tensor(batch['target']).to(device)
            #print(batch['seq_history'][0],'\n'*3)
            
            # Trick 1 Normalize advantages
            advs_mean = advs.mean()
            advs_std = advs.std()
            advs = (advs - advs_mean) / (advs_std + 1e-8)  # Adding a small value to avoid division by zero

            print('Iteration %d, replay buffer in %d out %d' % (iterations, self.replay_buffer.stats['sample_in'], self.replay_buffer.stats['sample_out']))
            
            # calculate PPO loss
            model.train(True) # Batch Norm training mode
            value_model.train(True) # Batch Norm training mode
            
            with torch.no_grad():
                old_logits = model(states)
                old_probs = F.softmax(old_logits, dim=1).gather(1, actions)
                old_log_probs = torch.log(old_probs).detach()
            for _ in range(self.config['epochs']):

                #value_targets_mb = [search_engine(per_info)  for per_info in per_info_mb]

                logits = model(states)
                
                values = value_model(per_states)
                ## values = value_model(state_with_perfect_information)
                ## MCTS -> target values  max_depth = 25 * 4 max_width = 54 (usually 26++) - max_depth//4 (usually 26++) UCT score
                action_dist = torch.distributions.Categorical(logits = logits)
                probs = F.softmax(logits, dim = 1).gather(1, actions)
                log_probs = torch.log(probs)
                ratio = torch.exp(log_probs - old_log_probs[:log_probs.size(0)])
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1 - self.config['clip'], 1 + self.config['clip']) * advs
                policy_loss = -torch.mean(torch.min(surr1, surr2)) 
                value_loss = torch.mean(F.mse_loss(values.squeeze(-1), targets))
                entropy_loss = -torch.mean(action_dist.entropy())
                policy_loss += self.config['entropy_coeff'] * entropy_loss
                #loss = policy_loss + self.config['value_coeff'] * value_loss + self.config['entropy_coeff'] * entropy_loss
                optimizer.zero_grad()
                optimizer_value.zero_grad()  
                policy_loss.backward()
                value_loss.backward()
                #print(value_loss.mean())
                #loss.backward()
                optimizer.step()
                optimizer_value.step()

                # 清理不再需要的变量和内存
                del logits, values, action_dist, probs, log_probs, ratio, surr1, surr2
                torch.cuda.empty_cache()

                gc.collect()  # 显式调用垃圾回收器

            # push new model
            model = model.to('cpu')
            model_pool.push(model.state_dict()) # push cpu-only tensor to model_pool
            model = model.to(device)

            value_model = value_model.to('cpu')
            model_pool_value.push(model.state_dict()) # push cpu-only tensor to model_pool
            value_model = value_model.to(device)
            # save checkpoints
            t = time.time()
            if t - cur_time > self.config['ckpt_save_interval']:
                path = self.config['ckpt_save_path'] + 'model_%d.pt' % iterations
                torch.save(model.state_dict(), path)
                cur_time = t
            iterations += 1


