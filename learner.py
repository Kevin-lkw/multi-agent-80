from multiprocessing import Process
import time
import numpy as np
import torch
from torch.nn import functional as F

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer, ModelPoolClient
from model import get_model
from torch.utils.data import DataLoader, TensorDataset, BatchSampler, SubsetRandomSampler

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
        
        # initialize model params
        device = torch.device(self.config['device'])
        model = get_model()
        
        # send to model pool
        model_pool.push(model.state_dict()) # push cpu-only tensor to model_pool
        model = model.to(device)
        
        # training
        optimizer = torch.optim.Adam(model.parameters(), lr = self.config['lr'], eps = 1e-5,)
        
        # wait for initial samples
        while self.replay_buffer.size() < self.config['min_sample']:
            time.sleep(0.1)
        
        cur_time = time.time()
        iterations = 0
        while True:
            # sample batch
            batch = self.replay_buffer.sample(self.config['batch_size'])
            obs = torch.tensor(batch['state']['observation']).to(device)
            mask = torch.tensor(batch['state']['action_mask']).to(device)
            seq = torch.tensor(batch['state']['seq_mat']).to(device)
            states = {
                'observation': obs,
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
            old_logits, _ = model(states)
            old_probs = F.softmax(old_logits, dim = 1).gather(1, actions)
            old_log_probs = torch.log(old_probs).detach()
            for _ in range(self.config['epochs']):
                # Create a new DataLoader for each epoch to ensure mini-batches are different
                dataset = TensorDataset(obs, mask, seq, actions, advs, targets)
                batch_sampler = BatchSampler(SubsetRandomSampler(range(len(dataset))), self.config['mini_batch_size'], False)
                data_loader = DataLoader(dataset, batch_sampler=batch_sampler)

                for mini_batch in data_loader:
                    obs_mb, mask_mb, seq_mb, actions_mb, advs_mb, targets_mb = mini_batch
                    states_mb = {
                        'observation': obs_mb,
                        'action_mask': mask_mb,
                        'seq_mat': seq_mb,
                    }
                    logits, values = model(states_mb)
                    ## values = value_model(state_with_perfect_information)
                    ## MCTS -> target values  max_depth = 25 * 4 max_width = 54 (usually 26++) - max_depth//4 (usually 26++) UCT score
                    action_dist = torch.distributions.Categorical(logits = logits)
                    probs = F.softmax(logits, dim = 1).gather(1, actions_mb)
                    log_probs = torch.log(probs)
                    ratio = torch.exp(log_probs - old_log_probs[:log_probs.size(0)])
                    surr1 = ratio * advs_mb
                    surr2 = torch.clamp(ratio, 1 - self.config['clip'], 1 + self.config['clip']) * advs_mb
                    policy_loss = -torch.mean(torch.min(surr1, surr2))
                    value_loss = torch.mean(F.mse_loss(values.squeeze(-1), targets_mb))
                    entropy_loss = -torch.mean(action_dist.entropy())
                    loss = policy_loss + self.config['value_coeff'] * value_loss + self.config['entropy_coeff'] * entropy_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # push new model
            model = model.to('cpu')
            model_pool.push(model.state_dict()) # push cpu-only tensor to model_pool
            model = model.to(device)
            
            # save checkpoints
            t = time.time()
            if t - cur_time > self.config['ckpt_save_interval']:
                path = self.config['ckpt_save_path'] + 'model_%d.pt' % iterations
                torch.save(model.state_dict(), path)
                cur_time = t
            iterations += 1

            # Check if this is the best model
            model_pool_client = ModelPoolClient(self.config['model_pool_name'])
            latest = model_pool_client.get_latest_model()
            if 'score' in latest and latest['score'] > self.best_score:
                self.best_score = latest['score']
                self.best_model_id = latest['id']
                best_path = self.config['best_model_path'] + 'best_model_%d.pt' % self.best_model_id
                torch.save(model.state_dict(), best_path)
                print(f"New best model saved: {self.best_model_id} with score {self.best_score}")
