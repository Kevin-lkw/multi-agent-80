from multiprocessing import Process
import time
import torch
import numpy as np
from model_pool import ModelPoolClient
from model import CNNModel
from env import TractorEnv
from wrapper import cardWrapper

class Evaluator(Process):
    def __init__(self, config):
        super(Evaluator, self).__init__()
        self.config = config

    def run(self):
        model_pool = ModelPoolClient(self.config['model_pool_name'])
        current_model = CNNModel()
        best_model = CNNModel()
        env = TractorEnv()
        self.wrapper = cardWrapper()

        best_score = float('-inf')
        best_model_id = None

        while True:
            latest = model_pool.get_latest_model()
            current_state_dict = model_pool.load_model(latest)
            current_model.load_state_dict(current_state_dict)

            if best_model_id is None:
                best_model.load_state_dict(current_state_dict)
            else:
                best_state_dict = torch.load(f"{self.config['best_model_path']}best_model_{best_model_id}.pt")
                best_model.load_state_dict(best_state_dict)

            total_reward = 0
        
            for _ in range(self.config['eval_episodes']):
                obs, action_options = env.reset()
                done = False
                episode_reward = {}
                #####################################################################################
                
                # run one episode and collect data
                obs, action_options = env.reset(major='r')
                episode_data = {agent_name: {
                    'state' : {
                        'observation': [],
                        'action_mask': []
                    },
                    'action' : [],
                    'reward' : [],
                    'value' : []
                } for agent_name in env.agent_names}

                done = False
                while not done:
                    state = {}
                    player = obs['id']
                    agent_name = env.agent_names[player]
                    agent_data = episode_data[agent_name]
                    obs_mat, action_mask = self.wrapper.obsWrap(obs, action_options)
                    agent_data['state']['observation'].append(obs_mat)
                    agent_data['state']['action_mask'].append(action_mask)
                    state['observation'] = torch.tensor(obs_mat, dtype = torch.float).unsqueeze(0)
                    state['action_mask'] = torch.tensor(action_mask, dtype = torch.float).unsqueeze(0)
                    best_model.train(False) # Batch Norm inference mode
                    current_model.train(False) # Batch Norm inference mode

                    with torch.no_grad():
                        if player % 2 == 0:  # Current model plays
                            logits, value = current_model(state)
                        else:  # Best model plays
                            logits, value = best_model(state)
                        action_dist = torch.distributions.Categorical(logits = logits)
                        action = action_dist.sample().item()
                        value = value.item()
                        
                    agent_data['action'].append(action)
                    agent_data['value'].append(value)
                    # interpreting actions
                    action_cards = action_options[action]
                    response = env.action_intpt(action_cards, player)
                    # interact with env
                    next_obs, action_options, rewards, done = env.step(response)
                    if rewards:
                        # rewards are added per four moves (1 move for each player) on all four players
                        for agent_name in rewards: 
                            episode_data[agent_name]['reward'].append(rewards[agent_name])
                    obs = next_obs

                for agent_name, agent_data in episode_data.items():
                    if len(agent_data['action']) < len(agent_data['reward']):
                        agent_data['reward'].pop(0)
                    obs = np.stack(agent_data['state']['observation'])
                    mask = np.stack(agent_data['state']['action_mask'])
                    actions = np.array(agent_data['action'], dtype = np.int64)
                    rewards = np.array(agent_data['reward'], dtype = np.float32)
                    values = np.array(agent_data['value'], dtype = np.float32)
                    next_values = np.array(agent_data['value'][1:] + [0], dtype = np.float32)
                    
                    td_target = rewards + next_values * self.config['gamma']
                    td_delta = td_target - values
                    advs = []
                    adv = 0
                    for delta in td_delta[::-1]:
                        adv = self.config['gamma'] * self.config['lambda'] * adv + delta
                        advs.append(adv) # GAE
                    advs.reverse()
                    advantages = np.array(advs, dtype = np.float32)
                    # Maybe GAE is better. But for eval, I prefer rewards may bring me more interpretability.
                    episode_reward[agent_name] = rewards 
                #####################################################################################
                #print(episode_reward) [  5.,   0., -10.,  10.,  30.,   0.,  10.,  10.,   0.,   5.,   5.,  5., -25.,  20.,  -5.,  -5.,  10.,   5., -10.,  10.,  10.,  -3.] Why 3 here ?
                total_reward += np.sum(episode_reward[env.agent_names[0]])



            avg_reward = total_reward / self.config['eval_episodes']
            print(f"Model {latest['id']} evaluation reward: {avg_reward}")

            if avg_reward > 0:
                best_score = avg_reward
                best_model_id = latest['id']
                torch.save(current_state_dict, f"{self.config['best_model_path']}best_model_{best_model_id}.pt")
                print(f"New best model saved: {best_model_id} with score {best_score}")

            # Update the model pool with the evaluation result
            model_pool.update_model_score(latest['id'], avg_reward)

            time.sleep(self.config['eval_interval'])
