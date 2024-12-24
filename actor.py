from multiprocessing import Process
import numpy as np
import torch

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolClient
from env import TractorEnv
from model import get_model, get_perfect_model
import copy
from wrapper import cardWrapper
from mcts import MCTS

class Actor(Process):
    
    def __init__(self, config, replay_buffer):
        super(Actor, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.name = config.get('name', 'Actor-?')
        
    def run(self):
        torch.set_num_threads(1)
    
        # connect to model pool
        model_pool = ModelPoolClient(self.config['model_pool_name'])
        mdoel_pool_value = ModelPoolClient('model_pool_value')

        # create network model
        model = get_model()
        value_model = get_perfect_model()

        # load initial model
        version = model_pool.get_latest_model()
        state_dict = model_pool.load_model(version)
        model.load_state_dict(state_dict)

        value_version = mdoel_pool_value.get_latest_model()
        value_state_dict = model_pool.load_model(value_version)
        value_model.load_state_dict(value_state_dict)
        
        # collect data
        env = TractorEnv()
        self.wrapper = cardWrapper()
        policies = {player : model for player in env.agent_names} # all four players use the latest model
        
        for episode in range(self.config['episodes_per_actor']):
            # update model
            latest = model_pool.get_latest_model()
            if latest['id'] > version['id']:
                state_dict = model_pool.load_model(latest)
                model.load_state_dict(state_dict)
                version = latest
            
            # run one episode and collect data
            print(111)
            obs, action_options, rule_based_action_options = env.reset(major='r',rule_based=True)
            episode_data = {agent_name: {
                'state' : {
                    'observation': [],
                    'action_mask': [],
                    'seq_mat': [],
                },
                'perfect_state' : {
                    'perfect_observation': [],
                    'action_mask': [],
                    'seq_mat': [],
                },
                'action' : [],
                'reward' : [],
                'value' : [],
                'per_info': [],
            } for agent_name in env.agent_names}
            done = False
            seq_history = []
            while not done:
                state = {}
                per_state = {}
                player = obs['id']
                perfect_info = env.pack_data()
                player_decks = perfect_info['player_decks']
                
                agent_name = env.agent_names[player]
                agent_data = episode_data[agent_name]
                agent_data['per_info'].append(perfect_info)

                # get policy input
                obs_mat, action_mask, seq_mat = self.wrapper.obsWrap(obs, rule_based_action_options, seq_history)
                agent_data['state']['observation'].append(obs_mat)
                agent_data['state']['action_mask'].append(action_mask)
                agent_data['state']['seq_mat'].append(seq_mat)
                state['observation'] = torch.tensor(obs_mat, dtype = torch.float).unsqueeze(0)
                state['action_mask'] = torch.tensor(action_mask, dtype = torch.float).unsqueeze(0)
                state['seq_mat'] = torch.tensor(seq_mat, dtype = torch.float).unsqueeze(0)

                # get value input
                per_obs_mat, action_mask, seq_mat = self.wrapper.obsWrap(obs, rule_based_action_options, seq_history, player_decks, perfect=True)
                agent_data['perfect_state']['perfect_observation'].append(per_obs_mat)
                agent_data['perfect_state']['action_mask'].append(action_mask)
                agent_data['perfect_state']['seq_mat'].append(seq_mat)
                per_state['observation'] = torch.tensor(per_obs_mat, dtype = torch.float).unsqueeze(0)
                per_state['action_mask'] = torch.tensor(action_mask, dtype = torch.float).unsqueeze(0)
                per_state['seq_mat'] = torch.tensor(seq_mat, dtype = torch.float).unsqueeze(0)

                model.train(False) # Batch Norm inference mode
                value_model.train(False) 

                with torch.no_grad():
                    logits = model(state)
                    value = value_model(per_state)
                    action_dist = torch.distributions.Categorical(logits = logits)
                    action = action_dist.sample().item()
                    value = value.item()
                    
                agent_data['action'].append(action)
                agent_data['value'].append(value)


                # interpreting actions
                # action_cards = action_options[action]
                action_cards = rule_based_action_options[action]
                response = env.action_intpt(action_cards, player)
                # print(player,action,action_cards,response)
                # response w.r.t. {'player': 2, 'action': [int[0,108)]}, where int is the list id of card
                seq_history.append({'player':player, 'action': action_cards}) 

                # interact with env
                next_obs, action_options, rewards, done, rule_based_action_options = env.step(response,rule_based= True)
                if rewards:
                    # rewards are added per four moves (1 move for each player) on all four players
                    for agent_name in rewards: 
                        episode_data[agent_name]['reward'].append(rewards[agent_name])
                obs = next_obs
            #print(self.name, 'Episode', episode, 'Model', latest['id'], 'Reward', rewards)

            
            # postprocessing episode data for each agent
            for agent_name, agent_data in episode_data.items():
                if len(agent_data['action']) < len(agent_data['reward']):
                    agent_data['reward'].pop(0)
                    print("after reward:", agent_data['reward'])
                per_obs = np.stack(agent_data['perfect_state']['perfect_observation'])
                obs = np.stack(agent_data['state']['observation'])
                mask = np.stack(agent_data['state']['action_mask'])
                seq_mat = copy.deepcopy(agent_data['state']['seq_mat'])
                perfect_info = copy.deepcopy(agent_data['per_info'])
                actions = np.array(agent_data['action'], dtype = np.int64)
                rewards = np.array(agent_data['reward'], dtype = np.float32)
                values = np.array(agent_data['value'], dtype = np.float32)
                next_values = np.array(agent_data['value'][1:] + [0], dtype = np.float32)
                
                td_target = rewards/10.0 + next_values * self.config['gamma']
                td_delta = td_target - values
                #search_engine = MCTS(simulate_number=50)
                #mcts_target = np.array([search_engine.search(per_info)  for per_info in perfect_info], dtype = np.float32)
                #mcts_target = np.array([np.random.randint(0,1)  for per_info in perfect_info], dtype = np.float32)
                advs = []
                adv = 0
                for delta in td_delta[::-1]:
                    adv = self.config['gamma'] * self.config['lambda'] * adv + delta
                    advs.append(adv) # GAE
                advs.reverse()
                advantages = np.array(advs, dtype = np.float32)
                # send samples to replay_buffer (per agent)
                self.replay_buffer.push({
                    'state': {
                        'observation': obs,
                        'action_mask': mask,
                        'seq_mat':seq_mat,
                    },
                    'perfect_state' : {
                    'perfect_observation': per_obs,
                    'action_mask': mask,
                    'seq_mat': seq_mat,
                    },
                    'action': actions,
                    'adv': advantages,
                    'target': td_target,
                    'per_info': perfect_info,
                })
        
        