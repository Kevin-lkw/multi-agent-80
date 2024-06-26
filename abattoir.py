from multiprocessing import Process
import time
import torch
import numpy as np
from model_pool import ModelPoolClient
from model import CNNModel
from env import TractorEnv
from wrapper import cardWrapper

config = {
        'replay_buffer_size': 50000,
        'replay_buffer_episode': 400,
        'model_pool_size': 20,
        'model_pool_name': 'model-pool',
        'num_actors': 4,
        'episodes_per_actor': 8000,
        'gamma': 0.98,
        'lambda': 0.95,
        'min_sample': 200,
        'batch_size': 256,
        'epochs': 5,
        'clip': 0.2,
        'lr': 3e-5,
        'value_coeff': 1,
        'entropy_coeff': 0.01,
        'device': 'cpu',
        'ckpt_save_interval': 300,
        'ckpt_save_path': 'checkpoint/',
        'eval_interval': 5,  # Sleep 5 seconds
        'plot_save_path': 'training_curve.png',
        ########################################################################
        ########################### Evaluate Config ############################
        ########################################################################
        'model_1_path': 'best_model/',
        'model_2_path': 'best_model/',
        'model_1_name':'best_model_705',
        'model_2_name':'best_model_40',
        'eval_episodes': 640,
    }



model_1 = CNNModel()
model_2 = CNNModel()
env = TractorEnv()
wrapper = cardWrapper()


state_dict_1 = torch.load(f"{config['model_1_path']}{config['model_1_name']}.pt")
model_1.load_state_dict(state_dict_1)

state_dict_2 = torch.load(f"{config['model_2_path']}{config['model_2_name']}.pt")
model_2.load_state_dict(state_dict_2)

total_reward = 0
for _ in range(config['eval_episodes']):
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
        obs_mat, action_mask = wrapper.obsWrap(obs, action_options)
        agent_data['state']['observation'].append(obs_mat)
        agent_data['state']['action_mask'].append(action_mask)
        state['observation'] = torch.tensor(obs_mat, dtype = torch.float).unsqueeze(0)
        state['action_mask'] = torch.tensor(action_mask, dtype = torch.float).unsqueeze(0)
        model_1.train(False) # Batch Norm inference mode
        model_2.train(False) # Batch Norm inference mode


        with torch.no_grad():
            if player % 2 == 0:  # Current model plays
                logits, value = model_1(state)
            else:  # Best model plays
                logits, value = model_2(state)
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
        
        td_target = rewards + next_values * config['gamma']
        td_delta = td_target - values
        advs = []
        adv = 0
        for delta in td_delta[::-1]:
            adv = config['gamma'] * config['lambda'] * adv + delta
            advs.append(adv) # GAE
        advs.reverse()
        advantages = np.array(advs, dtype = np.float32)
        # Maybe GAE is better. But for eval, I prefer rewards may bring me more interpretability.
        episode_reward[agent_name] = rewards 
    #####################################################################################
    #print(episode_reward) [  5.,   0., -10.,  10.,  30.,   0.,  10.,  10.,   0.,   5.,   5.,  5., -25.,  20.,  -5.,  -5.,  10.,   5., -10.,  10.,  10.,  -3.] Why 3 here ?
    total_reward += np.sum(episode_reward[env.agent_names[0]])



avg_reward = total_reward / config['eval_episodes']

print(avg_reward)



