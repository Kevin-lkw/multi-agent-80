import torch
import numpy as np
from model import get_model  # Assuming this is your ResNet18 model definition
from env import TractorEnv  # Assuming this is your TractorEnv definition
from wrapper import cardWrapper  # Assuming this is your cardWrapper definition
import random

# Configuration parameters
config = {
    'device': 'cuda',
    'model_1_path': 'best_model/', ## checkpoint or best_model
    'model_2_path': 'best_model/',
    'model_1_name': 'best_model_78', ## model_ or best_model_
    'model_2_name': 'best_model_0',
    'batch_size': 2048,
}

model_1 = get_model().to(config['device'])
model_2 = get_model().to(config['device'])
envs = [TractorEnv() for _ in range(config['batch_size'])]
wrapper = cardWrapper()

state_dict_1 = torch.load(f"{config['model_1_path']}{config['model_1_name']}.pt", map_location=config['device'])
model_1.load_state_dict(state_dict_1)

# state_dict_2 = torch.load(f"{config['model_2_path']}{config['model_2_name']}.pt", map_location=config['device'])
# model_2.load_state_dict(state_dict_2)

total_reward = 0

# Initialize environments
obs_batch = []
action_options_batch = []
seq_history_batch = []
envs = [TractorEnv() for _ in range(config['batch_size'])]
for env in envs:
    obs, action_options = env.reset(major='r')
    obs_batch.append(obs)
    action_options_batch.append(action_options)
    seq_history_batch.append([])

done_batch = [False] * config['batch_size']
episode_rewards = [{} for _ in range(config['eval_batch_size'])]

while not all(done_batch):
    player = obs_batch[0]['id']
    obs_mat_batch, action_mask_batch, seq_mat_batch = [], [], []
    for i, (obs, action_options, seq_history) in enumerate(zip(obs_batch, action_options_batch, seq_history_batch)):
        if not done_batch[i]:
            obs_mat, action_mask, seq_mat = wrapper.obsWrap(obs, action_options, seq_history)
            obs_mat_batch.append(obs_mat)
            action_mask_batch.append(action_mask)
            seq_mat_batch.append(seq_mat)
        else:
            obs_mat_batch.append(np.zeros_like(obs_mat))  # or some other default value
            action_mask_batch.append(np.zeros_like(action_mask))  # or some other default value
            seq_mat_batch.append(np.zeros_like(seq_mat))

    obs_mat_batch = torch.tensor(np.array(obs_mat_batch), dtype=torch.float).to(config['device'])
    action_mask_batch = torch.tensor(np.array(action_mask_batch), dtype=torch.float).to(config['device'])
    seq_mat_batch = torch.tensor(np.array(seq_mat_batch), dtype=torch.float).to(config['device'])

    model_1.eval()
    model_2.eval()

    with torch.no_grad():
        logits_batch, value_batch = [], []

        state = {'observation': obs_mat_batch, 'action_mask': action_mask_batch, 'seq_mat': seq_mat_batch}
        if player % 2 == random.randint(0,1):
            logits_batch, value_batch = model_1(state)
        else:
            logits_batch, value_batch = model_2(state)

        actions_batch = [torch.distributions.Categorical(logits=logits).sample().item() for logits in logits_batch]
        values_batch = [value.item() for value in value_batch]

    # Execute actions and get new state and reward
    for i, env in enumerate(envs):
        if not done_batch[i]:
            action_cards = action_options_batch[i][actions_batch[i]]
            seq_history_batch[i].append({'player': player, 'action': action_cards})
            response = env.action_intpt(action_cards, obs_batch[i]['id'])
            next_obs, next_action_options, rewards, done = env.step(response)
            if rewards:
                for agent_name in rewards:
                    if agent_name not in episode_rewards[i]:
                        episode_rewards[i][agent_name] = []
                    episode_rewards[i][agent_name].append(rewards[agent_name])
            obs_batch[i] = next_obs
            action_options_batch[i] = next_action_options
            done_batch[i] = done

total_rewards = np.array([sum(rewards[env.agent_names[0]]) for rewards in episode_rewards])
total_wins = np.array([sum(rewards[env.agent_names[0]])>0 for rewards in episode_rewards])
avg_reward = np.mean(total_rewards)
win_rate = np.mean(total_wins)
print(f"Average Reward: {avg_reward}\n Win Rate: {win_rate} ")
