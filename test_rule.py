import torch
import numpy as np
from model import get_model  # Assuming this is your ResNet18 model definition
from env import TractorEnv  # Assuming this is your TractorEnv definition
from wrapper import cardWrapper  # Assuming this is your cardWrapper definition
import random
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--model_path", type=str, default='best_LSTM_model/')
parser.add_argument('--model', type=str, default='best_model_1472')

# Configuration parameters
args = parser.parse_args()
config = {
    'device': args.device,
    'model_1_path': args.model_path, ## checkpoint or best_model
    'model_1_name': args.model,
    'batch_size': 1,
}

model_1 = get_model().to(config['device'])
wrapper = cardWrapper()

state_dict_1 = torch.load(f"{config['model_1_path']}{config['model_1_name']}.pt", map_location=config['device'], weights_only=True)
model_1.load_state_dict(state_dict_1)

def evaluate_single_game_with_length_tracking():
    env = TractorEnv()
    obs, action_options, rule_based_action_options = env.reset(banker_pos=np.random.randint(2), major='r', rule_based=True)
    seq_history = []
    done = False
    episode_rewards = {}
    
    # To track lengths
    action_options_lengths = []
    rule_based_action_options_lengths = []

    while not done:
        player = obs['id']
        obs_mat, action_mask, seq_mat = wrapper.obsWrap(obs, action_options, seq_history, None)

        obs_mat = torch.tensor(obs_mat, dtype=torch.float).unsqueeze(0).to(config['device'])
        action_mask = torch.tensor(action_mask, dtype=torch.float).unsqueeze(0).to(config['device'])
        seq_mat = torch.tensor(seq_mat, dtype=torch.float).unsqueeze(0).to(config['device'])

        model_1.eval()

        # Record lengths for this step
        action_options_lengths.append(len(action_options))
        rule_based_action_options_lengths.append(len(rule_based_action_options))

        with torch.no_grad():
            if player % 2 == 0:
                logits_batch = torch.randn((1, 54), device=config['device'])
                mask = action_mask.float()
                inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
                logits_batch = logits_batch + inf_mask
                action = torch.distributions.Categorical(logits=logits_batch[0]).sample().item()
            else:
                action = random.choice(range(len(rule_based_action_options)))

        if player % 2 == 0:
            action_cards = action_options[action]
        else:
            action_cards = rule_based_action_options[action]

        seq_history.append({'player': player, 'action': action_cards})
        response = env.action_intpt(action_cards, obs['id'])
        next_obs, next_action_options, rewards, done, next_rule_based_action_options = env.step(response, rule_based=True)

        if rewards:
            for agent_name in rewards:
                if agent_name not in episode_rewards:
                    episode_rewards[agent_name] = []
                episode_rewards[agent_name].append(rewards[agent_name])

        obs = next_obs
        action_options = next_action_options
        rule_based_action_options = next_rule_based_action_options

    total_reward = sum(episode_rewards[env.agent_names[0]])
    win = total_reward > 0

    # Return both the reward, win result, and tracked lengths
    return total_reward, win, action_options_lengths, rule_based_action_options_lengths

def evaluate_n_games_with_length_tracking(n):
    total_rewards = []
    wins = []
    action_lengths_all = []
    rule_based_lengths_all = []

    for _ in range(n):
        reward, win, action_lengths, rule_based_lengths = evaluate_single_game_with_length_tracking()
        total_rewards.append(reward)
        wins.append(win)
        action_lengths_all.append(action_lengths)
        rule_based_lengths_all.append(rule_based_lengths)

    avg_reward = np.mean(total_rewards)
    win_rate = np.mean(wins)

    # Compute per-game average lengths
    avg_action_lengths = [np.mean(lengths) for lengths in action_lengths_all]
    avg_rule_based_lengths = [np.mean(lengths) for lengths in rule_based_lengths_all]

    # Compute overall averages
    overall_avg_action_length = np.mean(avg_action_lengths)
    overall_avg_rule_based_length = np.mean(avg_rule_based_lengths)

    print(f"Average Reward: {avg_reward}")
    print(f"Win Rate: {win_rate}")
    print(f"Overall Average Action Options Length: {overall_avg_action_length}")
    print(f"Overall Average Rule-Based Action Options Length: {overall_avg_rule_based_length}")
    # Calculate average lengths for each step across all games
    print("\nAverage lengths per step across all games:")
    max_steps = max(len(lengths) for lengths in action_lengths_all)
    
    for step in range(max_steps):
        # Get lengths for this step from all games that reached this step
        step_action_lengths = [lengths[step] for lengths in action_lengths_all if step < len(lengths)]
        step_rule_lengths = [lengths[step] for lengths in rule_based_lengths_all if step < len(lengths)]
        
        # Calculate averages
        avg_action_len = np.mean(step_action_lengths)
        avg_rule_len = np.mean(step_rule_lengths)
        
        print(f"Step {step + 1}: Average AI Options: {avg_action_len:.2f}, Average Rule-based Options: {avg_rule_len:.2f}")
    # Return detailed results if needed
    return {
        'avg_reward': avg_reward,
        'win_rate': win_rate,
        'overall_avg_action_length': overall_avg_action_length,
        'overall_avg_rule_based_length': overall_avg_rule_based_length,
        'action_lengths_per_game': action_lengths_all,
        'rule_based_lengths_per_game': rule_based_lengths_all,
    }

# Example usage
results = evaluate_n_games_with_length_tracking(1000)
