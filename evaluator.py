import time
import torch
import numpy as np
from model_pool import ModelPoolClient
from model import get_model
from env import TractorEnv
from wrapper import cardWrapper

class Evaluator:
    def __init__(self, config):
        self.config = config

    def evaluate_single_game(self, current_model, best_model, env):
        obs, action_options = env.reset(major='r')
        seq_history = []

        done = False
        episode_rewards = {}

        while not done:
            player = obs['id']
            obs_mat, action_mask, seq_mat = self.wrapper.obsWrap(obs, action_options, seq_history)

            obs_mat = torch.tensor(obs_mat, dtype=torch.float).unsqueeze(0).to(self.config['device'])
            action_mask = torch.tensor(action_mask, dtype=torch.float).unsqueeze(0).to(self.config['device'])
            seq_mat = torch.tensor(seq_mat, dtype=torch.float).unsqueeze(0).to(self.config['device'])

            current_model.eval()
            best_model.eval()

            with torch.no_grad():
                state = {'observation': obs_mat, 'action_mask': action_mask, 'seq_mat': seq_mat}
                if player % 2 == 0:
                    logits = current_model(state)
                else:
                    logits = best_model(state)

                action = torch.distributions.Categorical(logits=logits).sample().item()

            # Execute action and get new state and reward
            action_cards = action_options[action]
            seq_history.append({'player': player, 'action': action_cards})
            response = env.action_intpt(action_cards, obs['id'])
            next_obs, next_action_options, rewards, done = env.step(response)

            if rewards:
                for agent_name in rewards:
                    if agent_name not in episode_rewards:
                        episode_rewards[agent_name] = []
                    episode_rewards[agent_name].append(rewards[agent_name])

            obs = next_obs
            action_options = next_action_options

        total_reward = sum(episode_rewards[env.agent_names[0]])
        win = total_reward > 0
        return total_reward, win

    def evaluate(self):
        model_pool = ModelPoolClient(self.config['model_pool_name'])
        current_model = get_model().to(self.config['device'])
        best_model = get_model().to(self.config['device'])
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
                best_state_dict = torch.load(f"{self.config['best_model_path']}best_model_{best_model_id}.pt", weights_only=True)
                best_model.load_state_dict(best_state_dict)

            total_rewards = []
            wins = []

            for _ in range(self.config['num_evaluations']):
                reward, win = self.evaluate_single_game(current_model, best_model, env)
                total_rewards.append(reward)
                wins.append(win)

            avg_reward = np.mean(total_rewards)
            win_rate = np.mean(wins)

            print(f"Model {latest['id']} evaluation results: Avg Reward = {avg_reward}, Win Rate = {win_rate}")

            if avg_reward > 0:
                best_model_id = latest['id']
                torch.save(current_state_dict, f"{self.config['best_model_path']}best_model_{best_model_id}.pt")
                print(f"New best model saved: {best_model_id} with score {best_score}")

            time.sleep(self.config['eval_interval'])
