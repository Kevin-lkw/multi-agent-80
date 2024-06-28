from multiprocessing import Process
import time
import torch
import numpy as np
from model_pool import ModelPoolClient
from model import get_model
from env import TractorEnv
from wrapper import cardWrapper
import random

class Evaluator(Process):
    def __init__(self, config):
        super(Evaluator, self).__init__()
        self.config = config

    def run(self):
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
                best_state_dict = torch.load(f"{self.config['best_model_path']}best_model_{best_model_id}.pt")
                best_model.load_state_dict(best_state_dict)

            total_reward = 0

            # Initialize environments
            obs_batch = []
            action_options_batch = []
            seq_history_batch = []
            envs = [TractorEnv() for _ in range(self.config['eval_batch_size'])]
            for env in envs:
                obs, action_options = env.reset(major='r')
                obs_batch.append(obs)
                action_options_batch.append(action_options)
                seq_history_batch.append([])

            done_batch = [False] * self.config['eval_batch_size']
            episode_rewards = [{} for _ in range(self.config['eval_batch_size'])]

            while not all(done_batch):
                player = obs_batch[0]['id']
                obs_mat_batch, action_mask_batch, seq_mat_batch = [], [], []
                for i, (obs, action_options, seq_history) in enumerate(zip(obs_batch, action_options_batch, seq_history_batch)):
                    if not done_batch[i]:
                        obs_mat, action_mask, seq_mat = self.wrapper.obsWrap(obs, action_options, seq_history)
                        obs_mat_batch.append(obs_mat)
                        action_mask_batch.append(action_mask)
                        seq_mat_batch.append(seq_mat)
                    else:
                        obs_mat_batch.append(np.zeros_like(obs_mat))  # or some other default value
                        action_mask_batch.append(np.zeros_like(action_mask))  # or some other default value
                        seq_mat_batch.append(np.zeros_like(seq_mat))

                obs_mat_batch = torch.tensor(np.array(obs_mat_batch), dtype=torch.float).to(self.config['device'])
                action_mask_batch = torch.tensor(np.array(action_mask_batch), dtype=torch.float).to(self.config['device'])
                seq_mat_batch = torch.tensor(np.array(seq_mat_batch), dtype=torch.float).to(self.config['device'])

                current_model.eval()
                best_model.eval()

                with torch.no_grad():
                    logits_batch, value_batch = [], []

                    state = {'observation': obs_mat_batch, 'action_mask': action_mask_batch, 'seq_mat': seq_mat_batch}
                    if player % 2 == random.randint(0,1):
                        logits_batch, value_batch = current_model(state)
                    else:
                        logits_batch, value_batch = best_model(state)

                    actions_batch = [torch.distributions.Categorical(logits=logits).sample().item() for logits in logits_batch]
                    values_batch = [value.item() for value in value_batch]

                # Execute actions and get new state and reward
                for i, env in enumerate(envs):
                    if not done_batch[i]:
                        action_cards = action_options_batch[i][actions_batch[i]]
                        seq_history_batch[i].append({'player': player, 'action': action_options})
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
            avg_reward = np.mean(total_rewards)
            print(f"Model {latest['id']} evaluation reward: {avg_reward}")

            if avg_reward > 0:
                best_score = avg_reward
                best_model_id = latest['id']
                torch.save(current_state_dict, f"{self.config['best_model_path']}best_model_{best_model_id}.pt")
                print(f"New best model saved: {best_model_id} with score {best_score}")

            # Update the model pool with the evaluation result
            model_pool.update_model_score(latest['id'], avg_reward)

            time.sleep(self.config['eval_interval'])
