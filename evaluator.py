from multiprocessing import Process
import time
import torch
from model_pool import ModelPoolClient
from model import CNNModel
from env import TractorEnv
from wrapper import cardWrapper
import matplotlib.pyplot as plt

class Evaluator(Process):
    def __init__(self, config, shared_list):
        super(Evaluator, self).__init__()
        self.config = config
        self.shared_list = shared_list

    def run(self):
        model_pool = ModelPoolClient(self.config['model_pool_name'])
        model = CNNModel()
        env = TractorEnv()
        wrapper = cardWrapper()

        best_score = float('-inf')
        best_model_id = None

        while True:
            latest = model_pool.get_latest_model()
            state_dict = model_pool.load_model(latest)
            model.load_state_dict(state_dict)
            model.eval()

            total_score = 0
            for _ in range(self.config['eval_episodes']):
                obs, action_options = env.reset()
                done = False
                while not done:
                    player = obs['id']
                    obs_mat, action_mask = wrapper.obsWrap(obs, action_options)
                    state = {
                        'observation': torch.tensor(obs_mat, dtype=torch.float).unsqueeze(0),
                        'action_mask': torch.tensor(action_mask, dtype=torch.float).unsqueeze(0)
                    }
                    with torch.no_grad():
                        logits, _ = model(state)
                        action = torch.argmax(logits, dim=1).item()  
                    
                    action_cards = action_options[action]
                    response = env.action_intpt(action_cards, player)
                    obs, action_options, reward, done = env.step(response)
                    if reward:
                        total_score += reward[f'player_{player}']

            avg_score = total_score / self.config['eval_episodes']
            print(f"Model {latest['id']} evaluation score: {avg_score}")

            self.shared_list.append((latest['id'], avg_score))

            if avg_score > best_score:
                best_score = avg_score
                best_model_id = latest['id']
                torch.save(state_dict, f"{self.config['best_model_path']}best_model_{best_model_id}.pt")
                print(f"New best model saved: {best_model_id} with score {best_score}")

            time.sleep(self.config['eval_interval'])

def plot_training_curve(eval_results, save_path):
    model_ids, scores = zip(*eval_results)
    
    plt.figure(figsize=(12, 6))
    plt.plot(model_ids, scores)
    plt.title('Model Performance over Training')
    plt.xlabel('Model ID')
    plt.ylabel('Average Score')
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()