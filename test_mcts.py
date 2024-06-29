from mcts import MCTS
from env import TractorEnv
import numpy as np
env = TractorEnv()
search_engine = MCTS()

obs, action_options = env.reset(major='r')

playing_rounds = 50
print("playing rounds",playing_rounds)
for _ in range(playing_rounds):
    action = np.random.choice(len(action_options))
    response = env.action_intpt(action_options[action], obs['id'])
    obs, action_options, rewards, done = env.step(response)
    if done:
        exit()
data = env.pack_data()
for key,value in data.items():
    # print(key, value)
    if key == 'covered_card':
        value = [env._id2name(i) for i in value]
    if key == 'player_decks' or key == 'played_cards' or key == 'history':
        value = [[env._id2name(i) for i in j] for j in value]
    print(key, value)
data = env.pack_data()
reward = search_engine.search(data)
print(reward)