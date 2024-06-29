from mcts import MCTS
from env import TractorEnv
import numpy as np
env = TractorEnv()
search_engine = MCTS(simulate_number=1000)

obs, action_options = env.reset(major='r')

playing_rounds = 0
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
l1=[]
l2=[]
for i in range(5):
    s1 = MCTS(simulate_number=1000, regular_reward=True)
    # s2 = MCTS(simulate_number=100, regular_reward=False)
    reward1 = s1.search(data)
    # reward2 = s2.search(data)
    l1.append(reward1)
    # l2.append(reward2)
print(l1)
# print(l2)
#output the variance of l1 and l2
print("variance of l1",np.var(l1))
# print("variance of l2",np.var(l2))
# print("with reg",reward1)
# print("without reg",reward2)