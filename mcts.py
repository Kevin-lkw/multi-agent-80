from env import TractorEnv
from mvGen import move_generator
import numpy as np
import copy
# this function takes deck,player,....
# and output the max reward player can get.
#-----------------#
# the pocress is as follows:
# 1. mv generation: _get_action_options in env.py,
# 2. get all actions the current player can take
# 3. use UCB to select the action
# 4. recursively calculate the node
class MCTS(TractorEnv):
    def __init__(self, simulate_number = 100, regular_reward=True):
        super(MCTS, self).__init__()
        self.simulate_number = simulate_number
        self.regular_reward = regular_reward
        # for same player deck, the action_options generate is the same
        # so we can use number to represent action
        self.visits = {}
        self.rewards = {}

    def load(self, packed_data):
        self.point_order = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']
        self.Major = ['jo', 'Jo']
        self.unpack(packed_data)

        self._setMajor()
        self.mv_gen = move_generator(self.level, self.major)
        self.score = 0 # score of the current game, unimportant in MCTS

        self.reward = None # reset the reward
        self.done = False
        # Do the first round
    
    def _get_full_obs(self):
        # Returning the full observation, including other player's card
        # use full_obs as state

        # there might be a problem: different order are considered different state.
        # dictionay can not be used as key
        obs = {
            "id": self.player,
            "deck": [[self._id2name(p) for p in self.player_decks[i]] for i in range(4)],
            "history": [[self._id2name(p) for p in move] for move in self.history],
            # "played": [[self._id2name(p) for p in move] for move in self.played_cards]
            # played card will not infernce the state
        }
        # tuple form to enable hash
        obs = (
            self.player,
            tuple(tuple(self._id2name(p) for p in self.player_decks[i]) for i in range(4)),
            tuple(tuple(self._id2name(p) for p in move) for move in self.history)
        )
        return obs
    def _select_action(self, action_options, state, player):
        # UCB algorithm
        # action_options: a list of actions
        # state: the current state
        # action: the selected action

        action = None
        max_ucb = float('-inf')
        unvisit = []
        for a in range(len(action_options)):
            if (state,a) not in self.visits:
                unvisit.append(a)
        if len(unvisit) > 0:
            return np.random.choice(unvisit)
            # add some randomness to check stablity
        for a in range(len(action_options)):
            reward_dic = self.rewards.get((state, a), {i:0 for i in self.agent_names})
            reward_val = reward_dic[self.agent_names[player]]
            if self.regular_reward:
                reward = reward_val // 5
            else:
                reward = reward_val
            visits_a = self.visits.get((state, a), 0)
            visits = self.visits.get(state, 0)
            if visits_a == 0:
                return a
            ucb = reward / visits_a + 1.41 * (2 * np.log(visits) / visits_a) ** 0.5
            if ucb > max_ucb:
                max_ucb = ucb
                action = a
        return action
    
    def update(self, state, action, rewards):

        self.visits[(state, action)] = self.visits.get((state, action), 0) + 1
        if self.rewards.get((state, action), None) is None:
            self.rewards[(state, action)] = {i:0 for i in self.agent_names}
        for key,value in rewards.items():
            self.rewards[(state, action)][key] += value
        self.visits[state] = self.visits.get(state, 0) + 1
    
    def simulate(self, curr_player, action_options):
        # get the action options
        state = self._get_full_obs()
        # choose the action using UCB algorithm
        action = self._select_action(action_options, state, curr_player) # the # of action in action_options
        response = self.action_intpt(action_options[action], curr_player)
        # interact with env
        next_obs, next_action_options, rewards, done = self.step(response)
        if done:
            # print("done!")
            # print("rewards",rewards)
            return rewards
        next_player = (curr_player + 1) % 4
        if len(self.history) == 4: # finishing a round
            winner = self._checkWinner(curr_player)
            next_player = winner
        all_rewards = self.simulate(next_player, next_action_options)
        if rewards is not None:
            for key,value in rewards.items():
                all_rewards[key] += value
        # self.print_state()
        # print("state ",state)
        # print("action ",action)
        # print("rewards ",rewards,all_rewards)
        self.update(state, action, all_rewards)
        ### !! probably wrong! what's is the push stack mechanism in python?
        return all_rewards
    
    def search(self,packed_data):
        for i in range(self.simulate_number):
            # print("searching",i)
            self.load(packed_data)
            action_options = self._get_action_options(self.player)
            #here player is the one we want to maximize reward
            self.simulate(self.player, action_options)
        self.load(packed_data)
        state = self._get_full_obs()
        action_options = self._get_action_options(self.player)
        max_reward = float('-inf')
        for i in range(len(action_options)):
            reward_dict = self.rewards.get((state, i), {j:0 for j in self.agent_names})
            # print("reward_dict",reward_dict)
            # print("vis time",self.visits.get((state, i), 1))
            max_reward = max(max_reward,
                reward_dict[self.agent_names[self.player]] / self.visits.get((state, i), 1))
        return max_reward
        
    
    def unpack(self,packed_data):
        self.level = packed_data['level']
        self.major = packed_data['major']
        self.covered_card = copy.deepcopy(packed_data['covered_card'])
        self.played_cards = copy.deepcopy(packed_data['played_cards'])
        self.player_decks = copy.deepcopy(packed_data['player_decks'])
        self.banker_pos = packed_data['banker_pos']
        self.history = copy.deepcopy(packed_data['history'])
        self.player = packed_data['player']
        self.round = packed_data['round']
        self.agent_names = packed_data['agent_names']

    def print_state(self):
        print("level",self.level)
        print("major",self.major)
        print("player_decks",self.player_decks)
        print("played_cards",self.played_cards)
        print("history",self.history)
        print("player",self.player)
        print("round",self.round)
        print("agent_names",self.agent_names)