from env import TractorEnv
from mvGen import move_generator
import numpy as np
# this function takes deck,player,....
# and output the max reward player can get.
#-----------------#
# the pocress is as follows:
# 1. mv generation: _get_action_options in env.py,
# 2. get all actions the current player can take
# 3. use UCB to select the action
# 4. recursively calculate the node
class MCTS(TractorEnv):
    def __init__(self, simulate_number = 1000):
        super(MCTS, self).__init__()
        self.simulate_number = simulate_number
        self.visits = {}
        self.rewards = {}
    # important status:
    # history: current round, length[1,4], only in the beginning 0 
    # player_decks: current player's deck
    # played_card: the player card's 

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
        obs = {
            "id": self.player,
            "deck": [[self._id2name(p) for p in self.player_decks[i]] for i in range(4)],
            "history": [[self._id2name(p) for p in move] for move in self.history],
            "major": self.Major,
            "played": [[self._id2name(p) for p in move] for move in self.played_cards]
        }
        return obs
    def _select_action(self, action_options, state):
        # UCB algorithm
        # action_options: a list of actions
        # state: the current state
        # action: the selected action
        raise NotImplemented
        action = None
        max_ucb = float('-inf')
        for a in action_options:
            if (state, a) not in self.visits:
                return a
            ucb = self.rewards[(state, a)] / self.visits[(state, a)] + 1.41 * (2 * np.log(self.visits[state]) / self.visits[(state, a)]) ** 0.5
            if ucb > max_ucb:
                max_ucb = ucb
                action = a
        return
    
    def update(self, state, action, rewards):
        self.visits[(state, action)] = self.visits.get((state, action), 0) + 1
        self.rewards[(state, action)] = self.rewards.get((state, action), 0) + rewards
        self.visits[state] = self.visits.get(state, 0) + 1
    
    def simulate(self, max_player, curr_player, action_options):
        # get the action options
        state = self._get_full_obs()
        # choose the action using UCB algorithm
        action = self._select_action(action_options, state)
        response = self.action_intpt(action, curr_player)
        # interact with env
        next_obs, next_action_options, rewards, done = self.step(response)
        if done:
            return rewards[max_player]
        next_player = (curr_player + 1) % 4
        if len(self.history) == 4: # finishing a round
            winner = self._checkWinner(curr_player)
            next_player = winner
        rewards = self.simulate(max_player, next_player, next_action_options)
        self.update(state, action, rewards)
        return rewards
    
    def search(self,packed_data):
        for _ in self.simulate_number:
            self.load(packed_data)
            action_options = self._get_action_options()
            #here player is the one we want to maximize reward
            self.simulate(self.player, self.player, action_options)
        self.load(packed_data)
        action_options = self._get_action_options()
        raise NotImplemented
        return self.reward
        
    
    def unpack(self,packed_data):
        self.level = packed_data['level']
        self.major = packed_data['major']
        self.covered_card = packed_data['covered_card']
        self.played_cards = packed_data['played_cards']
        self.banker_pos = packed_data['banker_pos']
        self.history = packed_data['history']
        self.played_cards = packed_data['played_cards']
        self.player = packed_data['player']
