import numpy as np
import torch

class cardWrapper:
    def __init__(self, suit_sequence=['s', 'h', 'c', 'd'], point_sequence = ['2','3','4','5','6','7','8','9','0','J','Q','K','A']):
        self.card_scale = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K']
        self.suit_sequence = suit_sequence
        self.point_sequence = point_sequence
        self.J_pos = self.suit_sequence.index('h')
        self.j_pos = self.suit_sequence.index('s')
        self.suit_set = ['s','h','c','d']
        
    def name2pos(self, cardname):
        if cardname[0] == "J":
            return (self.J_pos, 13)
        if cardname[0] == "j":
            return (self.j_pos, 13)
        pos = (self.suit_sequence.index(cardname[0]), self.point_sequence.index(cardname[1]))
        return pos
    
    def pos2name(self, cardpos):
        if cardpos[1] == 13:
            if cardpos[0] == self.j_pos:
                return "jo"
            if cardpos[0] == self.J_pos:
                return "Jo"
            else:
                raise "Card not exists."
        
        return self.suit_sequence[cardpos[0]] + self.point_sequence[cardpos[1]]
    
    # adding cards to a cardset 
    def add_card(self, cardset: np.array, cards): 
    # cardset: np.array(2,4,14)
    # cards: list[str(2)], cardnames.
        for card in cards:
            card_pos = self.name2pos(card)
            if cardset[0, card_pos[0], card_pos[1]] == 0:
                cardset[0, card_pos[0], card_pos[1]] = 1
            elif cardset[1, card_pos[0], card_pos[1]] == 0:
                cardset[1, card_pos[0], card_pos[1]] = 1    
            else:
                raise "More than two cards with same suits and points. Please recheck."

        return cardset
    
    # removing cards from cardset
    def remove_card(self, cardset: np.array, cards):
        for card in cards:
            card_pos = self.name2pos(card)
            if cardset[1, card_pos[0], card_pos[1]] != 0:
                cardset[1, card_pos[0], card_pos[1]] = 0
            elif cardset[0, card_pos[0], card_pos[1]] != 0:
                cardset[0, card_pos[0], card_pos[1]] = 0
            else:
                raise "Card not in cardset! Please recheck."

        return cardset
    
    # From cardset to cardnames
    def Unwrap(self, cardset): 
        cards = []
        card_poses = np.nonzero(cardset)
        for i in range(card_poses[0].size):
            card_name = self.pos2name((card_poses[1][i], card_poses[2][i]))
            cards.append(card_name)
        
        return cards
    
    def _id2name(self, card_id): # card_id: int[0, 107]
        # Locate in 1 single deck
        NumInDeck = card_id % 54
        # joker and Joker:
        if NumInDeck == 52:
            return "jo"
        if NumInDeck == 53:
            return "Jo"
        # Normal cards:
        pokernumber = self.card_scale[NumInDeck // 4]
        pokersuit = self.suit_set[NumInDeck % 4]
        return pokersuit + pokernumber

    def obsWrap(self, obs, options, seq_history, player_decks= None, perfect=False):
        '''
        Wrapping the observation and craft the action_mask
        obs: raw obs from env
        '''
        id = obs['id']

        

        if perfect:
            other_deck_mat = np.zeros((6,4,14))
            for i in range(3):
                player = (i-id) %4
                if player:
                    player -= 1
                    player_decks_id = player_decks[i] # 取出player1的局面
                    player_decks_name = []
                    for pos in player_decks_id:
                        pos = self._id2name(pos)
                        player_decks_name.append(pos)
                    self.add_card(other_deck_mat[player*2:(player +1)*2], player_decks_name) # 放到相对id的位置

        



        seq_mat = []
        for history_response in seq_history:
            player = (history_response['player'] - id) % 4  # 计算相对当前player的id
            history_action = history_response['action']
            player_mat = np.ones((1, 4, 14)) * player
            history_action_mat = np.zeros((2, 4, 14))
            self.add_card(history_action_mat, history_action)
            history_response_mat = np.concatenate((player_mat, history_action_mat), axis=0)  # 3*4*14
            seq_mat.append(history_response_mat)
        seq_mat = np.array(seq_mat)

        batch_size = 4

        batch_shape = (batch_size * 3, 4, 14)

        num_batches = len(seq_mat) // batch_size

        seq_mat_padded = np.zeros((24, *batch_shape))

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(seq_mat))
            batch = seq_mat[start_idx:end_idx]
            
            if len(batch) < batch_size:
                break
            
            batch = batch.reshape(batch_shape)
            
            seq_mat_padded[24 - num_batches + i] = batch

        major_mat = np.zeros((2, 4, 14))
        deck_mat = np.zeros((2, 4, 14))
        hist_mat = np.zeros((8, 4, 14))  # Holding no more than 4 sets of cards
        played_mat = np.zeros((8, 4, 14))
        option_mat = np.zeros((108, 4, 14))

        self.add_card(major_mat, obs['major'])
        self.add_card(deck_mat, obs['deck'])
        for i in range(len(obs['history'])):
            self.add_card(hist_mat[i * 2:(i + 1) * 2], obs['history'][i])
        played_cards = obs['played'][id:] + obs['played'][:id]
        for i in range(len(played_cards)):
            self.add_card(played_mat[i * 2:(i + 1) * 2], played_cards[i])
        for i in range(len(options)):
            if i * 2 >= option_mat.shape[0]:
                break
            self.add_card(option_mat[i * 2:(i + 1) * 2], options[i])

        action_mask = np.zeros(54)
        action_mask[:len(options)] = 1


        if perfect:
            # (128+6)*4*14
            return np.concatenate((major_mat, deck_mat, other_deck_mat, hist_mat, played_mat, option_mat)), action_mask, seq_mat_padded
            

        return np.concatenate((major_mat, deck_mat,  hist_mat, played_mat, option_mat)), action_mask, seq_mat_padded
        
            