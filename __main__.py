import json
import os
from model import CNNModel
import torch
from wrapper import cardWrapper
from mvGen import move_generator
import numpy as np
from collections import Counter

cardscale = ['A','2','3','4','5','6','7','8','9','0','J','Q','K']
suitset = ['s','h','c','d']
Major = ['jo', 'Jo']
pointorder = ['2','3','4','5','6','7','8','9','0','J','Q','K','A']

def setMajor(major, level):
    global Major
    if major != 'n': # 非无主
        Major = [major+point for point in pointorder if point != level] + [suit + level for suit in suitset if suit != major] + [major + level] + Major
    else: # 无主
        Major = [suit + level for suit in suitset] + Major
    pointorder.remove(level)
    
def Num2Poker(num): # num: int-[0,107]
    # Already a poker
    if type(num) is str and (num in Major or (num[0] in suitset and num[1] in cardscale)):
        return num
    # Locate in 1 single deck
    NumInDeck = num % 54
    # joker and Joker:
    if NumInDeck == 52:
        return "jo"
    if NumInDeck == 53:
        return "Jo"
    # Normal cards:
    pokernumber = cardscale[NumInDeck // 4]
    pokersuit = suitset[NumInDeck % 4]
    return pokersuit + pokernumber

def Poker2Num(poker, deck): # poker: str
    NumInDeck = -1
    if poker[0] == "j":
        NumInDeck = 52
    elif poker[0] == "J":
        NumInDeck = 53
    else:
        NumInDeck = cardscale.index(poker[1])*4 + suitset.index(poker[0])
    if NumInDeck in deck:
        return NumInDeck
    else:
        return NumInDeck + 54

def Poker2Num_seq(pokers, deck):
    id_seq = []
    deck_copy = deck + []
    for card_name in pokers:
        card_id = Poker2Num(card_name, deck_copy)
        id_seq.append(card_id)
        deck_copy.remove(card_id)
    return id_seq
    
def checkPokerType(poker, level): #poker: list[int]
    poker = [Num2Poker(p) for p in poker]
    if len(poker) == 1:
        return "single" #一张牌必定为单牌
    if len(poker) == 2:
        if poker[0] == poker[1]:
            return "pair" #同点数同花色才是对子
        else:
            return "suspect" #怀疑是甩牌
    if len(poker) % 2 == 0: #其他情况下只有偶数张牌可能是整牌型（连对）
    # 连对：每组两张；各组花色相同；各组点数在大小上连续(需排除大小王和级牌)
        count = Counter(poker)
        if "jo" in count.keys() and "Jo" in count.keys() and count['jo'] == 2 and count['Jo'] == 2:
            return "tractor"
        elif "jo" in count.keys() or "Jo" in count.keys(): # 排除大小王
            return "suspect"
        for v in count.values(): # 每组两张
            if v != 2:
                return "suspect"
        pointpos = []
        suit = list(count.keys())[0][0] # 花色相同
        for k in count.keys():
            if k[0] != suit or k[1] == level: # 排除级牌
                return "suspect"
            pointpos.append(pointorder.index(k[1])) # 点数在大小上连续
        pointpos.sort()
        for i in range(len(pointpos)-1):
            if pointpos[i+1] - pointpos[i] != 1:
                return "suspect"
        return "tractor" # 说明是拖拉机
    
    return "suspect"
weight_dict = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
               '9': 7, '0': 8, 'J': 9, 'Q': 10, 'K': 11, 'A':12}

def assignWight(poker, level):
    if poker[1] == level:
        return 13
    return weight_dict[poker[1]]
def expected_value(level):
    for i in pointorder:
        if i == level:
            sum += 13
        else:
            sum += weight_dict[i]
    return sum / 4
def call_Snatch(get_card, deck, called, snatched, level):
# get_card: new card in this turn (int)
# deck: your deck (list[int]) before getting the new card
# called & snatched: player_id, -1 if not called/snatched
# level: level
# return -> list[int]
    response = []
    # 报主策略：如果当前的主牌加权平均大于期望，报
    deck.append(get_card)
    current = expected_value(level)
    if called == -1:
        for suit in suitset:
            major_level_card = []
            level_count = 0
            value = 0
            for poker_num in deck:
                poker = Num2Poker(poker_num)
                if poker[0] != suit:
                    continue
                if poker[1] == level:
                    level_count += 1
                    major_level_card.append(poker_num)
                value += assignWight(poker, level)

            if value > current and level_count > 0:
                current = value
                response = major_level_card
    
    elif snatched == -1:
        # 反主策略：如果当前的可反花色加权平均大于主色，反
        # 不反无主
        suit = called
        current = 0
        for poker_num in deck:
            poker = Num2Poker(poker_num)
            if poker[0] != suit:
                continue
            current += assignWight(poker, level)

        for suit in suitset:
            major_level_card = []
            level_count = 0
            value = 0
            for poker_num in deck:
                poker = Num2Poker(poker_num)
                if poker[0] != suit:
                    continue
                if poker[1] == level:
                    level_count += 1
                    major_level_card.append(poker_num)
                value += assignWight(poker, level)
            if value > current and level_count == 2:
                current = value
                response = major_level_card
    return response

def cover_Pub(old_public, deck, level, major):
    deck.append(old_public)
    response = []
    # 尝试灭一色
    suit_card = {}
    for suit in suitset:
        if suit == major:
            continue
        suit_card[suit] = []
        for poker_num in deck:
            poker = Num2Poker(poker_num)
            if poker[0] == suit and poker[1] != level and poker[1] != 'A':
                suit_card[suit].append(poker_num)
    for suit,card in suit_card:
        suit_card[suit] = sorted(card, key=lambda x: weight_dict[x[1]])
        pass 
    suit_card = sorted(suit_card.items(), key=lambda x: len(x[1]))
    
    for suit, card in suit_card:
        for poker_num in card:
            response.append(poker_num)
            if len(response) == 8:
                return response
    if response < 8:
        raise NotImplementedError("No enough cards to cover the public cards")

def playCard(history, hold, played, selfid, wrapper, mv_gen, model):
    # generating obs
    obs = {
        "id": selfid,
        "deck": [Num2Poker(p) for p in hold],
        "history": [[Num2Poker(p) for p in move] for move in history],
        "major": [Num2Poker(p) for p in Major],
        "played": [[Num2Poker(p) for p in cardset] for cardset in played]
    }
    # generating action_options
    action_options = get_action_options(hold, history, selfid, mv_gen) 
    # generating state
    state = {}
    obs_mat, action_mask = wrapper.obsWrap(obs, action_options)
    state['observation'] = torch.tensor(obs_mat, dtype = torch.float).unsqueeze(0)
    state['action_mask'] = torch.tensor(action_mask, dtype = torch.float).unsqueeze(0)
    # getting actions
    action = obs2action(model, state)
    response = action_intpt(action_options[action], hold)
    return response


def get_action_options(deck, history, player, mv_gen):
    deck = [Num2Poker(p) for p in deck]
    if len(history) == 4 or len(history) == 0: # first to play
        return mv_gen.gen_all(deck)
    else:
        tgt = [Num2Poker(p) for p in history[0]]
        poktype = checkPokerType(history[0], (player-len(history))%4)
        if poktype == "single":
            return mv_gen.gen_single(deck, tgt)
        elif poktype == "pair":
            return mv_gen.gen_pair(deck, tgt)
        elif poktype == "tractor":
            return mv_gen.gen_tractor(deck, tgt)
        elif poktype == "suspect":
            return mv_gen.gen_throw(deck, tgt)    

def obs2action(model, obs):
    model.train(False) # Batch Norm inference mode
    with torch.no_grad():
        logits, value = model(obs)
        action_dist = torch.distributions.Categorical(logits = logits)
        action = action_dist.sample().item()
    return action

def action_intpt(action, deck):
    '''
    interpreting action(cardname) to response(dick{'player': int, 'action': list[int]})
    action: list[str(cardnames)]
    '''
    action = Poker2Num_seq(action, deck)
    return action

_online = os.environ.get("USER", "") == "root"
if _online:
    full_input = json.loads(input())
else:
    with open("log_forAI.json") as fo:
        full_input = json.load(fo)

# loading model
model = CNNModel()
data_dir = '/data/tractor_model.pt' # to be modified
model.load_state_dict(torch.load(data_dir, map_location = torch.device('cpu')))

hold = []
played = [[], [], [], []]
for i in range(len(full_input["requests"])-1):
    req = full_input["requests"][i]
    if req["stage"] == "deal":
        hold.extend(req["deliver"])
    elif req["stage"] == "cover":
        hold.extend(req["deliver"])
        action_cover = full_input["responses"][i]
        for id in action_cover:
            hold.remove(id)
    elif req["stage"] == "play":
        history = req["history"]
        selfid = (history[3] + len(history[1])) % 4
        if len(history[0]) != 0:
            self_move = history[0][(selfid-history[2]) % 4]
            #print(hold)
            #print(self_move)
            for id in self_move:
                hold.remove(id)
            for player_rec in range(len(history[0])): # Recovering played cards
                played[(history[2]+player_rec) % 4].extend(history[0][player_rec])
curr_request = full_input["requests"][-1]
if curr_request["stage"] == "deal":
    get_card = curr_request["deliver"][0]
    called = curr_request["global"]["banking"]["called"]
    snatched = curr_request["global"]["banking"]["snatched"]
    level = curr_request["global"]["level"]
    response = call_Snatch(get_card, hold, called, snatched, level)
elif curr_request["stage"] == "cover":
    publiccard = curr_request["deliver"]
    level = curr_request["global"]["level"]
    major = curr_request["global"]["banking"]["major"]
    response = cover_Pub(publiccard, hold, level, major)
elif curr_request["stage"] == "play":
    level = curr_request["global"]["level"]
    major = curr_request["global"]["banking"]["major"]
    setMajor(major, level)
    # instantiate move_generator and cardwrapper 
    card_wrapper = cardWrapper()
    mv_gen = move_generator(level, major)
    
    history = curr_request["history"]
    selfid = (history[3] + len(history[1])) % 4
    if len(history[0]) != 0:
        self_move = history[0][(selfid-history[2]) % 4]
        #print(hold)
        #print(self_move)
        for id in self_move:
            hold.remove(id)
        for player_rec in range(len(history[0])): # Recovering played cards
            played[(history[2]+player_rec) % 4].extend(history[0][player_rec])
        for player_rec in range(len(history[1])):
            played[(history[3]+player_rec) % 4].extend(history[1][player_rec])
    history_curr = history[1]
    
    response = playCard(history_curr, hold, played, selfid, card_wrapper, mv_gen, model)

print(json.dumps({
    "response": response
}))



