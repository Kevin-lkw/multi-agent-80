import random
from collections import Counter
from mvGen import move_generator
import copy
import random
from collections import Counter

def generate_rule_based_action_options(action_options, hand, major_suit, point_order_wo_level, level, first_round):
    """
    Generate rule-based actions based on the given rules.
    - action_options: List of all legal actions.
    - hand: Current player's hand in card names.
    - major_suit: The current major suit.
    - point_order: List of point order for card ranking.
    - level: The current level card.

    Returns:
        A list of rule-based actions.
    """

    # Utils        
    def is_major(card):
        """Determine if a card is a major card."""
        return card[0] == major_suit or card in ['jo', 'Jo'] or card[1] == level

    def filter_non_major(cards):
        """Filter non-major cards."""
        return [card for card in cards if not is_major(card)]

    def filter_ban_by_rank(cards, ranks):
        """Filter cards by ranks."""
        return [card for card in cards if card[1] not in ranks]

    point_order = point_order_wo_level + [level]
    rule_based_options = []
    
    # 跟牌逻辑
    if not first_round:
        if len(action_options)>4:
            # 按照牌数分开处理
            for action in action_options:
                if len(action) == 1:  # 单张
                    # 分离主牌和副牌
                    major_cards = [card for card in hand if is_major(card)]
                    non_major_cards = filter_non_major(hand)
                    major_cards = filter_ban_by_rank(major_cards, ['o'])
                    # 主牌的最大和最小
                    if major_cards:
                        major_max = max(major_cards, key=lambda x: point_order.index(x[1]))
                        major_min = min(major_cards, key=lambda x: point_order.index(x[1]))
                        if action[0] in [major_max, major_min]:
                            rule_based_options.append(action)

                    # 副牌的最大和最小
                    if non_major_cards:
                        non_major_max = max(non_major_cards, key=lambda x: point_order.index(x[1]))
                        non_major_min = min(non_major_cards, key=lambda x: point_order.index(x[1]))
                        if action[0] in [non_major_max, non_major_min]:
                            rule_based_options.append(action)

                    # 分数牌和王牌
                    if action[0][1] in ['5', '0', 'K'] or action[0] in ['jo', 'Jo']:
                        rule_based_options.append(action)

                elif len(action) == 2:  # 对子
                    # 包含分数牌的组合
                    if action[0][1] == action[1][1] and action[0][1] in ['5', '0', 'K']:
                        rule_based_options.append(action)

                    # 每种花色不含分数的最小组合
                    suits_count = Counter(card[0] for card in filter_non_major(hand))
                    if suits_count:
                        for suit in suits_count.keys():
                            suit_cards = [card for card in filter_non_major(hand) if card[0] == suit and card[1] not in ['5', '0', 'K']]
                            if len(suit_cards) >= 2:
                                suit_cards.sort(key=lambda x: point_order.index(x[1]))
                                smallest_pair = suit_cards[:2]
                                if action[0:2] == smallest_pair:
                                    rule_based_options.append(action)

                    # 所有可行的对子
                    if action[0] == action[1]:
                        rule_based_options.append(action)

        if rule_based_options:
            # 去重
            rule_based_options = list(map(list, set(map(tuple, rule_based_options))))
            return rule_based_options

        # 其他牌数的直接返回
        return action_options
    


    # 首轮出牌逻辑

    # Drag tractors (pairs of consecutive cards)
    for action in action_options:
        if len(action) >= 3:
            rule_based_options.append(action)
    
    if rule_based_options:
        return rule_based_options
    
    # Non-major A's 
    for action in action_options:
        if len(action) == 1 and action[0][1] == 'A' and not is_major(action[0]):
            rule_based_options.append(action)

    if rule_based_options:
        return rule_based_options

    # Pairs (not 10 or 5, per suit)
    for action in action_options:
        if len(action) == 2:
            card = action[0]
            if card[1] not in ['5', '0'] and action[0] == action[1]:
                rule_based_options.append(action)

    if rule_based_options:
        return rule_based_options

    # Largest cards from the least common non-major suit 
    non_major_cards = filter_non_major(hand)
    non_score_cards = filter_ban_by_rank(non_major_cards, ['5', '0'])
    suits_count = Counter(card[0] for card in non_major_cards)
    if suits_count:
        least_common_suit = suits_count.most_common()[-1][0]
        least_common_suit_cards = [card for card in non_major_cards if card[0] == least_common_suit]
        largest_card = max([card for card in least_common_suit_cards if card[0] == least_common_suit],
                            key=lambda x: point_order.index(x[1]))
        for action in action_options:
            if len(action) == 1 and action[0] == largest_card:
                rule_based_options.append(action)

    # Largest major cards (not jokers)
    major_cards = [card for card in hand if is_major(card) and card not in ['jo', 'Jo'] and card[1] not in ['5', '0', 'K']]
    if major_cards:
        largest_major = max(major_cards, key=lambda x: point_order.index(x[1]))
        for action in action_options:
            if len(action) == 1 and action[0] == largest_major:
                rule_based_options.append(action)
        
    
    # and pairs of 10 or 5
    for action in action_options:
        if len(action) == 2 and action[0] == action[1] and action[0][1] in ['5', '0']:
            rule_based_options.append(action)

    if rule_based_options:
        return rule_based_options

    # Jokers or single 5 or 10 (if no other options)
    jokers50= [card for card in hand if card in ['jo', 'Jo'] or card[1] in ['5', '0', 'K']]
    if jokers50:
        for action in action_options:
            if len(action) == 1 and action[0] in jokers50:
                rule_based_options.append(action)

    if rule_based_options:
        return rule_based_options
    
    if action_options:
        print(action_options)

    return action_options



class Error(Exception):
    def __init__(self, ErrorInfo):
        self.ErrorInfo = ErrorInfo
        
    def __str__(self):
        return self.ErrorInfo


class TractorEnv():
    def __init__(self, config={}):
        if 'seed' in config:
            self.seed = config['seed']
        else:
            self.seed = None

        self.suit_set = ['s','h','c','d']
        self.card_scale = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K']
        self.major = None
        self.level = None
        self.agent_names = ['player_%d' % i for i in range(4)]
        
        
    def reset(self, level='2', banker_pos=0, major='s', rule_based = False):
        self.point_order = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']
        self.Major = ['jo', 'Jo']
        self.level = level
        self.first_round = True # if first_round, banker is the determined during dealing stage (not pre_determined)
        self.banker_pos = banker_pos
        if major == 'r': # randomly generating major
            self.major = random.sample(self.suit_set, 1)[0]
        else:
            self.major = major
        # if self.banker_pos: # banker predetermined, cannot be first_round
        #     self.first_round = False
        # initializing reporters and snathcers
        # self.reporter = None
        # self.snatcher = None
        # initializing decks
        self.total_deck = [i for i in range(108)] 
        random.shuffle(self.total_deck)
        # self.public_card = self.total_deck[100:] # saving 8 public cards
        self.covered_card = self.total_deck[100:]
        self.card_todeal = self.total_deck[:100]
        self.player_decks = [[] for _ in range(4)]
        self.player_decks[0] = self.card_todeal[:25]
        self.player_decks[1] = self.card_todeal[25:50]
        self.player_decks[2] = self.card_todeal[50:75]
        self.player_decks[3] = self.card_todeal[75:100]
        self._setMajor()
        self.mv_gen = move_generator(self.level, self.major)
        # assuming that covered_cards are publiccards
        # self.covered_cards = [] 
        # loading and initializing agents and game states
        self.score = 0
        self.history = []
        self.played_cards = [[] for _ in range(4)]
        self.reward = None
        self.done = False
        self.round = 0 # 轮次计数器
        
        self.round += 1
        # Do the first round
        self.current_player = self.banker_pos

        # rule_based_added
        if rule_based:
            action_options = self._get_action_options(self.banker_pos)
            rule_based_action_options = generate_rule_based_action_options(action_options, 
                                                                        [self._id2name(p) for p in self.player_decks[self.banker_pos]], 
                                                                        self.major, self.point_order, self.level, self.round %4 ==1)
            return self._get_obs(self.banker_pos), action_options, rule_based_action_options
        return self._get_obs(self.banker_pos), self._get_action_options(self.banker_pos)

    
    def step(self, response, rule_based = False): #response: dict{'player': player_id, 'action': action}
        # Each step receives a response and provides an obs
        self.reward = None
        curr_player = response['player']
        action = response['action']
        real_action = self._checkLegalMove(action, curr_player)
        real_action = self._name2id_seq(real_action, self.player_decks[curr_player])
        self._play(curr_player, real_action)
        next_player = (curr_player + 1) % 4
        if len(self.history) == 4: # finishing a round
            winner = self._checkWinner(curr_player)
            next_player = winner
            if len(self.player_decks[0]) == 0: # Ending the game
                self._reveal(curr_player, winner)
                self.done = True
        self.round += 1
        self.current_player = next_player


               # rule based added
        if rule_based:
            next_player = (response['player'] + 1) % 4
            action_options = self._get_action_options(next_player)
            rule_based_action_options = generate_rule_based_action_options(action_options, 
                                                                        [self._id2name(p) for p in self.player_decks[next_player]], 
                                                                        self.major, self.point_order, self.level,self.round %4 ==1)
            return self._get_obs(next_player), action_options, self.reward, self.done, rule_based_action_options

        return self._get_obs(next_player), self._get_action_options(next_player), self.reward, self.done
        
 

        
    
    def _raise_error(self, player, info):
        raise Error("Player_"+str(player)+": "+info)
        
    def _get_obs(self, player):
        obs = {
            "id": player,
            "deck": [self._id2name(p) for p in self.player_decks[player]],
            "history": [[self._id2name(p) for p in move] for move in self.history],
            "major": self.Major,
            "played": [[self._id2name(p) for p in move] for move in self.played_cards]
        }
        return obs

    def _get_action_options(self, player):
        deck = [self._id2name(p) for p in self.player_decks[player]]
        if len(self.history) == 4 or len(self.history) == 0: # first to play
            return self.mv_gen.gen_all(deck)
        else:
            tgt = [self._id2name(p) for p in self.history[0]]
            poktype = self._checkPokerType(self.history[0], (player-len(self.history))%4)
            if poktype == "single":
                return self.mv_gen.gen_single(deck, tgt)
            elif poktype == "pair":
                return self.mv_gen.gen_pair(deck, tgt)
            elif poktype == "tractor":
                return self.mv_gen.gen_tractor(deck, tgt)
            elif poktype == "suspect":
                return self.mv_gen.gen_throw(deck, tgt)    
    
    def _done(self):
        return self.done    
    
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
    
    def _name2id(self, card_name, deck):
        NumInDeck = -1
        if card_name[0] == "j":
            NumInDeck = 52
        elif card_name[0] == "J":
            NumInDeck = 53
        else:
            NumInDeck = self.card_scale.index(card_name[1])*4 + self.suit_set.index(card_name[0])
        if NumInDeck in deck:
            return NumInDeck
        else:
            return NumInDeck + 54
    
    def _name2id_seq(self, card_names, deck):
        id_seq = []
        deck_copy = deck + []
        for card_name in card_names:
            card_id = self._name2id(card_name, deck_copy)
            id_seq.append(card_id)
            deck_copy.remove(card_id)
        return id_seq
        
    
    def _play(self, player, cards):
        for card in cards:
            self.player_decks[player].remove(card)
            self.played_cards[player].append(card)
        if len(self.history) == 4: # beginning of a new round
            self.history = []
        self.history.append(cards)
            
    def _reveal(self, currplayer, winner): # 扣底
        if self._checkPokerType(self.history[0], (currplayer-3)%4) != "suspect":
            mult = len(self.history[0])
        else:
            divided, _ = self._checkThrow(self.history[0], (currplayer-3)%4, check=False)
            divided.sort(key=lambda x: len(x), reverse=True)
            if len(divided[0]) >= 4:
                mult = len(divided[0]) * 2
            elif len(divided[0]) == 2:
                mult = 4
            else: 
                mult = 2

        publicscore = 0
        for pok in self.covered_card: 
            p = self._id2name(pok)
            if p[1] == "5":
                publicscore += 5
            elif p[1] == "0" or p[1] == "K":
                publicscore += 10
        
        self._reward(winner, publicscore*mult)        
    
    def _setMajor(self):
        if self.major != 'n': # 非无主
            self.Major = [self.major+point for point in self.point_order if point != self.level] + [suit + self.level for suit in self.suit_set if suit != self.major] + [self.major + self.level] + self.Major
        else: # 无主
            self.Major = [suit + self.level for suit in self.suit_set] + self.Major
        self.point_order.remove(self.level)
        
    def _checkPokerType(self, poker, currplayer): #poker: list[int]
        level = self.level
        poker = [self._id2name(p) for p in poker]
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
            if "jo" in count.keys() and "Jo" in count.keys() and count['jo'] == 2 and count['Jo'] == 2 and len(poker) == 4:
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
                pointpos.append(self.point_order.index(k[1])) # 点数在大小上连续
            pointpos.sort()
            for i in range(len(pointpos)-1):
                if pointpos[i+1] - pointpos[i] != 1:
                    return "suspect"
            return "tractor" # 说明是拖拉机
        
        return "suspect"

    # 甩牌判定功能函数
    # return: ExistBigger(True/False)
    # 给定一组常规牌型，鉴定其他三家是否有同花色的更大牌型
    def _checkBigger(self, poker, currplayer):
    # poker: 给定牌型 list
    # own: 各家持牌 list
        own = self.player_decks
        level = self.level
        major = self.major
        tyPoker = self._checkPokerType(poker, currplayer)
        poker = [self._id2name(p) for p in poker]
        assert tyPoker != "suspect", "Type 'throw' should contain common types"
        own_pok = [[self._id2name(num) for num in hold] for hold in own]
        if poker[0] in self.Major: # 主牌型应用主牌压
            for i in range(len(own_pok)):
                if i == currplayer:
                    continue
                hold = own_pok[i]
                major_pok = [pok for pok in hold if pok in self.Major]
                count = Counter(major_pok)
                if len(poker) <= 2:
                    if poker[0][1] == level and poker[0][0] != major: # 含有副级牌要单算
                        if major == 'n': # 无主
                            for k,v in count.items(): 
                                if (k == 'jo' or k == 'Jo') and v >= len(poker):
                                    return True
                        else:
                            for k,v in count.items():
                                if (k == 'jo' or k == 'Jo' or k == major + level) and v >= len(poker):
                                    return True
                    else: 
                        for k,v in count.items():
                            if self.Major.index(k) > self.Major.index(poker[0]) and v >= len(poker):
                                return True
                else: # 拖拉机
                    if "jo" in poker: # 必定是大小王连对
                        return False # 不可能被压
                    if len(poker) == 4 and "jo" in count.keys() and "Jo" in count.keys():
                        if count["jo"] == 2 and count["Jo"] == 2: # 大小王连对必压
                            return True
                    pos = []
                    for k, v in count.items():
                        if v == 2:
                            if k != 'jo' and k != 'Jo' and k[1] != level and self.point_order.index(k[1]) > self.point_order.index(poker[-1][1]): # 大小王和级牌当然不会参与拖拉机
                                pos.append(self.point_order.index(k[1]))
                    if len(pos) >= 2:
                        pos.sort()
                        tmp = 0
                        suc_flag = False
                        for i in range(len(pos)-1):
                            if pos[i+1]-pos[i] == 1:
                                if not suc_flag:
                                    tmp = 2
                                    suc_flag = True
                                else:
                                    tmp += 1
                                if tmp >= len(poker)/2:
                                    return True
                            elif suc_flag:
                                tmp = 0
                                suc_flag = False
        else: # 副牌甩牌
            suit = poker[0][0]
            for i in range(len(own_pok)):
                if i == currplayer:
                    continue
                hold = own_pok[i]
                suit_pok = [pok for pok in hold if pok[0] == suit and pok[1] != level]
                count = Counter(suit_pok)
                if len(poker) <= 2:
                    for k, v in count.items():
                        if self.point_order.index(k[1]) > self.point_order.index(poker[0][1]) and v >= len(poker):
                            return True
                else:
                    pos = []
                    for k, v in count.items():
                        if v == 2:
                            if self.point_order.index(k[1]) > self.point_order.index(poker[-1][1]):
                                pos.append(self.point_order.index(k[1]))
                    if len(pos) >= 2:
                        pos.sort()
                        tmp = 0
                        suc_flag = False
                        for i in range(len(pos)-1):
                            if pos[i+1]-pos[i] == 1:
                                if not suc_flag:
                                    tmp = 2
                                    suc_flag = True
                                else:
                                    tmp += 1
                                if tmp >= len(poker)/2:
                                    return True
                            elif suc_flag:
                                tmp = 0
                                suc_flag = False

        return False

    # 甩牌是否可行
    # return: poker(最终实际出牌:list[str])、ilcnt(非法牌张数)
    # 如果甩牌成功，返回的是对甩牌的拆分(list[list])
    def _checkThrow(self, poker, currplayer, check=False):
    # poker: 甩牌牌型 list[int]
    # own: 各家持牌 list
    # level & major: 级牌、主花色
        own = self.player_decks
        level = self.level
        major = self.major
        ilcnt = 0
        pok = [self._id2name(p) for p in poker]
        outpok = []
        failpok = []
        count = Counter(pok)
        if check:
            if list(count.keys())[0] in self.Major: # 如果是主牌甩牌
                for p in count.keys():
                    if p not in self.Major:
                        self._raise_error(currplayer, "INVALID_POKERTYPE")
            else: # 是副牌
                suit = list(count.keys())[0][0] # 花色相同
                for k in count.keys():
                    if k[0] != suit:
                        self._raise_error(currplayer, "INVALID_POKERTYPE")
        # 优先检查整牌型（拖拉机）
        pos = []
        tractor = []
        suit = ''
        for k, v in count.items():
            if v == 2:
                if k != 'jo' and k != 'Jo' and k[1] != level: # 大小王和级牌当然不会参与拖拉机
                    pos.append(self.point_order.index(k[1]))
                    suit = k[0]
        if len(pos) >= 2:
            pos.sort()
            tmp = []
            suc_flag = False
            for i in range(len(pos)-1):
                if pos[i+1]-pos[i] == 1:
                    if not suc_flag:
                        tmp = [suit + self.point_order[pos[i]], suit + self.point_order[pos[i]], suit + self.point_order[pos[i+1]], suit + self.point_order[pos[i+1]]]
                        del count[suit + self.point_order[pos[i]]]
                        del count[suit + self.point_order[pos[i+1]]] # 已计入拖拉机的，从牌组中删去
                        suc_flag = True
                    else:
                        tmp.extend([suit + self.point_order[pos[i+1]], suit + self.point_order[pos[i+1]]])
                        del count[suit + self.point_order[pos[i+1]]]
                elif suc_flag:
                    tractor.append(tmp)
                    suc_flag = False
            if suc_flag:
                tractor.append(tmp)
        # 对牌型作基础的拆分 
        for k,v in count.items(): 
            outpok.append([k for i in range(v)])
        outpok.extend(tractor)

        if check:
            for poktype in outpok:
                if self._checkBigger(poktype, currplayer): # 甩牌失败
                    ilcnt += len(poktype)
                    failpok.append(poktype)  
        
        if ilcnt > 0:
            finalpok = []
            kmin = ""
            for poktype in failpok:
                getmark = poktype[-1] 
                if kmin == "":
                    finalpok = poktype
                    kmin = getmark
                elif kmin in self.Major: # 主牌甩牌
                    if self.Major.index(getmark) < self.Major.index(kmin):
                        finalpok = poktype
                        kmin = getmark
                else: # 副牌甩牌
                    if self.point_order.index(getmark[1]) < self.point_order.index(kmin[1]):
                        finalpok = poktype
                        kmin = getmark
            finalpok = [[finalpok[0]]]
        else: 
            finalpok = outpok

        return finalpok, ilcnt 
        
        
    def _checkRes(self, poker, own): # poker: list[int]
        level = self.level
        pok = [self._id2name(p) for p in poker]
        own_pok = [self._id2name(p) for p in own]
        if pok[0] in self.Major:
            major_pok = [pok for pok in own_pok if pok in self.Major]
            count = Counter(major_pok)
            if len(poker) <= 2:
                for v in count.values():
                    if v >= len(poker):
                        return True
            else: # 拖拉机 
                pos = []
                for k, v in count.items():
                    if v == 2:
                        if k != 'jo' and k != 'Jo' and k[1] != level: # 大小王和级牌当然不会参与拖拉机
                            pos.append(self.point_order.index(k[1]))
                if len(pos) >= 2:
                    pos.sort()
                    tmp = 0
                    suc_flag = False
                    for i in range(len(pos)-1):
                        if pos[i+1]-pos[i] == 1:
                            if not suc_flag:
                                tmp = 2
                                suc_flag = True
                            else:
                                tmp += 1
                            if tmp >= len(poker)/2:
                                return True
                        elif suc_flag:
                            tmp = 0
                            suc_flag = False
        else:
            suit = pok[0][0]
            suit_pok = [pok for pok in own_pok if pok[0] == suit and pok[1] != level]
            count = Counter(suit_pok)
            if len(poker) <= 2:
                for v in count.values():
                    if v >= len(poker):
                        return True
            else:
                pos = []
                for k, v in count.items():
                    if v == 2:
                        pos.append(self.point_order.index(k[1]))
                if len(pos) >= 2:
                    pos.sort()
                    tmp = 0
                    suc_flag = False
                    for i in range(len(pos)-1):
                        if pos[i+1]-pos[i] == 1:
                            if not suc_flag:
                                tmp = 2
                                suc_flag = True
                            else:
                                tmp += 1
                            if tmp >= len(poker)/2:
                                return True
                        elif suc_flag:
                            tmp = 0
                            suc_flag = False
        return False
    
    def _checkLegalMove(self, poker, currplayer): # own: All players' hold before this move
    # poker: list[int] player's move
    # history: other players' moves in the current round: list[list]
        level = self.level
        major = self.major
        own = self.player_decks
        banker = self.banker_pos
        history = self.history
        pok = [self._id2name(p) for p in poker]
        hist = [[self._id2name(p) for p in move] for move in history]
        outpok = pok
        own_pok = [self._id2name(p) for p in own[currplayer]]
        if len(history) == 0 or len(history) == 4: # The first move in a round
            # Player can only throw in the first round
            typoker = self._checkPokerType(poker, currplayer)
            if typoker == "suspect":
                outpok_s, ilcnt = self._checkThrow(poker, currplayer, True)
                if ilcnt > 0:
                    self._punish(currplayer, ilcnt*10)
                outpok = [p for poktype in outpok_s for p in poktype] # 符合交互模式，把甩牌展开
        else:
            tyfirst = self._checkPokerType(history[0], currplayer)
            if len(poker) != len(history[0]):
                self._raise_error(currplayer, "ILLEGAL_MOVE")
            if tyfirst == "suspect": # 这里own不一样了，但是可以不需要check
                outhis, ilcnt = self._checkThrow(history[0], currplayer, check=False)
                # 甩牌不可能失败，因此只存在主牌毙或者贴牌的情形，且不可能有应手
                # 这种情况下的非法行动：贴牌不当
                # outhis是已经拆分好的牌型(list[list])
                flathis = [p for poktype in outhis for p in poktype]
                if outhis[0][0] in self.Major: 
                    major_pok = [p for p in pok if p in self.Major]
                    if len(major_pok) != len(poker): # 这种情况下，同花(主牌)必须已经贴完
                        major_hold = [p for p in own_pok if p in self.Major]
                        if len(major_pok) != len(major_hold):
                            self._raise_error(currplayer, "ILLEGAL_MOVE")
                    else: #全是主牌
                        outhis.sort(key=lambda x: len(x), reverse=True) # 牌型从大到小来看
                        major_hold = [p for p in own_pok if p in self.Major]
                        matching = True
                        if self._checkPokerType(outhis[0], currplayer) == "tractor": # 拖拉机来喽
                            divider, _ = self._checkThrow(poker, currplayer, check=False)
                            divider.sort(key=lambda x: len(x), reverse=True)
                            dividcnt = [len(x) for x in divider]
                            own_divide, r = self._checkThrow(major_hold, currplayer, check=False)
                            own_divide.sort(key=lambda x: len(x), reverse=True)
                            own_cnt = [len(x) for x in own_divide]
                            for poktype in outhis: # 可以使用这种方法的原因在于同一组花色/主牌可组成的牌型数量太少，不会出现多解
                                if dividcnt[0] >= len(poktype):
                                    dividcnt[0] -= len(poktype)
                                    dividcnt.sort(reverse=True)
                                else:
                                    matching = False
                                    break
                            if not matching: # 不匹配，看手牌是否存在应手
                                res_ex = True
                                for chtype in own_cnt:
                                    if own_cnt[0] >= len(chtype):
                                        own_cnt[0] -= len(chtype)
                                        own_cnt.sort(reverse=True)
                                    else: 
                                        res_ex = False
                                        break
                                if res_ex: # 存在应手，说明贴牌不当
                                    self._raise_error(currplayer, "ILLEGAL_MOVE")
                                else: # 存在应手，继续检查
                                    pair_own = sum([len(x) for x in own_divide if len(x) >= 2])
                                    pair_his = sum([len(x) for x in outhis if len(x) >= 2])
                                    pair_pok = sum([len(x) for x in divider if len(x) >= 2])
                                    if pair_pok < min(pair_own, pair_his):
                                        self._raise_error(currplayer, "ILLEGAL_MOVE")
                else:
                    suit = hist[0][0][0]
                    suit_pok = [p for p in pok if p not in self.Major and p[0] == suit]
                    if len(suit_pok) != len(poker): # 这种情况下，同花(主牌)必须已经贴完
                        suit_hold = [p for p in own_pok if p not in self.Major and p[0] == suit]
                        if len(suit_pok) != len(suit_hold):
                            self._raise_error(currplayer, "ILLEGAL_MOVE")
                    else: 
                        outhis.sort(key=lambda x: len(x), reverse=True) # 牌型从大到小来看
                        suit_hold = [p for p in own_pok if p not in self.Major and p[0] == suit]
                        matching = True
                        if self._checkPokerType(outhis[0], currplayer) == "tractor": # 拖拉机来喽
                            divider, _ = self._checkThrow(poker, currplayer, check=False)
                            divider.sort(key=lambda x: len(x), reverse=True)
                            dividcnt = [len(x) for x in divider]
                            own_divide, r = self._checkThrow(suit_hold, currplayer, check=False)
                            own_divide.sort(key=lambda x: len(x), reverse=True)
                            own_cnt = [len(x) for x in own_divide]
                            for poktype in outhis: # 可以使用这种方法的原因在于同一组花色/主牌可组成的牌型数量太少，不会出现多解
                                if dividcnt[0] >= len(poktype):
                                    dividcnt[0] -= len(poktype)
                                    dividcnt.sort(reverse=True)
                                else:
                                    matching = False
                                    break
                            if not matching: # 不匹配，看手牌是否存在应手
                                res_ex = True
                                for chtype in outhis:
                                    if own_cnt[0] >= len(chtype):
                                        own_cnt[0] -= len(chtype)
                                        own_cnt.sort(reverse=True)
                                    else: 
                                        res_ex = False
                                        break
                                if res_ex: # 存在应手，说明贴牌不当
                                    self._raise_error(currplayer, "ILLEGAL_MOVE")
                                else: # 存在应手，继续检查
                                    pair_own = sum([len(x) for x in own_divide if len(x) >= 2])
                                    pair_his = sum([len(x) for x in outhis if len(x) >= 2])
                                    pair_pok = sum([len(x) for x in divider if len(x) >= 2])
                                    if pair_pok < min(pair_own, pair_his):
                                        self._raise_error(currplayer, "ILLEGAL_MOVE")
                            # 到这里关于甩牌贴牌的问题基本上解决，是否存在反例还有待更详细的讨论

            else: # 常规牌型
            # 该情形下的非法行动：(1) 有可以应手的牌型但贴牌或用主牌毙 (2) 贴牌不当(有同花不贴/拖拉机有对子不贴)
                if self._checkRes(history[0], own[currplayer]): #(1) 有应手但贴牌或毙
                    if self._checkPokerType(poker, currplayer) != tyfirst:
                        self._raise_error(currplayer,"ILLEGAL_MOVE")
                    if hist[0][0] in self.Major and pok[0] not in self.Major:
                        self._raise_error(currplayer,"ILLEGAL_MOVE")
                    if hist[0][0] not in self.Major and (pok[0] in self.Major or pok[0][0] != hist[0][0][0]):
                        self._raise_error(currplayer, "ILLEGAL_MOVE") 
                elif self._checkPokerType(poker, currplayer) != tyfirst: #(2) 贴牌不当: 有同花不贴完/同花色不跟整牌型
                    own_pok = [self._id2name(p) for p in own[currplayer]]
                    if hist[0][0] in self.Major:
                        major_pok = [p for p in pok if p in self.Major]
                        major_hold = [p for p in own_pok if p in self.Major]
                        if len(major_pok) != len(poker): # 这种情况下，同花(主牌)必须已经贴完
                            if len(major_pok) != len(major_hold):
                                self._raise_error(currplayer, "ILLEGAL_MOVE")
                        else: # 完全是主牌
                            count = Counter(major_hold)
                            if tyfirst == "pair":
                                for v in count.values():
                                    if v == 2:
                                        self._raise_error(currplayer, "ILLEGAL_MOVE")
                            elif tyfirst == "tractor":
                                trpairs = len(history[0])/2
                                pkcount = Counter(pok)
                                pkpairs = 0
                                hdpairs = 0
                                for v in pkcount.values():
                                    if v >= 2:
                                        pkpairs += 1
                                for v in count.values():
                                    if v >= 2:
                                        hdpairs += 1
                                if pkpairs < trpairs and pkpairs < hdpairs: # 并不是所有对子都用上了
                                    self._raise_error(currplayer, "ILLEGAL_MOVE")

                    else: 
                        suit = hist[0][0][0]
                        suit_pok = [p for p in pok if p[0] == suit and p not in self.Major]
                        suit_hold = [p for p in own_pok if p[0] == suit and p not in self.Major]
                        if len(suit_pok) != len(poker):    
                            if len(suit_pok) != len(suit_hold):
                                self._raise_error(currplayer, "ILLEGAL_MOVE")
                        else: # 完全是同种花色
                            count = Counter(suit_hold)
                            if tyfirst == "pair":
                                for v in count.values():
                                    if v == 2:
                                        self._raise_error(currplayer, "ILLEGAL_MOVE")
                            elif tyfirst == "tractor":
                                trpairs = len(history[0])/2
                                pkcount = Counter(pok)
                                pkpairs = 0
                                hdpairs = 0
                                for v in pkcount.values():
                                    if v >= 2:
                                        pkpairs += 1
                                for v in count.values():
                                    if v >= 2:
                                        hdpairs += 1
                                if pkpairs < trpairs and pkpairs < hdpairs: # 并不是所有对子都用上了
                                    self._raise_error(currplayer, "ILLEGAL_MOVE")
                        
        return outpok
    
    def _checkWinner(self, currplayer):
        level = self.level
        major = self.major
        history = self.history
        histo = history + []
        hist = [[self._id2name(p) for p in x] for x in histo]
        score = 0 
        for move in hist:
            for pok in move:
                if pok[1] == "5":
                    score += 5
                elif pok[1] == "0" or pok[1] == "K":
                    score += 10
        win_seq = 0 # 获胜方在本轮行动中的顺位，默认为0
        win_move = hist[0] # 获胜方的出牌，默认为首次出牌
        tyfirst = self._checkPokerType(history[0], currplayer)
        if tyfirst == "suspect": # 甩牌
            first_parse, _ = self._checkThrow(history[0], currplayer, check=False)
            first_parse.sort(key=lambda x: len(x), reverse=True)
            for i in range(1,4):
                move_parse, r = self._checkThrow(history[i], currplayer, check=False)
                move_parse.sort(key=lambda x: len(x), reverse=True)
                move_cnt = [len(x) for x in move_parse]
                matching = True
                for poktype in first_parse: # 杀毙的前提是牌型相同
                    if move_cnt[0] >= len(poktype):
                        move_cnt[0] -= len(poktype)
                        move_cnt.sort(reverse=True)
                    else:
                        matching = False
                        break
                if not matching:
                    continue
                if hist[i][0] not in self.Major: # 副牌压主牌，算了吧
                    continue
                if win_move[0] not in self.Major and hist[i][0] in self.Major: # 主牌压副牌，必须的
                    win_move = hist[i]
                    win_seq = i
                # 两步判断后，只剩下hist[i]和win_move都是主牌的情况
                elif len(first_parse[0]) >= 4: # 有拖拉机再叫我checkThrow来
                    if major == 'n': # 如果这里无主，拖拉机只可能是对大小王，不可能有盖毙
                        continue
                    win_parse, s = self._checkThrow(history[win_seq], currplayer, check=False)
                    win_parse.sort(key=lambda x: len(x), reverse=True)
                    if self.Major.index(win_parse[0][-1]) < self.Major.index(move_parse[0][-1]):
                        win_move = hist[i]
                        win_seq = i
                else: 
                    step = len(first_parse[0])
                    win_count = Counter(win_move)
                    win_max = 0
                    for k,v in win_count.items():
                        if v >= step and self.Major.index(k) >= win_max: # 这里可以放心地这么做，因为是何种花色的副2不会影响对比的结果
                            win_max = self.Major.index(k)
                    move_count = Counter(hist[i])
                    move_max = 0
                    for k,v in move_count.items():
                        if v >= step and self.Major.index(k) >= move_max:
                            move_max = self.Major.index(k)
                    if major == 'n': # 无主
                        if self.Major[win_max][1] == level:
                            if self.Major[move_max] == 'jo' or self.Major[move_max] == 'Jo':
                                win_move = hist[i]
                                win_seq = i
                        elif self.Major.index(move_max) > self.Major.index(win_max):
                            win_move = hist[i]
                            win_seq = i
                    elif self.Major[win_max][1] == level and self.Major[win_max][0] != major:
                        if (self.Major[move_max][0] == major and self.Major[move_max][1] == level) or self.Major[move_max] == "jo" or self.Major[move_max] == "Jo":
                            win_move = hist[i]
                            win_seq = i
                    elif self.Major.index(win_max) < self.Major.index(move_max):
                        win_move = hist[i]
                        win_seq = i

        else: # 常规牌型
            #print("Common: Normal")
            for i in range(1, 4):
                if self._checkPokerType(history[i], currplayer) != tyfirst: # 牌型不对
                    continue
                #print("check: Normal")
                if (hist[0][0] in self.Major and hist[i][0] not in self.Major) or (hist[0][0] not in self.Major and (hist[i][0] not in self.Major and hist[i][0][0] != hist[0][0][0])):
                # 花色不对，贴
                    continue
                elif win_move[0] in self.Major: # 主牌不会被主牌杀，且该分支内应手均为主牌
                    if hist[i][0] not in self.Major: # 副牌就不用看了
                        continue
                    #print("here")
                    if major == 'n':
                        if win_move[-1][1] == level:
                            if hist[i][-1] == 'jo' or hist[i][-1] == 'Jo': # 目前胜牌是级牌，只有大小王能压
                                win_move = hist[i]
                                win_seq = i
                        elif self.Major.index(hist[i][-1]) > self.Major.index(win_move[-1]):
                            win_move = hist[i]
                            win_seq = i
                    else:
                        if win_move[-1][0] != major and win_move[-1][1] == level:
                            if (hist[i][-1][0] == major and hist[i][-1][1] == level) or hist[i][-1] == 'jo' or hist[i][-1] == 'Jo':
                                win_move = hist[i]
                                win_seq = i
                        elif self.Major.index(hist[i][-1]) > self.Major.index(win_move[-1]):
                            win_move = hist[i]
                            win_seq = i
                else: # 副牌存在被主牌压的情况
                    if hist[i][0] in self.Major: # 主牌，正确牌型，必压
                        win_move = hist[i]
                        win_seq = i
                    elif self.point_order.index(win_move[0][-1]) < self.point_order.index(hist[i][0][-1]):
                        win_move = hist[i]
                        win_seq = i
        # 找到获胜方，加分
        win_id = (currplayer - 3 + win_seq) % 4
        self._reward(win_id, score)

        return win_id
    
    def _reward(self, player, points):
        # print("player ", player, " get ", points, " points")
        if (player-self.banker_pos) % 2 != 0: # farmer getting points
            self.score += points
        self.reward = {}
        for i in range(4):
            if (i-player) % 2 == 0:
                self.reward[self.agent_names[i]] = points
            else:
                self.reward[self.agent_names[i]] = -points

    def _punish(self, player, points):
        if (player-self.banker_pos) % 2 != 0:
            self.score -= points
        else:
            self.score += points
    
    def action_intpt(self, action, player):
        '''
        interpreting action(cardname) to response(dick{'player': int, 'action': list[int]})
        action: list[str(cardnames)]
        '''
        player_deck = self.player_decks[player]
        action = self._name2id_seq(action, player_deck)
        return {'player': player, 'action': action}
    
    def pack_data(self):
        return {
            "level": self.level,
            "major": self.major,
            "covered_card": copy.deepcopy(self.covered_card),
            "player_decks": copy.deepcopy(self.player_decks),
            "played_cards": copy.deepcopy(self.played_cards),
            "history": copy.deepcopy(self.history),
            "player":  copy.deepcopy(self.current_player),
            "banker_pos": self.banker_pos,
            "round": self.round,
            "agent_names": self.agent_names,
        }