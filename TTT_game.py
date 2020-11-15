# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 23:50:07 2020

@author: Aritz (modifications by me)

Tic-tac-toe game

From: https://towardsdatascience.com/reinforcement-learning-implement-tictactoe-189582bea542
"""

"""
Updating Q value function, which is the estimating value of (state, action) pair
Two parts: train 2 agents to play against each other and save their policy
Secondly, load the policy and make the agent to play against human
"""

"""
IMPORTS
"""
import numpy as np
import pickle
import tqdm
import termcolor
from termcolor import colored 


"""
VARIABLES
"""
BOARD_ROWS = 3
BOARD_COLS = 3

"""
3 major components: state, action, and reward.
State: board state of both the agent and its opponent
Initialise a 3X3 board with zeros indicating available positions and update positions with 1
if player 1 takes move and -1 if player 2 takes the move.
The action is what positions a player can choose based on the current board state.
Reward is between 0 and 1 and is only given at the end of the game.
"""

#--------------------------------------------------------------------------- 
      
class State:

    """
    Init
    
    We initialise a vacant board and two players: Agent1 and Agent2 (we initialise Agent1 to play first).
    Each player has a playSymbol, when a player takes an action, its playerSymbol will be filled
    in the board and the board state will be updated.
    """
    def __init__(self, Agent1, Agent2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.Agent1 = Agent1
        self.Agent2 = Agent2
        self.isEnd = False
        self.boardHash = None
        # Agent1 player first
        self.playerSymbol = 1
    
    """
    Board State
    
        * getHash hashes the current board state so that it can be stored in the state-value dictionary
        * availablePositions. When a player takes an action, its corresponding symbol will be filled in the board.
        * updateState. After the state being updated, the board will also update the current vacant positions on the board
          and feed it back to the next player in turn.
    """
    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        return self.boardHash

    def availablePositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    positions.append((i, j))  # need to be tuple
        return positions

    def updateState(self, position):
        self.board[position] = self.playerSymbol
        # switch to another player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    """
    Check Winner
    
    After each action being taken by the player, the function continuously checks if the game has ended.
    If end, it judges the winner of the game and gives its corresponding reward to each player.
    """
    def winner(self):
        # row
        for i in range(BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1
        # col
        for i in range(BOARD_COLS):
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1
        # diagonal
        diag_sum1 = sum([self.board[i, i] for i in range(BOARD_COLS)]) # main diagonal
        diag_sum2 = sum([self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)]) # anti-diagonal
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == 3:
            self.isEnd = True
            if diag_sum1 == 3 or diag_sum2 == 3:
                return 1
            else:
                return -1

        # tie
        # no available positions
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
        # not end
        self.isEnd = False
        return None
    
    """
    Winner
    
    The function checks sum of rows, columns and diagonals, and return 1 if Agent1 wins, -1 if Agent2 wins, 0 if draw
    and None if the game is not yet ended. At the end of the game, 1 is rewarded to winner and 0 to loser.
    One thing to notice is that we consider draw is also a bad end, so we give our agent Agent1 0.1reward even the
    game is tie (try different ones)
    """
    # only when game ends
    def giveReward(self):
        result = self.winner()
        # backpropagate reward
        if result == 1:
            self.Agent1.feedReward(1)
            self.Agent2.feedReward(0)
        elif result == -1:
            self.Agent1.feedReward(0)
            self.Agent2.feedReward(1)
        else:
            self.Agent1.feedReward(0.1)
            self.Agent2.feedReward(0.5)
            
    """ Board reset """
    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    """
    Training
    
    The agent is able to learn by updating value estimation and the board is all set up. It's time to let two players
    play against each other.
    During training, the process for each player is:
        * Look for available positions
        * Choose action
        * Update board state and add the action to player's states
        * Judge if reach the end of the game and give reward accordingly
    """
    
    def play(self, rounds=100, whatToDo="train"):
        counter_Agent1 = 0 # initialize counter for Agent1 win
        counter_Agent2 = 0 # initialize counter for Agent2 win
        counter_tie = 0 # initialize counter for tie
        if whatToDo == "train":
            displayToDo = ">> Training"
        elif whatToDo == "compete":
            displayToDo = ">> Competing" 
            print(colored("\n   {} vs. {}".format(Agent1.name, Agent2.name), "magenta"))
        for i in tqdm.tqdm(range(rounds), desc=displayToDo):
            # if i % 1000 == 0: # display every 1000
            #     print("    Trained rounds: {}".format(i))
            while not self.isEnd:
                # Player 1
                positions = self.availablePositions()
                Agent1_action = self.Agent1.chooseAction(positions, self.board, self.playerSymbol)
                # take action and upate board state
                self.updateState(Agent1_action)
                board_hash = self.getHash()
                self.Agent1.addState(board_hash)
                # check board status if it is end

                win = self.winner()
                if win is not None:
                    # self.showBoard()
                    # ended with Agent1 either win or draw
                    if win == 1:
                        # print("Winner: {}!".format(self.Agent1.name))
                        counter_Agent1 = counter_Agent1 + 1
                    else:
                        # print("Tie")
                        counter_tie = counter_tie +1
                    self.giveReward()
                    self.Agent1.reset()
                    self.Agent2.reset()
                    self.reset()
                    break

                else:
                    # Player 2
                    positions = self.availablePositions()
                    Agent2_action = self.Agent2.chooseAction(positions, self.board, self.playerSymbol)
                    self.updateState(Agent2_action)
                    board_hash = self.getHash()
                    self.Agent2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        # self.showBoard()
                        # ended with Agent2 either win or draw
                        if win == -1:
                            # print(self.Agent2.name, "wins!")
                            counter_Agent2 = counter_Agent2 + 1
                        else:
                            # print("Tie")
                            counter_tie = counter_tie +1
                        self.giveReward()
                        self.Agent1.reset()
                        self.Agent2.reset()
                        self.reset()
                        break 
        if whatToDo == "train":
            Agent1.savePolicy(rounds)
        print("\n>> Results: {}-{} ({} ties)".format(counter_Agent1, counter_Agent2, counter_tie))
        if whatToDo == "compete":
            if counter_Agent1 < counter_Agent2:
                print()
                print(termcolor.cprint("   Winner: {}!".format(Agent2.name), 'grey', 'on_white'))
            elif counter_Agent1 > counter_Agent2:
                print(termcolor.cprint("   Winner: {}!".format(Agent1.name), 'grey', 'on_white'))
                
    """
    Modify a bit on the play function when Human 
    We let player 1 (agent) play first, and at each step, the board is printed
    """
    def playH(self):
        print(colored("\n   {} vs. {}".format(Agent1.name, Agent2.name), "magenta"))
        while not self.isEnd:
            # Player 1
            positions = self.availablePositions()
            Agent1_action = self.Agent1.chooseAction(positions, self.board, self.playerSymbol)
            # take action and upate board state
            self.updateState(Agent1_action)
            print("\n>> AI's turn")
            self.showBoard()
            # check board status if it is end
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(termcolor.cprint("   Winner: {}!".format(Agent1.name), 'white', 'on_red', attrs=['bold']))
                else:
                    print(termcolor.cprint("   It's a tie", 'grey', 'on_white'))
                self.reset()
                break
            
            else:
                # Player 2
                positions = self.availablePositions()
                Agent2_action = self.Agent2.chooseAction(positions)

                self.updateState(Agent2_action)
                self.showBoard()
                win = self.winner()
                if win is not None:
                    if win == -1:
                        print(termcolor.cprint("   Winner: {}!".format(Agent2.name), 'white', 'on_green', attrs=['bold']))
                    else:
                        print(termcolor.cprint("   It's a tie", 'grey', 'on_white'))
                    self.reset()
                    break

    """
    Show Board
    """        
    def showBoard(self):
        # Agent1: x  Agent2: o
        for i in range(0, BOARD_ROWS):
            print('   -------------')
            out = '   | '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = colored('x', 'yellow')
                if self.board[i, j] == -1:
                    token = colored('o', 'cyan')
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('   -------------')
        
#---------------------------------------------------------------------------       

"""
Player setting

Player class that represents the agent.
The player is able to:
    * Choose actions based on current estimation of the states
    * Record all the states of the game
    * Update states-value estimation after each game
    * Save and load the policy
"""
class Player: 
    
    """
    Init
    
    Initialise a dict storing state-value pair and update the estimates at the end of each game.
    We keep track of all positions the player's been taken during each game in a list self.states
    and update the corresponding states in self.states_value dict. In terms of choosing an action, 
    we use e-greedy method to balance between exploration and exploitation. Here we set
    exp_rate=0.3, which means e=0.3, so 70% of the time our agent will take greedy action, which is
    choosing action based on current estimation of states-value, and 30% of the time our agent will take
    random action.
    """
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.states = []  # record all positions taken
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}  # state -> value
        
    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return boardHash

    """
    Choose Action
    
    Store the hash of board state into state-value dict, and while exploitation, we hash the next board state and choose
    the action that returns the maximum value of next state.
    """
    def chooseAction(self, positions, current_board, symbol):
        if np.random.uniform(0, 1) <= self.exp_rate:
            # take random action
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_boardHash = self.getHash(next_board)
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
                # print("value", value)
                if value >= value_max:
                    value_max = value
                    action = p
        # print("{} takes action {}".format(self.name, action))
        return action
        
    # append a hash state
    def addState(self, state):
        self.states.append(state)
    
    """
    State-Value update
    
    We will apply value iteration which is updated based on the formula.
    The formula tells us that the updated value of state t equals the current value of state t
    adding the difference between the value of next state and the value of current state, which is
    multiplied by a learning rate alpha (given the reward of intermediate state is 0).
    We update the current value slowly.
    The positions of each game is stored in self.states and when the agent reaches the end of the game,
    the estimates are updated in reversed fashion.
    """
    # at the end of game, backpropagate and update states value    
    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]
            
    def reset(self):
        self.states = []

    """
    Saving & Loading Policy
    
    At the end of the training (playing after a certain amount of rounds), our agent is able to learn its policy
    which is stored in the state-value dict. We need to save this policy to play against a human player.
    """
    def savePolicy(self, rounds):
        fw = open('Policies/policy_{}rounds_'.format(rounds) + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()
        
#---------------------------------------------------------------------------

"""
Human VS Computer

Human class to play against the agent.
This class includes only 1 usable function chooseAction which requires us to input the board position we hope to take
"""
class HumanPlayer: 
    def __init__(self, name):
        self.name = name
    
    def chooseAction(self, positions):
        while True:
            print("\n>> Your turn")
            print("\n   Positions: {}".format(positions))
            row = int(input("   Choose a row: "))
            col = int(input("   Choose a column: "))
            action = (row, col)
            if action in positions:
                return action
            else:
                print(colored("\n   The chosen position is not available", "red"))
            
    # append a hash state
    def addState(self, state):
        pass
    
    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        pass
    
    def reset(self):
        pass
 
#---------------------------------------------------------------------------
    
"""
To do/implement:

    * What is that None display   
    * Try/test new parameters
    * Why does Agent1 always win
    * Train and name them unbeatable etc.
"""

#---------------------------------------------------------------------------
 
"""
Training (COMMENT/UNCOMMENT)
"""
# if __name__ == "__main__":
#     # training
#     Agent1 = Player("AI1")
#     Agent2 = Player("AI2")

#     st = State(Agent1, Agent2)
#     st.play(50000, "train") 
    
#---------------------------------------------------------------------------

"""
Agent vs. Agent game (COMMENT/UNCOMMENT)
""" 
if __name__ == "__main__":  
    Agent1 = Player("easy AI", exp_rate=1)
    Agent1.loadPolicy("Policies/easy")
    
    Agent2 = Player("medium AI", exp_rate=1)
    Agent2.loadPolicy("Policies/medium")

    st = State(Agent1, Agent2)
    st.play(100, "compete")

#---------------------------------------------------------------------------
    
"""
Human vs. Agent game (COMMENT/UNCOMMENT)
""" 
# if __name__ == "__main__": 
#     Agent1 = Player("AI", exp_rate=1)
#     Agent1.loadPolicy("Policies/unbeatable")
    
#     Agent2 = HumanPlayer("堃堃")
    
#     st = State(Agent1, Agent2)
#     st.playH() # 100 rounds by default
