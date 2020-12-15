#coding:utf-8
import sys
from state import *
from player import *
import signal

import torch.utils.data as data
import torch.nn.functional as F
from draw import *
import torch
import numpy as np
from datapacker import dataloader

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('log')

cuda = torch.device('cuda')
from torchCNN import *



GRID_WIDTH = 40

COLUMN = 0
ROW = 0

list1 = []  # AI
list2 = []  # human
list3 = []  # all

list_all = []  # 整个棋盘的点
next_point = [0, 0]  # AI下一步最应该下的位置

ratio = 1  # 进攻的系数   大于1 进攻型，  小于1 防守型
DEPTH = 3  # 搜索深度   只能是单数。  如果是负数， 评估函数评估的的是自己多少步之后的自己得分的最大值，并不意味着是最好的棋， 评估函数的问题


# 棋型的评估分数
shape_score = [(50, (0, 1, 1, 0, 0)),
               (50, (0, 0, 1, 1, 0)),
               (200, (1, 1, 0, 1, 0)),
               (500, (0, 0, 1, 1, 1)),
               (500, (1, 1, 1, 0, 0)),
               (5000, (0, 1, 1, 1, 0)),
               (5000, (0, 1, 0, 1, 1, 0)),
               (5000, (0, 1, 1, 0, 1, 0)),
               (5000, (1, 1, 1, 0, 1)),
               (5000, (1, 1, 0, 1, 1)),
               (5000, (1, 0, 1, 1, 1)),
               (5000, (1, 1, 1, 1, 0)),
               (5000, (0, 1, 1, 1, 1)),
               (50000, (0, 1, 1, 1, 1, 0)),
               (99999999, (1, 1, 1, 1, 1))]

class Gomoku(object):
    def __init__(self, dimension, chain):
        super(Gomoku, self).__init__()
        self.state = State()
        self.dimension = dimension
        self.chain = chain
        #self.limit = limit
        self.winner = None
        self.initial = None
        self.current = None
        self.oponent = None
        self.player_x = Player('X')
        self.player_o = Player('O')
        self.max_depth = int(dimension/2)
        self.minimax_moves = []

        COLUMN=dimension
        ROW=dimension
        for i in range(COLUMN+1):
            for j in range(ROW+1):
                list_all.append((i, j))

    def start(self):
        self.state.board.initializeBoard(self.dimension)
        # self.manualGame()
        if self.mode == 1:
            self.manualGame()
        elif self.mode == 2:
            self.agentTournamentGame()
        elif self.mode == 3:
            self.agentRandomGame() 
        elif self.mode == 4:
            self.TreeVsRandom()
        elif self.mode == 5:
            self.TreeVsCNN()  
            

    def swapTurn(self, player):
        if player == self.player_x:
            self.current = self.player_o
            self.oponent = self.player_x
            return self.player_o
        else:
            self.current = self.player_x
            self.oponent = self.player_o
            return self.player_x

    def swapInitial(self, player):
        if player == self.player_x:
            return self.player_o
        else:
            return self.player_x

    def manualGame(self):
        print("Start:                                       ", end='\n')
        print(" 1. Player X (X)                             ", end='\n')
        print(" 2. Player O (O)                             ", end='\n')

        i = int(input("Please choose starting player: "))
        if i == 1:
            self.initial = self.player_x
            self.oponent = self.player_o
            self.current = self.initial
        elif i == 2:
            self.initial = self.player_o
            self.oponent = self.player_x
            self.current = self.initial
        else:
            print("Only Two Players Available")
            sys.exit()

        print("------------------------------------------", end='\n')
        print("               Initial Board              ", end='\n')
        print("------------------------------------------", end='\n')
        while self.isOver() is not True:
            self.state.board.printBoard()
            print("------------------------------------------", end='\n')
            move = self.current.getMove()
            if self.state.board.isValidMove(move) is True:
                self.state = self.state.createNewState(move, self.current)
                # Play back the chess piece first to prevent it from being overwritten
                self.state.board.printBoard()
                # Find the winner, the priority of finding the winner should be higher
                if self.state.isWinner(self.current, self.chain) is True:
                   self.winner = self.current
                # Select a piece to delete, at least one round can be deleted before it is over
                if (i==1) and (self.current==self.player_o):
                    rand = self.state.board.randomDelete()
                    print("------------------------------------------", end='\n')
                    print("           Random Delete",rand,"          ", end='\n')
                    print("------------------------------------------", end='\n')
                    #print("Random Delete",rand)
                    self.state = self.state.createDeleteState(rand, self.current)
                if (i==2) and (self.current==self.player_x):
                    rand = self.state.board.randomDelete()
                    print("------------------------------------------", end='\n')
                    print("           Random Delete",rand,"          ", end='\n')
                    print("------------------------------------------", end='\n')
                    self.state = self.state.createDeleteState(rand, self.current)
                if (i==1) and (self.current==self.player_x):
                    print("------------------------------------------", end='\n')
                    print("             No Random Delete!            ", end='\n')
                    print("------------------------------------------", end='\n')
                    #print("No Random Delete!")
                if (i==2) and (self.current==self.player_o):
                    print("------------------------------------------", end='\n')
                    print("             No Random Delete!            ", end='\n')
                    print("------------------------------------------", end='\n')

                # Exchange order
                self.current = self.swapTurn(self.current)
            else:
                print("Spot Taken or Out of bounds,please input again!")

        if self.winner is not None:
            self.printWinMessage()
        else:
            self.printGameEnded()
            self.state.board.printBoard()

            print("------------------------------------------", end='\n')
            print("                 Tie                      ", end='\n')
            print("------------------------------------------", end='\n')
    
    # Human vs. AI(Tree-based)
    def agentTournamentGame(self):
        print("Start:                                   ", end='\n')
        print(" 1. Player AI (X)                        ", end='\n')
        print(" 2. Player Human (O)                     ", end='\n')

        i = int(input("Please choose starting player: "))
        if i == 1:
            self.minimax_val = 1
            self.initial = self.player_x
            self.oponent = self.player_o
            # Initial move randomly
            rand = self.state.board.randomMove()
            self.state = self.state.createNewState(rand, self.initial)
            print("------------------------------------------", end='\n')
            print("Initial AI Move: %s" % (rand,))
            #print("rand的结构是" ,rand)
            print("------------------------------------------", end='\n')
            self.printCurrentBoard()
            self.current = self.oponent
        elif i == 2:
            self.minimax_val = 1
            self.initial = self.player_o
            self.oponent = self.player_x
            self.current = self.initial
            
            self.printCurrentBoard()
        else:
            print("Only Two Players Available")
            sys.exit()

        while self.isOver() is not True:
            # Notice
            initial = self.current

            # AI First
            if initial.piece == 'X':
                # AI 选择
                move = self.ai()
                list1.append(move)
                list3.append(move)
                print("Tree-based AI Move: %s" % (move,))
                print("------------------------------------------", end='\n')
                self.state = self.state.createNewState(move, initial)
                self.printCurrentBoard()
                # heuristic = None
                # try:
                #     heuristic = self.alphaBetaSearch(initial,
                #                                      self.max_depth)
                # except Exception:
                #     self.state = self.state.createNewState(self.state.board.closeMove(self.state.move), initial)
                #     print("AI Best Move: %s" % (self.state.move,))
                #     self.printCurrentBoard()
                    
                #     #print("------------------------------------------", end='\n')
                #     self.minimax_moves = []

                # if heuristic is not None:
                #     best = self.getMaxMove()
                #     self.state = self.state.createNewState(best, initial)
                #     self.printCurrentBoard()
                #     print("AI Best Move: %s" % (best,))
                #     print("Heuristic Value: %d" % heuristic)
                #    # print("------------------------------------------", end='\n')
                #     self.minimax_moves = []
            # Human First
            else:
                flag = True
                while flag is True:
                    move = initial.getMove()
                    list2.append(move)
                    list3.append(move)
                    if self.state.board.isValidMove(move):
                        flag = None
                    else:
                        print("Spot Taken or Out of bounds,please input again!")
                self.state = self.state.createNewState(move, initial)
                self.printCurrentBoard()

            # 判断输赢
            # Find the winner, the priority of finding the winner should be higher
            if self.state.isWinner(self.current, self.chain) is True:
                self.winner = self.current
            # Random delete
            if (i==1) and (self.current==self.player_o):
                    rand = self.state.board.randomDelete()
                    print("------------------------------------------", end='\n')
                    print("           Random Delete",rand,"          ", end='\n')
                    print("------------------------------------------", end='\n')
                    #print("Random Delete",rand)
                    self.state = self.state.createDeleteState(rand, self.current)
                    self.printCurrentBoard()
            if (i==2) and (self.current==self.player_x):
                    rand = self.state.board.randomDelete()
                    print("------------------------------------------", end='\n')
                    print("           Random Delete",rand,"          ", end='\n')
                    print("------------------------------------------", end='\n')
                    self.state = self.state.createDeleteState(rand, self.current)
                    self.printCurrentBoard()
            if (i==1) and (self.current==self.player_x):
                    print("------------------------------------------", end='\n')
                    print("             No Random Delete!            ", end='\n')
                    print("------------------------------------------", end='\n')
                    #print("No Random Delete!")
            if (i==2) and (self.current==self.player_o):
                    print("------------------------------------------", end='\n')
                    print("             No Random Delete!            ", end='\n')
                    print("------------------------------------------", end='\n')
            

            # Exchange order
            self.current = self.swapTurn(initial)

        if self.winner is not None:
            self.printWinMessage()
        else:
            self.printGameEnded()
            self.state.board.printBoard()
            print("------------------------------------------", end='\n')
            print("                 Tie                      ", end='\n')
            print("------------------------------------------", end='\n')

    # Human vs. AI(Random)
    def agentRandomGame(self):
        print("Start:                                   ", end='\n')
        print(" 1. Player AI (X)                        ", end='\n')
        print(" 2. Player Human (O)                     ", end='\n')

        i = int(input("Please choose starting player: "))
        if i == 1:
            self.initial = self.player_x
            self.oponent = self.player_o
            self.current = self.initial
        elif i == 2:
            self.initial = self.player_o
            self.oponent = self.player_x
            self.current = self.initial
        else:
            print("Only Two Players Available")
            sys.exit()

        print("------------------------------------------", end='\n')
        print("               Initial Board              ", end='\n')
        print("------------------------------------------", end='\n')
        while self.isOver() is not True:
            self.state.board.printBoard()
            print("------------------------------------------", end='\n')

            if(self.current==self.player_o):
                move = self.current.getMove()
            else:
                rand = self.state.board.randomMove()
                move = rand
                print("Random AI Move: %s" % (move,))
                print("------------------------------------------", end='\n')
            if self.state.board.isValidMove(move) is True:
                self.state = self.state.createNewState(move, self.current)
                # Play back the chess piece first to prevent it from being overwritten
                self.state.board.printBoard()
                # Find the winner, the priority of finding the winner should be higher
                if self.state.isWinner(self.current, self.chain) is True:
                   self.winner = self.current
                # Select a piece to delete, at least one round can be deleted before it is over
                if (i==1) and (self.current==self.player_o):
                    rand = self.state.board.randomDelete()
                    print("------------------------------------------", end='\n')
                    print("           Random Delete",rand,"          ", end='\n')
                    print("------------------------------------------", end='\n')
                    #print("Random Delete",rand)
                    self.state = self.state.createDeleteState(rand, self.current)
                if (i==2) and (self.current==self.player_x):
                    rand = self.state.board.randomDelete()
                    print("------------------------------------------", end='\n')
                    print("           Random Delete",rand,"          ", end='\n')
                    print("------------------------------------------", end='\n')
                    self.state = self.state.createDeleteState(rand, self.current)
                if (i==1) and (self.current==self.player_x):
                    print("------------------------------------------", end='\n')
                    print("             No Random Delete!            ", end='\n')
                    print("------------------------------------------", end='\n')
                    #print("No Random Delete!")
                if (i==2) and (self.current==self.player_o):
                    print("------------------------------------------", end='\n')
                    print("             No Random Delete!            ", end='\n')
                    print("------------------------------------------", end='\n')

                # Exchange order
                self.current = self.swapTurn(self.current)
            else:
                print("Spot Taken or Out of bounds,please input again!")

        if self.winner is not None:
            self.printWinMessage()
        else:
            self.printGameEnded()
            self.state.board.printBoard()

            print("------------------------------------------", end='\n')
            print("                 Tie                      ", end='\n')
            print("------------------------------------------", end='\n')

    # AI(Tree-based) vs. AI(Random)
    def TreeVsRandom(self):
        global total_search_count_random
        
        total_search_count_random=0

        print("Start:                                   ", end='\n')
        print(" 1. Player AI(Tree-based) (X)                        ", end='\n')
        print(" 2. Player AI(Random) (O)                     ", end='\n')

        i = int(input("Please choose starting player: "))
        #i=1
        if i == 1:
            self.minimax_val = 2
            self.initial = self.player_x
            self.oponent = self.player_o
            # Initial move randomly
            rand = self.state.board.randomMove()
            self.state = self.state.createNewState(rand, self.initial)
            print("------------------------------------------", end='\n')
            print("Initial AI Move: %s" % (rand,))
            print("------------------------------------------", end='\n')
            self.printCurrentBoard()
            self.current = self.oponent
        elif i == 2:
            self.minimax_val = 1
            self.initial = self.player_o
            self.oponent = self.player_x
            self.current = self.initial
            
            self.printCurrentBoard()
        else:
            print("Only Two Players Available")
            sys.exit()

        while self.isOver() is not True:
            # Notice
            initial = self.current

            # AI First
            if initial.piece == 'X':
                move = self.ai()
                # print("move的结构是:",move)
                list1.append(move)
                list3.append(move)
                print("Tree-based AI Move: %s" % (move,))
                print("------------------------------------------", end='\n')
                self.state = self.state.createNewState(move, initial)
                self.printCurrentBoard()
            # Human First
            else:    
                flag = True
                while flag is True:
                    # total_search_count_random += len(self.state.board.getValidMoves())
                    # print("In this turn, Random AI search time is:", len(self.state.board.getValidMoves()))
                    rand = self.state.board.randomMove()
                    move = rand
                    list2.append(move)
                    list3.append(move)
                    print("Random AI Move: %s" % (move,))
                    print("------------------------------------------", end='\n')
                    if self.state.board.isValidMove(move):
                        flag = None
                    else:
                        print("Spot Taken or Out of bounds,please input again!")
                self.state = self.state.createNewState(move, initial)
                self.printCurrentBoard()

            if self.state.isWinner(initial, self.chain) is True:
                self.winner = initial
               
                print("Tree-based AI Score:", enemy_score)
                print("Random AI Score:", my_score)
                print("Tree-based AI Search Number:", total_search_count_ai)
                print("Random AI Search Number:", total_search_count_random)

                a=open('log.txt', 'a')
                #a.write('追加写入')
                a.write("Tree-based AI Score:")
                a.write(str(enemy_score))
                a.write("\n")
                a.write("Random AI Score:")
                a.write(str(my_score))
                a.write("\n")
                a.write("Tree-based AI Search Number:")
                a.write(str(total_search_count_ai))
                a.write("\n")
                # a.write("Random AI Search Number:")
                # a.write(str(total_search_count_random))
                a.write("\n")
                a.write("\n")
                a.close()


            # Random delete
            if (i==1) and (self.current==self.player_o):
                    rand = self.state.board.randomDelete()
                    print("------------------------------------------", end='\n')
                    print("           Random Delete",rand,"          ", end='\n')
                    print("------------------------------------------", end='\n')
                    #print("Random Delete",rand)
                    self.state = self.state.createDeleteState(rand, self.current)
                    self.printCurrentBoard()
            if (i==2) and (self.current==self.player_x):
                    rand = self.state.board.randomDelete()
                    print("------------------------------------------", end='\n')
                    print("           Random Delete",rand,"          ", end='\n')
                    print("------------------------------------------", end='\n')
                    self.state = self.state.createDeleteState(rand, self.current)
                    self.printCurrentBoard()
            if (i==1) and (self.current==self.player_x):
                    print("------------------------------------------", end='\n')
                    print("             No Random Delete!            ", end='\n')
                    print("------------------------------------------", end='\n')
                    #print("No Random Delete!")
            if (i==2) and (self.current==self.player_o):
                    print("------------------------------------------", end='\n')
                    print("             No Random Delete!            ", end='\n')
                    print("------------------------------------------", end='\n')
            

            # Exchange order
            self.current = self.swapTurn(initial)

        if self.winner is not None:
            self.printWinMessage()
        else:
            self.printGameEnded()
            self.state.board.printBoard()
            print("------------------------------------------", end='\n')
            print("                 Tie                      ", end='\n')
            print("------------------------------------------", end='\n')
    
    # AI(Tree-based) vs. AI(CNN)####################################################
    def TreeVsCNN(self):
        # 在这里把模型加载进来

        nnn = PolicyValueNet()
        nnn.policy_value_net = torch.load('model.pth')
        
        # pass
        print("Start:                                   ", end='\n')
        print(" 1. Player AI(Tree-based) (X)                        ", end='\n')
        print(" 2. Player AI(CNN) (O)                     ", end='\n')

        #i = int(input("Please choose starting player: "))
        i=1
        if i == 1:
            self.minimax_val = 2
            self.initial = self.player_x
            self.oponent = self.player_o
            # Initial move randomly
            rand = self.state.board.randomMove()
            self.state = self.state.createNewState(rand, self.initial)
            print("------------------------------------------", end='\n')
            print("Initial AI Move: %s" % (rand,))
            print("------------------------------------------", end='\n')
            self.printCurrentBoard()
            self.current = self.oponent
        elif i == 2:
            self.minimax_val = 1
            self.initial = self.player_o
            self.oponent = self.player_x
            self.current = self.initial
            self.printCurrentBoard()
        else:
            print("Only Two Players Available")
            sys.exit()

        while self.isOver() is not True:
            # Notice
            initial = self.current

            # AI(Tree-based) First
            if initial.piece == 'X':
                move = self.ai()
                # print("move的结构是:",move)
                list1.append(move)
                list3.append(move)
                print("Tree-based AI Move: %s" % (move,))
                print("------------------------------------------", end='\n')
                self.state = self.state.createNewState(move, initial)
                self.printCurrentBoard()

            # AI(CNN)
            else:
                flag = True
                while flag is True:
                    tmp=self.state.board.boardList()
                    gpu = True
                    if gpu is True:
                        #print("test!!!")
                        
                        #print(tmp.size())
                        #pass
                        predict = nnn.classify(tmp.float().cuda())
                    else:
                        #pass
                        predict = nnn.classify(tmp.float())

                    # total_search_count_random += len(self.state.board.getValidMoves())
                    # print("In this turn, Random AI search time is:", len(self.state.board.getValidMoves()))
                    #rand = self.state.board.randomMove()
                    #print(predict,self.state.board.dimension)
                    move = (int(predict/self.state.board.dimension),int(predict%self.state.board.dimension))
                    
                    if self.state.board.isValidMove(move):
                        flag = None
                        list2.append(move)
                        list3.append(move)
                        print("CNN AI Move: %s" % (move,))
                        print("------------------------------------------", end='\n')
                        self.state = self.state.createNewState(move, initial)
                        self.printCurrentBoard()
                    else:
                        try:
                            rand = self.state.board.randomMove()
                            move=rand
                            list2.append(move)
                            list3.append(move)
                            print("CNN AI Move: %s" % (move,))
                            print("------------------------------------------", end='\n')
                            self.state = self.state.createNewState(move, initial)
                            self.printCurrentBoard()
                        #print("Spot Taken or Out of bounds,please input again!")

                        except Exception:
                            print("Tree-based AI Score:", enemy_score)
                            print("CNN AI Score:", my_score)
                            c=open('battle.txt', 'a')
                            #a.write('追加写入')
                            c.write("Tree-based AI Score:")
                            c.write(str(enemy_score))
                            c.write("\n")
                            c.write("CNN AI Score:")
                            c.write(str(my_score))
                            c.write("\n")
                            c.write("\n")
                            c.write("\n")
                            c.close()
                            sys.exit()

            

            if (self.state.isWinner(initial, self.chain) is True) or (self.isOver is True):
                self.winner = initial
               
                print("Tree-based AI Score:", enemy_score)
                print("CNN AI Score:", my_score)

                c=open('battle.txt', 'a')
                #a.write('追加写入')
                c.write("Tree-based AI Score:")
                c.write(str(enemy_score))
                c.write("\n")
                c.write("CNN AI Score:")
                c.write(str(my_score))
                c.write("\n")
                c.write("\n")
                c.write("\n")
                c.close()

            # Random delete
            if (i==1) and (self.current==self.player_o):
                    rand = self.state.board.randomDelete()
                    print("------------------------------------------", end='\n')
                    print("           Random Delete",rand,"          ", end='\n')
                    print("------------------------------------------", end='\n')
                    #print("Random Delete",rand)
                    self.state = self.state.createDeleteState(rand, self.current)
                    self.printCurrentBoard()
            if (i==2) and (self.current==self.player_x):
                    rand = self.state.board.randomDelete()
                    print("------------------------------------------", end='\n')
                    print("           Random Delete",rand,"          ", end='\n')
                    print("------------------------------------------", end='\n')
                    self.state = self.state.createDeleteState(rand, self.current)
                    self.printCurrentBoard()
            if (i==1) and (self.current==self.player_x):
                    print("------------------------------------------", end='\n')
                    print("             No Random Delete!            ", end='\n')
                    print("------------------------------------------", end='\n')
                    #print("No Random Delete!")
            if (i==2) and (self.current==self.player_o):
                    print("------------------------------------------", end='\n')
                    print("             No Random Delete!            ", end='\n')
                    print("------------------------------------------", end='\n')
            

            # Exchange order
            self.current = self.swapTurn(initial)

        if self.winner is not None:
            self.printWinMessage()
        else:
            self.printGameEnded()
            self.state.board.printBoard()
            print("------------------------------------------", end='\n')
            print("                 Tie                      ", end='\n')
            print("------------------------------------------", end='\n')


    def printCurrentBoard(self):
        print("------------------------------------------", end='\n')
        print("              Current Board               ")
        print("------------------------------------------", end='\n')
        self.state.board.printBoard()
        print("------------------------------------------", end='\n')

    def displayMenu(self):
        print("------------------------------------------", end='\n')
        print(" Welcome to Gomoku (with random delete) ! ", end='\n')
        print("------------------------------------------", end='\n')
        print("Modes:                                    ")
        print(" 1. Human vs. Human                       ")
        print(" 2. Human vs. AI (Tree-based)             ")
        print(" 3. Human vs. AI (Random)                 ")
        print(" 4. AI (Tree-based) vs. AI (Random)       ")
        print(" 5. AI (Tree-based) vs. AI (CNN)          ")
        print("------------------------------------------", end='\n')

    def printWinMessage(self):
        print("------------------------------------------", end='\n')
        self.state.board.printBoard()
        print("------------------------------------------", end='\n')
        print("         The Winner is: Player %s         " % self.winner.piece)
        print("------------------------------------------", end='\n')

    def printGameEnded(self):
        print("------------------------------------------", end='\n')
        print("              Game Ended!                 ")
        print("------------------------------------------", end='\n')

    def isOver(self):
        return self.winner is not None or\
            not self.state.board.getValidMoves()
    
    # 旧版minimax，效果不好，弃用#######################################
    def alphaBetaSearch(self, player, depth):
        alpha = float('-inf')
        beta = float('inf')

        valid = self.state.getValidTransitions(player)
        #print("Tree nodes number is:",len(valid))

        utility = self.maxValue(self.state, player,
                                alpha, beta, depth-1)

        return utility

    def minValue(self, state, player, alpha, beta, depth):
        if depth == 0:
            return state.heuristic(self, player)
        else:
            valid = state.getValidTransitions(player)
            utility = float('inf')
            for move, state in valid:
                utility = min(utility,
                              self.maxValue(state, self.swapInitial(player),
                                            alpha, beta, depth-1))
                if utility <= alpha:
                    return utility
                beta = min(beta, utility)

            return utility

    def maxValue(self, state, player, alpha, beta, depth):
        if depth == 0:
            return state.heuristic(self, player)
        else:
            valid = state.getValidTransitions(player)
            utility = float('-inf')
            for move, state in valid:
                utility = max(utility,
                              self.minValue(state, self.swapInitial(player),
                                            alpha, beta, depth-1))
                self.minimax_moves.append((utility, move))
                if utility >= beta:
                    return utility
                alpha = max(alpha, utility)

            return utility
    
    def getMaxMove(self):
        max_move = max(self.minimax_moves, key=lambda item: item[0])[1]
        # print(max_move)
        return max_move
    #####################################################################
   
    def ai(self):
        global cut_count   # 统计剪枝次数
        cut_count = 0
        global search_count   # 统计搜索次数
        search_count = 0
        global total_search_count_ai
        total_search_count_ai=0
        self.negamax(True, DEPTH, -99999999, 99999999)
        print("In this trun, Tree-based purning time is:" + str(cut_count))
        print("In this trun, Tree-based search time is:" + str(search_count))
        total_search_count_ai += search_count
        return next_point[0], next_point[1]
    
    # 负值极大算法搜索 alpha + beta剪枝
    def negamax(self,is_ai, depth, alpha, beta):
        # 游戏是否结束 | | 探索的递归深度是否到边界
        if self.game_win(list1) or self.game_win(list2) or depth == 0:
            return self.evaluation(is_ai)

        blank_list = self.state.board.getValidMoves()
        self.order(blank_list)   # 搜索顺序排序  提高剪枝效率
        # 遍历每一个候选步
        for next_step in blank_list:

            global search_count
            search_count += 1

            # 如果要评估的位置没有相邻的子， 则不去评估  减少计算
            if not self.has_neightnor(next_step):
                continue

            if is_ai:
                list1.append(next_step)
            else:
                list2.append(next_step)
            list3.append(next_step)

            value = -self.negamax(not is_ai, depth - 1, -beta, -alpha)
            if is_ai:
                list1.remove(next_step)
            else:
                list2.remove(next_step)
            list3.remove(next_step)

            if value > alpha:
                # print(str(value) + "alpha:" + str(alpha) + "beta:" + str(beta))
                # print(list3)
                if depth == DEPTH:
                    next_point[0] = next_step[0]
                    next_point[1] = next_step[1]
                # alpha + beta剪枝点
                if value >= beta:
                    global cut_count
                    cut_count += 1
                    return beta
                alpha = value

        return alpha

    #  离最后落子的邻居位置最有可能是最优点
    def order(self,blank_list):
        last_pt = list3[-1]
        for item in blank_list:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    if (last_pt[0] + i, last_pt[1] + j) in blank_list:
                        blank_list.remove((last_pt[0] + i, last_pt[1] + j))
                        blank_list.insert(0, (last_pt[0] + i, last_pt[1] + j))

    def has_neightnor(self,pt):
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if (pt[0] + i, pt[1]+j) in list3:
                    return True
        return False   

    # 评估函数
    def evaluation(self,is_ai):
        total_score = 0

        if is_ai:
            my_list = list1
            enemy_list = list2
        else:
            my_list = list2
            enemy_list = list1

        # 算自己的得分
        score_all_arr = []  # 得分形状的位置 用于计算如果有相交 得分翻倍
        global my_score
        my_score=0
        for pt in my_list:
            m = pt[0]
            n = pt[1]
            my_score += self.cal_score(m, n, 0, 1, enemy_list, my_list, score_all_arr)
            my_score += self.cal_score(m, n, 1, 0, enemy_list, my_list, score_all_arr)
            my_score += self.cal_score(m, n, 1, 1, enemy_list, my_list, score_all_arr)
            my_score += self.cal_score(m, n, -1, 1, enemy_list, my_list, score_all_arr)

        #  算敌人的得分， 并减去
        score_all_arr_enemy = []
        global enemy_score 
        enemy_score = 0
        for pt in enemy_list:
            m = pt[0]
            n = pt[1]
            enemy_score += self.cal_score(m, n, 0, 1, my_list, enemy_list, score_all_arr_enemy)
            enemy_score += self.cal_score(m, n, 1, 0, my_list, enemy_list, score_all_arr_enemy)
            enemy_score += self.cal_score(m, n, 1, 1, my_list, enemy_list, score_all_arr_enemy)
            enemy_score += self.cal_score(m, n, -1, 1, my_list, enemy_list, score_all_arr_enemy)

        total_score = my_score - enemy_score*ratio*0.1

        return total_score


    # 每个方向上的分值计算
    def cal_score(self,m, n, x_decrict, y_derice, enemy_list, my_list, score_all_arr):
        add_score = 0  # 加分项
        # 在一个方向上， 只取最大的得分项
        max_score_shape = (0, None)

        # 如果此方向上，该点已经有得分形状，不重复计算
        for item in score_all_arr:
            for pt in item[1]:
                if m == pt[0] and n == pt[1] and x_decrict == item[2][0] and y_derice == item[2][1]:
                    return 0

        # 在落子点 左右方向上循环查找得分形状
        for offset in range(-5, 1):
            # offset = -2
            pos = []
            for i in range(0, 6):
                if (m + (i + offset) * x_decrict, n + (i + offset) * y_derice) in enemy_list:
                    pos.append(2)
                elif (m + (i + offset) * x_decrict, n + (i + offset) * y_derice) in my_list:
                    pos.append(1)
                else:
                    pos.append(0)
            tmp_shap5 = (pos[0], pos[1], pos[2], pos[3], pos[4])
            tmp_shap6 = (pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])

            for (score, shape) in shape_score:
                if tmp_shap5 == shape or tmp_shap6 == shape:
                    if tmp_shap5 == (1,1,1,1,1):
                        pass
                        #print('wwwwwwwwwwwwwwwwwwwwwwwwwww')
                    if score > max_score_shape[0]:
                        max_score_shape = (score, ((m + (0+offset) * x_decrict, n + (0+offset) * y_derice),
                                                (m + (1+offset) * x_decrict, n + (1+offset) * y_derice),
                                                (m + (2+offset) * x_decrict, n + (2+offset) * y_derice),
                                                (m + (3+offset) * x_decrict, n + (3+offset) * y_derice),
                                                (m + (4+offset) * x_decrict, n + (4+offset) * y_derice)), (x_decrict, y_derice))

        # 计算两个形状相交， 如两个3活 相交， 得分增加 一个子的除外
        if max_score_shape[1] is not None:
            for item in score_all_arr:
                for pt1 in item[1]:
                    for pt2 in max_score_shape[1]:
                        if pt1 == pt2 and max_score_shape[0] > 10 and item[0] > 10:
                            add_score += item[0] + max_score_shape[0]

            score_all_arr.append(max_score_shape)

        return add_score + max_score_shape[0]
    
    def game_win(self,list):
        for m in range(COLUMN):
            for n in range(ROW):

                if n < ROW - 4 and (m, n) in list and (m, n + 1) in list and (m, n + 2) in list and (
                        m, n + 3) in list and (m, n + 4) in list:
                    return True
                elif m < ROW - 4 and (m, n) in list and (m + 1, n) in list and (m + 2, n) in list and (
                            m + 3, n) in list and (m + 4, n) in list:
                    return True
                elif m < ROW - 4 and n < ROW - 4 and (m, n) in list and (m + 1, n + 1) in list and (
                            m + 2, n + 2) in list and (m + 3, n + 3) in list and (m + 4, n + 4) in list:
                    return True
                elif m < ROW - 4 and n > 3 and (m, n) in list and (m + 1, n - 1) in list and (
                            m + 2, n - 2) in list and (m + 3, n - 3) in list and (m + 4, n - 4) in list:
                    return True
        return False

    

    
        

        
        
    
    





    



