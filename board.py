#coding:utf-8
import random
import torch

class Board(object):
    def __init__(self):
        super(Board, self).__init__()
        self.board = {}
        self.dimension = None

    def initializeBoard(self, dimension):
        self.dimension = dimension

        for row in range(dimension):
            for col in range(dimension):
                self.board[(row, col)] = '.'

    def printBoard(self):
        for row in range(self.dimension):
            for col in range(self.dimension):
                if (row, col) in self.board:
                    print(self.board[(row, col)], end=" ")
                else:
                    print(" ", end=" ")
            print("\n", end="")
    
    # 把board转化为张量，然后feed in CNN
    def boardList(self):
         board_tmp=torch.FloatTensor(1,1,self.dimension,self.dimension)
         for row in range(self.dimension):
            for col in range(self.dimension):
                if (row, col) in self.board:
                    if(self.board[(row, col)]=='.'):
                        board_tmp[0][0][row][col]=0.0
                    if(self.board[(row, col)]=='X'):
                        board_tmp[0][0][row][col]=1.0
                    if(self.board[(row, col)]=='O'):
                        board_tmp[0][0][row][col]=2.0
                    #print(self.board[(row, col)])
                else:
                    print(" ", end=" ")
         return board_tmp
            #print("\n", end="")


    def makeMove(self, move, player):
        if self.isValidMove(move):
            self.board[(move[0], move[1])] = player.piece
            return move
        else:
            return (-1, -1)
    
    # Implement the operation of deleting the pieces
    def deleteMove(self, move):
        self.board[(move[0], move[1])] = '.'
        # if self.isValidMove(move):
        #     self.board[(move[0], move[1])] = player.piece
        #     return move
        # else:
        #     return (-1, -1)


    def isValidMove(self, move):
        if move[0] >= 0 and move[0] < self.dimension  and move[1] >= 0 and move[1] < self.dimension:
            if self.board[(move[0], move[1])] == '.':
                return True
            else:
                return False
        return False

    def randomMove(self):
        valid = self.getValidMoves()

        rand = random.randint(0, len(valid)-1)

        return valid[rand]

    def getCloseMoves(self, row, col):
        moves = [(row - 1, col), (row + 1, col), (row, col + 1), (row, col - 1), (row + 1, col + 1), (row - 1, col - 1)]

        return moves


    def closeMove(self, move):
        close = self.getCloseMoves(move[0], move[1])

        rand = random.randint(0, len(close) - 1)

        move = close[rand]

        while self.isValidMove(move) is False:
            rand = random.randint(0, len(close) - 1)
            move = close[rand]

        return move


    def getValidMoves(self):
        valid = []
        for row in range(self.dimension):
            for col in range(self.dimension):
                move = (row, col)
                if self.isValidMove(move):
                    valid.append(move)

        # return reversed(valid)
        return valid

    #Get the position of the chess pieces already on the current board
    def getinValidMoves(self):
        invalid = []
        for row in range(self.dimension):
            for col in range(self.dimension):
                move = (row, col)
                if self.isValidMove(move) is not True:
                    invalid.append(move)

        # return reversed(valid)
        return invalid

    # Randomly select pieces to delete
    def randomDelete(self):
        invalid = self.getinValidMoves()
        #print(invalid)
        tmp = random.randint(0, len(invalid)-1)
        # print(tmp)
        # print(invalid[tmp])
        return invalid[tmp]
    
