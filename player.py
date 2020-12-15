#coding:utf-8
from state import *


class Player(object):
    def __init__(self, piece):
        super(Player, self).__init__()
        self.piece = piece

    def getMove(self):
        print("Now, it is Player %s's Turn!" % self.piece)
        i = input("Move: ")
        return list(map(int, i.split()))

    def __str__(self):
        return "Player " + self.piece
