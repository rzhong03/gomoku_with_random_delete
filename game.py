#coding:utf-8
import sys
from gomoku import *


def main():
    if len(sys.argv) != 3:
        sys.exit('Usage: %s <board dimension>\
                 <winning chain length>' % sys.argv[0])

    game = Gomoku(int(sys.argv[1]), int(sys.argv[2]))

    game.displayMenu()
    game.mode = int(input("Please choose game mode: "))
    #game.mode =5
    game.start()

if __name__ == "__main__":
    main()
