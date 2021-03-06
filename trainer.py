#coding:utf-8
import sys
import torch.utils.data as data
from SGFFileProcess import *
import torch.nn.functional as F
from draw import *
import torch
import numpy as np
from datapacker import dataloader

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('log')

cuda = torch.device('cuda')
from torchCNN import *
sgf = SGFflie()
nnn = PolicyValueNet()

def train(gpu = True):
    loader = dataloader('training/',3 )
    for step, (board, nextmove) in enumerate(loader):
        print(board.size())
        # feed in nn
        if gpu is True:
            nnn.train(board.float().cuda(),nextmove.cuda())
        else:
            nnn.train(board.float(), nextmove)


    return

def evaluate(path , gpu = True):
    correct =0
    total = 0
    around1 = 0
    around2 = 0
    loader = dataloader(path, 1)
    for step, (board, nextmove) in enumerate(loader):

        if gpu is True:
            predict = nnn.classify(board.float().cuda())
        else:
            predict = nnn.classify(board.float())

        b=open('scatter.txt', 'a')#追加写入
        b.write("x:")
        b.write(str(predict))
        b.write("y:")
        b.write(str(nextmove.item()))
        b.write("\n")
        b.close()

        correct += 1 if (predict == nextmove.item()) else 0
        b16, b25 = around(predict,nextmove.item())
        around1+=b16
        around2+=b25
        total += 1
        print("\rEvaluating: ", step, end="")

    return  (correct/total , around1/total, around2/total)

# see if predicted value is around the label
def around(predict, label):
    offset = abs(predict- label)
    b16  = 1 if (offset<=16 and offset>=14 )or offset <=1 else 0

    b25 = 1 if (offset <= 32 and offset >= 28) or offset <= 2 or (offset <= 17 and offset >= 13) \
                  else 0

    return b16, b25




if __name__ == '__main__':


    torch.cuda.empty_cache()
    accuracies = [[] for i in range(6)]

    for i in range(100):
        ## begin to train data.
        train()
        print("\nEpoch done", i)

        #  nnn.policy_value_net = torch.load('model.pth')
        # evaluate every 2 epochs

        training_accuracy, training_around1_accuracy,training_around2_accuracy = evaluate('training/')
        testing_accuracy,testing_around1_accuracy,testing_around2_accuracy = evaluate('testing/')

        print("\ntraining_accuracy: " ,training_accuracy)
        print("\naround1_accuracy: ", training_around1_accuracy)
        print("\naround2_accuracy: ", training_around2_accuracy)

        print("\ntesting_accuracy: ",testing_accuracy)
        print("\naround1_accuracy: ", testing_around1_accuracy)
        print("\naround2_accuracy: ", testing_around2_accuracy)
        #draw
        accuracies[0].append(training_accuracy)
        accuracies[1].append(testing_accuracy)
        accuracies[2].append(training_around1_accuracy)
        accuracies[3].append(testing_around1_accuracy)
        accuracies[4].append(training_around2_accuracy)
        accuracies[5].append(testing_around2_accuracy)

        #time.sleep(0.003)
    torch.save(nnn.policy_value_net, 'model.pth')
    draw(accuracies)








