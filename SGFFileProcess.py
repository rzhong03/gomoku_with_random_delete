#coding:utf-8
import time
import os
import numpy as np

# 全局变量 控制训练的大小
training_size=int(input("Please input the training size: "))   
#training_size=6

class SGFflie():
    def __init__(self):
        """
        初始化：
        POS：棋盘坐标的对应字母顺序
        """
        self.POS = 'abcdefghijklmno'

    def openfile(self, filepath):
        """打开文件,读取棋谱"""
        f = open(filepath, 'r', newline='', encoding='ISO-8859-1')
        data = f.read()

        #分割数据
        effective_data = data.split(';')
        s = effective_data[2:-1]

        board = []
        step = 0
        for point in s:
            x = self.POS.find(point[2])
            y = self.POS.find(point[3])
            color = step % 2
            step += 1
            board.append([x, y, color, step])

        f.close()

        return board

    def createTraindataFromqipu(self, path, color=0):
        """将棋谱中的数据生成神经网络训练需要的数据"""
        qipu = self.openfile(path)

        bla = qipu[::2]
        whi = qipu[1::2]
        bla_step = len(bla)
        whi_step = len(whi)

        train_x = []
        train_y = []

        if color == 0:
            temp_x = [0.0 for i in range(225)]
            for index in range(bla_step):
                _x = [0.0 for i in range(225)]
                _y = [0.0 for i in range(225)]
                if index == 0:
                    lx = []
                    lx.append(_x)
                    train_x.append(_x)
                    _y[bla[index][0]*15 + bla[index][1]] = 2.0
                    ly = []
                    ly.append(_y)
                    train_y.append(_y)
                else:
                    _x = temp_x.copy()
                    lx = []
                    lx.append(_x)
                    train_x.append(_x)
                    _y[bla[index][0] * 15 + bla[index][1]] = 2.0
                    ly = []
                    ly.append(_y)
                    train_y.append(_y)

                temp_x[bla[index][0] * 15 + bla[index][1]] = 2.0
                if index < whi_step:
                    temp_x[whi[index][0] * 15 + whi[index][1]] = 1.0
        
        # 数据清洗,根据训练大小进行数据清洗,主要选取15*15的切片中的中心部分
        
        train_x=(np.array(train_x))
        train_x=train_x.reshape(len(train_x),15,15)
        train_x=train_x[:,int(8-training_size/2):int(8-training_size/2+training_size),int(8-training_size/2):int(8-training_size/2+training_size)]
        train_x=train_x.reshape(len(train_x),1,training_size**2)
        train_x=train_x.tolist()

        train_y=(np.array(train_y))
        train_y=train_y.reshape(len(train_y),15,15)
        train_y=train_y[:,int(8-training_size/2):int(8-training_size/2+training_size),int(8-training_size/2):int(8-training_size/2+training_size)]
        train_y=train_y.reshape(len(train_y),1,training_size**2)
        train_y=train_y.tolist()
    
        return train_x, train_y
    

    def createTraindata(self, path):
        """生成训练数据"""
        filepath = self.allFileFromDir(path)
        train_x = []
        train_y = []
        for path in filepath:
            x, y = self.createTraindataFromqipu(path)

            train_x.extend(x)
            train_y.extend(y)
        #print(train_x, train_y)
        return train_x, train_y

    def allFileFromDir(self, Dirpath):
        """获取文件夹中所有文件的路径"""
        pathDir = os.listdir(Dirpath)
        pathfile = []
        for allDir in pathDir:
            child = os.path.join('%s%s' % (Dirpath, allDir))
            pathfile.append(child)
        return pathfile


