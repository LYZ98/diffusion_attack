# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
# import pickle as pkl

def load_sz_data(dataset):
    sz_adj = pd.read_csv(r'data/sz_adj.csv', header=None)
    adj = np.mat(sz_adj)
    sz_tf = pd.read_csv(r'data/sz_speed.csv')
    return sz_tf, adj

def load_los_data(dataset):
    los_adj = pd.read_csv(r'data/los_adj.csv', header=None)
    adj = np.mat(los_adj)
    los_tf = pd.read_csv(r'data/los_speed.csv')
    return los_tf, adj

def load_hk_data(dataset):
    hk_adj = pd.read_csv(r'data/hk_adj.csv', header=None)
    adj = np.mat(hk_adj)
    hk_tf = pd.read_csv(r'data/hk_speed.csv')
    return hk_tf, adj

def load_los_locations(dataset):
    los_locations = pd.read_csv(r'data/los_locations.csv', header=None)
    return np.array(los_locations)

def load_hk_locations(dataset):
    hk_locations = pd.read_csv(r'data/hk_locations.csv', header=None)
    return np.array(hk_locations)


def preprocess_data(data, time_len, rate, seq_len, pre_len):
    train_size = int(time_len * rate)
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]

    trainX, trainY, testX, testY = [], [], [], []

    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0: seq_len])
        trainY.append(a[seq_len: seq_len + pre_len])  # 切割a的后面部分，trainY增加一个元素（矩阵），代表预测的时间序列数据，长度pre_len
    for i in range(len(test_data) - seq_len - pre_len):
        b = test_data[i: i + seq_len + pre_len]  # 同理构造训练集
        testX.append(b[0 : seq_len])
        testY.append(b[seq_len : seq_len + pre_len])

    # 三阶张量的训练数据trainX， trainY已经构造完毕
    # a=[], b[1,2,3], c[4,5,6], a.append(b) = [[1,2,3]], a.append(c) = [[1,2,3],[4,5,6]]
    # b.append(c) = [1,2,3,[4,5,6]]

    # 这里将矩阵转化为array数组？
    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    return trainX1, trainY1, testX1, testY1
