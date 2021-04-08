# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
# import pickle as pkl


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


def preprocess_data(data, time_len, rate, seq_len, pre_len):
    train_size = int(time_len * rate)
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]

    trainX, trainY, testX, testY = [], [], [], []

    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0: seq_len])
        trainY.append(a[seq_len: seq_len + pre_len])
    for i in range(len(test_data) - seq_len - pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0 : seq_len])
        testY.append(b[seq_len : seq_len + pre_len])

    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    return trainX1, trainY1, testX1, testY1
