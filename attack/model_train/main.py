#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""The codes which contain 3 models: st-gcn, a3t-gcn, st-gcn,
for t-gcn and a3t-gcn, the codes are based on codes in https://github.com/lehaifeng/T-GCN.
for st-gcn, the codes are based on codes in https://github.com/VeritasYin/STGCN_IJCAI-18.
"""
# import pickle as pkl
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
from input_data import preprocess_data, load_los_data, load_hk_data
from tgcn import tgcnCell
from sample import Sampler
from utils import sparse_to_tuple
from scipy import sparse
#from gru import GRUCell
from visualization import plot_result, plot_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
#import matplotlib.pyplot as plt
import time
from math_graph import scaled_laplacian, cheb_poly_approx
from utils import calculate_laplacian

time_start = time.time()
###### Settings ######
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('training_epoch', 1, 'Number of epochs to train.')
flags.DEFINE_integer('gru_units', 100, 'hidden units of gru.')
flags.DEFINE_integer('batch_size', 32, 'batch_size.')
flags.DEFINE_integer('seq_len', 12, '  time length of inputs.')
flags.DEFINE_float('train_rate', 0.8, 'rate of training set.')
flags.DEFINE_string('dataset', 'hk', 'los or hk.')
flags.DEFINE_string('model_name', 'stgcn', 'stgcn or tgcn or tgcn_att')
flags.DEFINE_string('dropmode', 'out', 'node or edge or out')
flags.DEFINE_float('out_keep_prob', 1.0, 'dropout keep rate')  # if less than 1, ne_keep_prob should be 1
flags.DEFINE_float('ne_keep_prob', 1.0, 'dropnode or dropedge keep rate')  # if less than 1, out_keep_prob should be 1
model_name = FLAGS.model_name
dropmode = FLAGS.dropmode
data_name = FLAGS.dataset
train_rate = FLAGS.train_rate
seq_len = FLAGS.seq_len
lr = FLAGS.learning_rate
training_epoch = FLAGS.training_epoch
gru_units = FLAGS.gru_units
batch_size = FLAGS.batch_size
ne_keep_prob = FLAGS.ne_keep_prob
out_keep_prob = FLAGS.out_keep_prob

if model_name == 'stgcn':
    pre_len = 1
elif model_name == 'tgcn' or 'tgcn_att':
    pre_len = 3

output_dim = pre_len

if dropmode == 'out':
    k = out_keep_prob  # k is a marker of model file
elif dropmode == 'node' or 'edge':
    k = ne_keep_prob

blocks = [[1, 32, 64], [64, 32, 128]]
Ks = 3
Kt = 3

###### load data ######
if data_name == 'los':
    data, adj = load_los_data('los')
if data_name == 'hk':
    data, adj = load_hk_data('hk')

adj_sampler = Sampler(adj)
time_len = data.shape[0]
num_nodes = data.shape[1]
data1 = np.mat(data, dtype=np.float32)

#### normalization
max_value = np.max(data1)
data1 = data1/max_value
trainX, trainY, testX, testY = preprocess_data(data1, time_len, train_rate, seq_len, pre_len)
totalbatch = int(trainX.shape[0]/batch_size)
training_data_count = len(trainX)

def TGCN(_X, _weights, _biases, tmp_adj, keep_rate):
    ###
    tmp_adj = tf.sparse.from_dense(tmp_adj)
    cell_1 = tgcnCell(gru_units, tmp_adj, keep_rate, num_nodes=num_nodes)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
    m = []
    for i in outputs:
        o = tf.reshape(i, shape=[-1, num_nodes, gru_units])
        o = tf.reshape(o, shape=[-1, gru_units])
        m.append(o)
    last_output = m[-1]
    output = tf.matmul(last_output, _weights['out']) + _biases['out']
    output = tf.reshape(output, shape=[-1, num_nodes, pre_len])
    output = tf.transpose(output, perm=[0, 2, 1])
    output = tf.reshape(output, shape=[-1, num_nodes])
    return output, m, states

def STGCN(_X, graph_kernel, keep_rate):
    # from math_graph import scaled_laplacian, cheb_poly_approx
    from base_model import build_model
    # W = np.array(tmp_adj).astype(np.float32)
    # print (tf.get_collection("tmp_adj"))
    # W = np.array(tf.get_collection("tmp_adj")[0]).astype(np.float32)
    # n = num_nodes
    n_his = seq_len
    # L = scaled_laplacian(W)
    # # Alternative approximation method: 1st approx - first_approx(W, n).
    # Lk = cheb_poly_approx(L, Ks, n)
    tf.add_to_collection(name='graph_kernel', value=tf.cast(graph_kernel, tf.float32))
    x = tf.expand_dims(_X, axis=3)
    train_loss, pred = build_model(x, n_his, Ks, Kt, blocks, keep_rate)
    pred = tf.transpose(pred, perm=[0, 2, 1])
    pred = tf.reshape(pred, shape=[-1, num_nodes])
    return train_loss, pred

def TGCN_att(_X, weights, biases, tmp_adj, keep_rate):
    ###
    tmp_adj = tf.sparse.from_dense(tmp_adj)
    cell_1 = tgcnCell(gru_units, tmp_adj, keep_rate, num_nodes=num_nodes)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)

    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)

    out = tf.concat(outputs, axis=0)
    out = tf.reshape(out, shape=[seq_len, -1, num_nodes, gru_units])
    out = tf.transpose(out, perm=[1, 0, 2, 3])

    last_output, alpha = self_attention1(out, weight_att, bias_att)

    output = tf.reshape(last_output, shape=[-1, seq_len])
    output = tf.matmul(output, weights['out']) + biases['out']
    output = tf.reshape(output, shape=[-1, num_nodes, pre_len])
    output = tf.transpose(output, perm=[0, 2, 1])
    output = tf.reshape(output, shape=[-1, num_nodes])

    return output, outputs, states

def self_attention1(x, weight_att, bias_att):
    x = tf.matmul(tf.reshape(x, [-1, gru_units]), weight_att['w1']) + bias_att['b1']
    f = tf.matmul(tf.reshape(x, [-1, num_nodes]), weight_att['w2']) + bias_att['b2']
    g = tf.matmul(tf.reshape(x, [-1, num_nodes]), weight_att['w2']) + bias_att['b2']
    h = tf.matmul(tf.reshape(x, [-1, num_nodes]), weight_att['w2']) + bias_att['b2']

    f1 = tf.reshape(f, [-1, seq_len])
    g1 = tf.reshape(g, [-1, seq_len])
    h1 = tf.reshape(h, [-1, seq_len])
    s = g1 * f1

    beta = tf.nn.softmax(s, dim=-1)  # attention map

    context = tf.expand_dims(beta, 2) * tf.reshape(x, [-1, seq_len, num_nodes])

    context = tf.transpose(context, perm=[0, 2, 1])
    return context, beta

###### placeholders ######
inputs = tf.placeholder(tf.float32, shape=[None, seq_len, num_nodes], name='inputs')
labels = tf.placeholder(tf.float32, shape=[None, pre_len, num_nodes], name='labels')
graph_kernel = tf.placeholder(tf.float32, shape=[num_nodes, Ks * num_nodes], name='graph_kernel')
tf_tmp_adj = tf.placeholder(tf.float32, shape=[num_nodes, num_nodes], name='tf_tmp_adj')
tf_keep_prob = tf.placeholder(tf.float32, name='tf_keep_prob')


if model_name == 'tgcn':
    # Graph weights
    weights = {
        'out': tf.Variable(tf.random_normal([gru_units, pre_len], mean=1.0), name='weight_o')}
    biases = {
        'out': tf.Variable(tf.random_normal([pre_len]), name='bias_o')}
    pred, ttts, ttto = TGCN(inputs, weights, biases, tf_tmp_adj, tf_keep_prob)
elif model_name == 'stgcn':
    stgcn_loss, pred = STGCN(inputs, graph_kernel, tf_keep_prob)
elif model_name == 'tgcn_att':
    # Graph weights
    weights = {
        'out': tf.Variable(tf.random_normal([seq_len, pre_len], mean=1.0), name='weight_o')}
    bias = {
        'out': tf.Variable(tf.random_normal([pre_len]), name='bias_o')}
    weight_att = {
        'w1': tf.Variable(tf.random_normal([gru_units, 1], stddev=0.1), name='att_w1'),
        'w2': tf.Variable(tf.random_normal([num_nodes, 1], stddev=0.1), name='att_w2')}
    bias_att = {
        'b1': tf.Variable(tf.random_normal([1]), name='att_b1'),
        'b2': tf.Variable(tf.random_normal([1]), name='att_b2')}
    pred, ttto, ttts = TGCN_att(inputs, weights, bias, tf_tmp_adj, tf_keep_prob)


y_pred = pred
tf.add_to_collection('y_pred', y_pred)


###### optimizer ######
lambda_loss = 0.0015
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
label = tf.reshape(labels, [-1, num_nodes])
##loss
loss = tf.reduce_mean(tf.nn.l2_loss(y_pred-label) + Lreg)
tf.add_to_collection('loss', loss)
##rmse
error = tf.sqrt(tf.reduce_mean(tf.square(y_pred-label)))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

###### Initialize session ######
variables = tf.global_variables()
saver = tf.train.Saver(tf.global_variables())
#sess = tf.Session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

out = 'out/%s'%(model_name)
#out = 'out/%s_%s'%(model_name,'perturbation')
path1 = '%s_%s_pre%r_epoch%r_drop%s_%r' % (data_name, model_name, pre_len, training_epoch, dropmode, k)
path = os.path.join(out, path1)
if not os.path.exists(path):
    os.makedirs(path)

###### evaluation ######
def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a, b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a-b, 'fro')/la.norm(a, 'fro')
    r2 = 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
    var = 1-(np.var(a-b))/np.var(a)
    return rmse, mae, 1-F_norm, r2, var


x_axe, batch_loss, batch_rmse, batch_pred = [], [], [], []
test_loss, test_rmse, test_mae, test_acc, test_r2, test_var, test_pred = [], [], [], [], [], [], []

# STGCN
n = num_nodes
n_his = seq_len
keep_rate = 1
total_W = np.array(adj_sampler.adj.todense()).astype(np.float32)
L = scaled_laplacian(total_W)
total_LK = cheb_poly_approx(L, Ks, n)

# TGCN or TGCN_att
TGCN_total_adj = calculate_laplacian(adj_sampler.adj.todense()).toarray()


for epoch in range(training_epoch):
    if dropmode == 'node':
        tmp_adj = adj_sampler.randomedge_sampler(ne_keep_prob)
    elif dropmode == 'edge':
        tmp_adj = adj_sampler.randomvertex_sampler(ne_keep_prob)
    elif dropmode == 'out':
        tmp_adj = adj_sampler.adj.todense()

    W = np.array(tmp_adj).astype(np.float32)
    L = scaled_laplacian(W)
    # Alternative approximation method: 1st approx - first_approx(W, n).
    Lk = cheb_poly_approx(L, Ks, n)
    # tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))

    TGCN_tmp_adj = calculate_laplacian(tmp_adj).toarray()
    # print (type(TGCN_tmp_adj), TGCN_tmp_adj)
    # print (type(Lk))
    for m in range(totalbatch):
        mini_batch = trainX[m * batch_size: (m+1) * batch_size]
        mini_label = trainY[m * batch_size: (m+1) * batch_size]
        _, loss1, rmse1, train_output = sess.run([optimizer, loss, error, y_pred],
                                                 feed_dict={inputs: mini_batch, labels: mini_label, graph_kernel: Lk,
                                                            tf_tmp_adj: TGCN_tmp_adj, tf_keep_prob: out_keep_prob})
        batch_loss.append(loss1)
        batch_rmse.append(rmse1 * max_value)
     # Test completely at every epoch
    loss2, rmse2, test_output = sess.run([loss, error, y_pred],
                                         feed_dict={inputs: testX, labels: testY, graph_kernel: total_LK,
                                                    tf_tmp_adj: TGCN_total_adj, tf_keep_prob: 1.0})
                                                    # close drop when calculating test loss
    test_label = np.reshape(testY, [-1, num_nodes])
    rmse, mae, acc, r2_score, var_score = evaluation(test_label, test_output)
    test_label1 = test_label * max_value
    test_output1 = test_output * max_value
    test_loss.append(loss2)
    test_rmse.append(rmse * max_value)
    test_mae.append(mae * max_value)
    test_acc.append(acc)
    test_r2.append(r2_score)
    test_var.append(var_score)
    test_pred.append(test_output1)

    print('Iter:{}'.format(epoch),
          'train_rmse:{:.4}'.format(batch_rmse[-1]),
          'test_loss:{:.4}'.format(loss2),
          'test_rmse:{:.4}'.format(rmse),
          'test_acc:{:.4}'.format(acc))

    if (epoch % 301 == 0):
        saver.save(sess, path+'/model_100/%s_%s_pre_%r_epoch%r_drop%s_%r' % (data_name, model_name, pre_len,
                                                                             epoch, dropmode, k), global_step=epoch)
        mytxt = open('%s_%s_pre%r_epoch%r_drop%s_%r.txt' % (data_name, model_name, pre_len,
                                                            training_epoch, dropmode, k), mode='a', encoding='utf-8')
        print('Iter:', epoch, 'train_rmse:', batch_rmse[-1], 'test_loss:', loss2,
              'test_rmse:', rmse, 'test_acc:', acc, file=mytxt)
        mytxt.close()


time_end = time.time()
print(time_end-time_start, 's')

############## visualization ###############
b = int(len(batch_rmse)/totalbatch)
batch_rmse1 = [i for i in batch_rmse]
train_rmse = [(sum(batch_rmse1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]
batch_loss1 = [i for i in batch_loss]
train_loss = [(sum(batch_loss1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]

index = test_rmse.index(np.min(test_rmse))
test_result = test_pred[index]
var = pd.DataFrame(test_result)
var.to_csv(path+'/test_result.csv', index=False, header=False)
plot_result(test_result, test_label1, path)
plot_error(train_rmse, train_loss, test_rmse, test_acc, test_mae, path)

print('min_rmse:%r'%(np.min(test_rmse)),
      'min_mae:%r'%(test_mae[index]),
      'max_acc:%r'%(test_acc[index]),
      'r2:%r'%(test_r2[index]),
      'var:%r'%test_var[index])




