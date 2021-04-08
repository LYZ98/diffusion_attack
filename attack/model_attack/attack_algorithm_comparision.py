"""
The codes are in https://github.com/LYZ98
"""


import tensorflow as tf
import numpy as np
import copy
from sample import Sampler
from math_graph import scaled_laplacian, cheb_poly_approx
from utils import calculate_laplacian
from SPSA import SPSA
import random
from SPSA import plot_progress
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
import os
from input_data import load_los_data, load_hk_data, load_hk_locations, load_los_locations
from diffusion_attack_tool import calculate_goal, find_attack_set_by_degree, find_neighbor, \
find_attack_set_by_kmedoids, find_attack_set_by_pagerank, \
find_attack_set_by_betweenness, calculate_total_budget_of_attack_set, \
calculate_degree_of_nodes_in_set, add_node, find_attack_set_by_Kg_pagerank, \
find_attack_set_by_Kg_betweenness



###### Settings ######
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('training_epoch', 301, 'Number of epochs to train.')
flags.DEFINE_integer('seq_len', 12, '  time length of inputs.')
# flags.DEFINE_integer('pre_len', 3, 'time length of prediction.')  # if choose 'stgcn' model, per_len have to be 1
flags.DEFINE_string('dataset', 'los', 'los or hk.')
flags.DEFINE_string('model_name', 'stgcn', 'stgcn or tgcn or tgcn_att')
flags.DEFINE_string('dropmode', 'out', 'node or edge or out')
flags.DEFINE_string('method', 'degree', 'spsa, Kg_spsa or degree or kmedoids or '
                                                       'pagerank or Kg_pagerank'
                                                       'or betweenness or Kg_betweenness or'
                                                       'or random')
flags.DEFINE_float('out_keep_prob', 1.0, 'out keep')  # if less than 1, ne_keep_prob should be 1. In our trained models, you could choose 0.9, 0.7, 0.5 or 1
flags.DEFINE_float('ne_keep_prob', 1.0, 'node edge keep')  # if less than 1, out_keep_prob should be 1. In our trained models, you could choose 0.9, 0.7, 0.5 or 1
model_name = FLAGS.model_name
dropmode = FLAGS.dropmode
data_name = FLAGS.dataset
seq_len = FLAGS.seq_len
training_epoch = FLAGS.training_epoch
ne_keep_prob = FLAGS.ne_keep_prob
out_keep_prob = FLAGS.out_keep_prob
method = FLAGS.method

if model_name == 'stgcn':
    pre_len = 1
elif model_name == 'tgcn' or 'tgcn_att':
    pre_len = 3

if dropmode == 'out':
    k = out_keep_prob
elif dropmode == 'node' or 'edge':
    k = ne_keep_prob

# load model(here we have trained 3 models in file "new_model")
# if load models, the file name should be the same as name in training
out = 'out/%s'%(model_name)
path1 = '%s_%s_pre%r_epoch%r_drop%s_%r' % (data_name, model_name, pre_len, training_epoch, dropmode, k)
path = os.path.join(out, path1)
path2 = '/model_100/%s_%s_pre_%r_epoch%r_drop%s_%r-%r.meta' % (data_name, model_name, pre_len, training_epoch -1, dropmode, k, training_epoch-1)
path3 = '/model_100/%s_%s_pre_%r_epoch%r_drop%s_%r-%r' % (data_name, model_name, pre_len, training_epoch -1, dropmode, k, training_epoch -1)
sess = tf.Session()
new_saver = tf.train.import_meta_graph(path + path2)
new_saver.restore(sess, path + path3)
y_pred = tf.get_collection('y_pred')[0]
graph = tf.get_default_graph()
inputs = graph.get_operation_by_name('inputs').outputs[0]  # load placeholder
labels = graph.get_operation_by_name('labels').outputs[0]
graph_kernel = graph.get_operation_by_name('graph_kernel').outputs[0]
tf_tmp_adj = graph.get_operation_by_name('tf_tmp_adj').outputs[0]
tf_keep_prob = graph.get_operation_by_name('tf_keep_prob').outputs[0]

# load data, include feature matrix, adjacency matrix,
# and location matrix(include longitude and latitude of each node)
if data_name == 'los':
    X2, A2 = load_los_data('los')
    locations = load_los_locations('los')
elif data_name == 'hk':
    X2, A2 = load_hk_data('hk')
    locations = load_hk_locations('hk')
else:
    print('error!')

blocks = [[1, 32, 64], [64, 32, 128]]
Ks = 3
Kt = 3
adj_sampler = Sampler(A2)
node_num = A2.shape[0]
total_W = np.array(adj_sampler.adj.todense()).astype(np.float32)
L = scaled_laplacian(total_W)
total_LK = cheb_poly_approx(L, Ks, node_num)
GCN_total_adj = calculate_laplacian(adj_sampler.adj.todense()).toarray()

X3 = np.mat(X2, dtype=np.float32)
max_value = np.max(X3)  # max speed
# min_value = np.min(X3)
# print("max_speed:", max_value, "min_speed:", min_value)
X3 = X3 / max_value  # normalize to 0~1

time_sample = [1890]  # time sample from feature matrix, here we adopt the average  as final result
B = [50]  # total budget
# time_sample = [1890, 1910, 1930, 1950, 1970]  # time sample from feature matrix, here we adopt the average  as final result
# B = [20, 50, 100, 150, 200]  # total budget
all_nodes = np.linspace(0, node_num-1, node_num)
all_nodes = all_nodes.astype(np.int32)
b = calculate_degree_of_nodes_in_set(A2, all_nodes)  # a vector, denotes cost of each node
w = np.ones(node_num)  # a vector, denotes value weight of each node, here we set it equal
attack_influence1 = np.zeros((len(time_sample), len(B)))
attack_influence2 = np.zeros((len(time_sample), len(B)))

for time_step in range(len(time_sample)):
    X4 = X3[time_sample[time_step]: time_sample[time_step] + seq_len]
    label1 = X3[time_sample[time_step] + seq_len: time_sample[time_step] + seq_len + pre_len]
    _X = []
    _X.append(X4[0:seq_len])
    A = np.array(A2)
    X = np.array(_X)
    label = np.array(label1)
    # get a prediction "pred" of input from trained neural network in this iteration (the input is a time_sample)
    pred = sess.run(y_pred, feed_dict={inputs: X, graph_kernel: total_LK,
                                       tf_tmp_adj: GCN_total_adj, tf_keep_prob: 1.0})
    adj = copy.deepcopy(A)
    if model_name == 'stgcn':
        pred = np.squeeze(pred)
        pred = np.expand_dims(pred, 0)
    # here, pred is a pre_len*node_num matrix,
    # the element (i,j) in this matrix denotes the prediction speed after/
    # i interval(from the final time in choosen time_sample) of node j

    def find_attack_set_by_random(b, B):
        chosen_nodes = []
        k = random.randint(0, node_num-1)
        while calculate_total_budget_of_attack_set(b, add_node([k], chosen_nodes)) < B:
            chosen_nodes.append(k)
            k = random.randint(0, node_num - 1)
        return np.array(chosen_nodes)

    # def find_attack_set_by_spsa_step(l, m, u, iter, b, B, w):
    #     chosen_nodes = []
    #     total_score_matrix = np.zeros([node_num, node_num])
    #     t = len(chosen_nodes)
    #     k = 0
    #     while calculate_total_budget_of_attack_set(b, chosen_nodes) < B:
    #         score_matrix = np.zeros((node_num, node_num))
    #         for i in range(node_num):
    #             if i in chosen_nodes:
    #                 continue
    #             else:
    #                 #  添加扰动
    #                 def add_perturbations(x):
    #                     Xp = copy.deepcopy(X)
    #                     Xp[0, :, i] = Xp[0, :, i] + x[0: seq_len]
    #                     for j in range(t):
    #                         Xp[0, :, chosen_nodes[j]] = Xp[0, :, chosen_nodes[j]] + x[(j + 1)*seq_len: (j+2)*seq_len]
    #                     return Xp
    #
    #                 def evaluation(x):
    #                     Xp = add_perturbations(x)
    #                     yp = sess.run(y_pred, feed_dict={inputs: Xp, graph_kernel: total_LK, tf_tmp_adj: GCN_total_adj, tf_keep_prob: 1.0})  # 扰动后的数据放入模型打标签
    #                     if model_name == 'stgcn':
    #                         yp = np.squeeze(yp)
    #                         yp = np.expand_dims(yp, 0)
    #                     goal = calculate_goal(yp, pred)
    #                     # sum = 0
    #                     # for i in chosen_nodes:
    #                     #     sum = sum + yp[0, i] - pred[0, i]
    #                     # return (goal * node_num - sum) / (node_num - len(chosen_nodes))
    #                     return goal
    #
    #                 xh = []
    #                 xf = copy.deepcopy(X)
    #                 xc = xf[0, :, i]
    #                 xh.append(xc)
    #                 for j in range(t):
    #                     xh.append(xf[0, :, chosen_nodes[j]])
    #                 xh = np.array(xh)
    #                 xv = xh.flatten()
    #                 xl = l * xv
    #                 xi = m * xv
    #                 xu = u * xv
    #
    #
    #                 params, minimum, progress = SPSA(evaluation, xi, iter, report=False, theta_min=xl, theta_max=xu,
    #                                                   return_progress=10)
    #
    #                 Xp = add_perturbations(params)
    #                 yp = sess.run(y_pred, feed_dict={inputs: Xp, graph_kernel: total_LK, tf_tmp_adj: GCN_total_adj, tf_keep_prob: 1.0})  # 扰动后的数据放入模型打标签
    #                 if model_name == 'stgcn':
    #                     yp = np.squeeze(yp)
    #                     yp = np.expand_dims(yp, 0)
    #                 # w[i] = 0
    #                 weighted_score = np.multiply(w, yp[0, :] - pred[0, :])
    #                 # w[i] = 1
    #                 score_matrix[i, :] = weighted_score
    #                 # print('iter_node%r:%r' % (t, i))
    #
    #         # mean_score = np.mean(score_matrix)
    #         # con = min(mean_score, 0)
    #         # score_matrix[np.where(score_matrix > con)] = 0
    #         # print(score_matrix)
    #         # np.save('score_matrix%r' % t, score_matrix)
    #         new_score_vector = np.sum(score_matrix, axis=1)  # total attack influence
    #         # print(new_score_vector)
    #         if t == 0:
    #             final_score_vector = new_score_vector / b  # marginal value of each node(if added to attack set), less than 0
    #         else:
    #             final_score_vector = (new_score_vector - total_score_matrix[t - 1, chosen_nodes[len(chosen_nodes) - 1]]) / b
    #             # marginal value(the minimum is the best)
    #             # total_score_matrix[...] denotes the influence on the whole system at iteration t-1.
    #
    #         total_score_matrix[t, :] = new_score_vector
    #         # np.save('total_score_matrix%r' % t, total_score_matrix)
    #         # print('total_score_matrix%r:' % t, total_score_matrix)
    #         sort_sequence = final_score_vector.argsort()  # sequence produced by gcg
    #         m = 0
    #         while calculate_total_budget_of_attack_set(b, add_node([sort_sequence[m]], chosen_nodes)) > B \
    #                 or sort_sequence[m] in chosen_nodes:
    #             m = m + 1
    #             if m == node_num:
    #                 k = 1
    #                 break
    #         else:
    #             if m == node_num:
    #                 break
    #             chosen_nodes.append(sort_sequence[m])
    #             # print('The %rth attack node is：' % t, sort_sequence[m])
    #             t = t + 1
    #
    #         if k == 1:
    #             break
    #
    #     # print('attack_set:', np.array(attack_set))
    #     return np.array(chosen_nodes)

    def find_attack_set_by_spsa(l, m, u, iter, b, B, w):
        "l: lower bound, -epsilon in paper"
        "m: initial value"
        "u: upper bound, +epsilon in paper"
        "iter: iteration for approximating influence of each node"
        "return vector: attack set"

        chosen_nodes = []
        total_score_matrix = np.zeros([node_num, node_num])
        t = len(chosen_nodes)
        k = 0
        while calculate_total_budget_of_attack_set(b, chosen_nodes) < B:
            score_matrix = np.zeros((node_num, node_num))
            for i in range(node_num):
                if i in chosen_nodes:
                    continue
                else:

                    def add_perturbations(x):
                        Xp = copy.deepcopy(X)
                        Xp[0, :, i] = Xp[0, :, i] + x[0: seq_len]
                        for j in range(t):
                            Xp[0, :, chosen_nodes[j]] = Xp[0, :, chosen_nodes[j]] + x[(j + 1) * seq_len: (j + 2) * seq_len]
                        return Xp

                    def evaluation(x):
                        Xp = add_perturbations(x)
                        yp = sess.run(y_pred, feed_dict={inputs: Xp, graph_kernel: total_LK, tf_tmp_adj: GCN_total_adj,
                                                         tf_keep_prob: 1.0})
                        if model_name == 'stgcn':
                            yp = np.squeeze(yp)
                            yp = np.expand_dims(yp, 0)
                        goal = calculate_goal(yp, pred)
                        return goal

                    xh = []
                    xf = copy.deepcopy(X)
                    xc = xf[0, :, i]
                    xh.append(xc)
                    for j in range(t):
                        xh.append(xf[0, :, chosen_nodes[j]])
                    xh = np.array(xh)
                    xv = xh.flatten()
                    xl = l * xv
                    xi = m * xv
                    xu = u * xv

                    params, minimum, progress = SPSA(evaluation, xi, iter, report=False, theta_min=xl, theta_max=xu,
                                                     return_progress=10)

                    Xp = add_perturbations(params)
                    yp = sess.run(y_pred, feed_dict={inputs: Xp, graph_kernel: total_LK, tf_tmp_adj: GCN_total_adj,
                                                     tf_keep_prob: 1.0})
                    if model_name == 'stgcn':
                        yp = np.squeeze(yp)
                        yp = np.expand_dims(yp, 0)
                    weighted_score = np.multiply(w, yp[0, :] - pred[0, :])
                    score_matrix[i, :] = weighted_score
                    # print('iter_node%r:%r' % (t, i))

            new_score_vector = np.sum(score_matrix, axis=1)  # total attack influence
            # print(new_score_vector)
            if t == 0:
                final_score_vector = new_score_vector  # marginal value of each node(if added to attack set), less than 0
            else:
                final_score_vector = (new_score_vector - total_score_matrix[t - 1, chosen_nodes[len(chosen_nodes)-1]]) / b
                # marginal value(the minimum is the best)
                # total_score_matrix[...] denotes the influence on the whole system at iteration t-1.

            total_score_matrix[t, :] = new_score_vector
            # np.save('total_score_matrix%r' % t, total_score_matrix)
            # print('total_score_matrix%r:' % t, total_score_matrix)
            sort_sequence = final_score_vector.argsort()  # sequence produced by spsa algorithm
            m = 0
            while calculate_total_budget_of_attack_set(b, add_node([sort_sequence[m]], chosen_nodes)) > B \
                    or sort_sequence[m] in chosen_nodes:
                m = m + 1
                if m == node_num:
                    k = 1
                    break
            else:
                if m == node_num:
                    break
                chosen_nodes.append(sort_sequence[m])
                # print('The %rth attack node is：' % t, sort_sequence[m])
                t = t + 1

            if k == 1:
                break

        # print('attack_set：', np.array(attack_set))
        return np.array(chosen_nodes)

    def find_attack_set_by_Kg_spsa(l, m, u, iter, b, B, w):
        "l: lower bound, -epsilon in paper"
        "m: initial value"
        "u: upper bound, +epsilon in paper"
        "iter: iteration for approximating influence of each node"
        "return vector: attack set"

        chosen_nodes = []
        score_vector = np.zeros(node_num)
        # t = len(chosen_nodes)
        for i in range(node_num):

            #  add perturbations on original input
            def add_perturbations(x):
                Xp = copy.deepcopy(X)
                Xp[0, :, i] = Xp[0, :, i] + x[0: seq_len]
                return Xp

            # define optimization goal
            def evaluation(x):
                Xp = add_perturbations(x)
                yp = sess.run(y_pred, feed_dict={inputs: Xp, graph_kernel: total_LK, tf_tmp_adj: GCN_total_adj,
                                                 tf_keep_prob: 1.0})
                if model_name == 'stgcn':
                    yp = np.squeeze(yp)
                    yp = np.expand_dims(yp, 0)
                goal = calculate_goal(yp, pred)
                return goal

            # define the limits of perturbation
            xh = []
            xf = copy.deepcopy(X)
            xc = xf[0, :, i]
            xh.append(xc)
            xh = np.array(xh)
            xv = xh.flatten()
            xl = l * xv
            xi = m * xv
            xu = u * xv

            params, minimum, progress = SPSA(evaluation, xi, iter, report=False, theta_min=xl, theta_max=xu,
                                             return_progress=10)

            Xp = add_perturbations(params)
            yp = sess.run(y_pred, feed_dict={inputs: Xp, graph_kernel: total_LK, tf_tmp_adj: GCN_total_adj,
                                             tf_keep_prob: 1.0})
            if model_name == 'stgcn':
                yp = np.squeeze(yp)
                yp = np.expand_dims(yp, 0)
            weighted_score = np.multiply(w, yp[0, :] - pred[0, :])
            score_vector[i] = np.sum(weighted_score) / b[i]
            print('node%r:' % i)

        sort_sequence = score_vector.argsort()  # sequence produced by Kg_spsa algorithm
        # print(sort_sequence)
        k = 0
        m = 0
        while calculate_total_budget_of_attack_set(b, chosen_nodes) < B:
            while calculate_total_budget_of_attack_set(b, add_node([sort_sequence[m]], chosen_nodes)) > B \
                    or sort_sequence[m] in chosen_nodes:
                m = m + 1
                if m >= node_num:
                    k = 1
                    break
            else:
                if m >= node_num:
                    break
                elif sort_sequence[m] == 26:  # an isolated node in dataset, could be ignored
                    m = m + 1
                    continue
                chosen_nodes.append(sort_sequence[m])
                m = m + 1
                print('The %rth attack node is：' % m, sort_sequence[m])

            if k == 1:
                break

        # print('attack set：', np.array(attack_set))
        return np.array(chosen_nodes)

    ############# diffusion attack on fixed attack_set
    def diffusion_attack(attack_set, l, m, u, iter):
        "parameter l: lower bound, -epsilon in paper"
        "parameter m: initial value"
        "parameter u: upper bound, +epsilon in paper"
        "parameter iter: iteration for determine total influence of attack set"
        "return vector result1: degradation ratio of speed of each node, the average is AAIR in paper"
        "return vector result2: degradation(km/h) of speed of each node, the average is AAI  in paper"

        #  add perturbations x on original input
        def add_perturbations(x):
            Xp = copy.deepcopy(X)
            for i in range(attack_set.shape[0]):
                Xp[0, :, attack_set[i]] = Xp[0, :, attack_set[i]] + x[i * seq_len: (i + 1) * seq_len]  # 增加扰动
            return Xp  # Xp is an adversarial sample

        # define the limits of perturbation
        xh = []
        for i in range(attack_set.shape[0]):
            Xf = copy.deepcopy(X)
            xc = Xf[0, :, attack_set[i]]
            xh.append(xc)
        xh = np.array(xh)
        xv = xh.flatten()
        xl = l * xv
        xi = m * xv
        xu = u * xv

        def evaluation(x):  # evaluate the attack influence or optimization goal,
            # here we try to minimize it (degrade the speed in order to create traffic congestion)
            Xp = add_perturbations(x)
            # yp is a prediction matrix(the same size as the original output "pred") when the input is an adversarial sample Xp
            # in other word, yp denotes pertured speed of each node
            yp = sess.run(y_pred, feed_dict={inputs: Xp, graph_kernel: total_LK, tf_tmp_adj: GCN_total_adj, tf_keep_prob: 1.0})  # 扰动后的数据放入模型打标签
            if model_name == 'stgcn':
                yp = np.squeeze(yp)
                yp = np.expand_dims(yp, 0)
            goal = calculate_goal(yp, pred)
            return goal

        params, minimum, progress = SPSA(evaluation, xi, iter, report=100, theta_min=xl, theta_max=xu,
                                         return_progress=10)

        # if necessary, draw the optimization goal as iteration increase in spsa algorithm

        # print(f"The parameters that minimise the function are {params}\nThe minimum value of f is: {minimum}")
        # plot_progress(progress, title="SPSA", xlabel=r"Iteration", ylabel=r"evaluation")
        # plot_progress(progress, title="SPSA", xlabel=r"Iteration", ylabel=r"evaluation",
        #               moving_average=10)

        Xp = add_perturbations(params)  # get an optimized adversarial sample
        yp = sess.run(y_pred, feed_dict={inputs: Xp, graph_kernel: total_LK,
                                         tf_tmp_adj: GCN_total_adj, tf_keep_prob: 1.0})
        if model_name == 'stgcn':
            yp = np.squeeze(yp)
            yp = np.expand_dims(yp, 0)

        difference = np.zeros(yp.shape[1])
        speed = np.zeros(yp.shape[1])

        for i in range(node_num):
            difference[i] = yp[0, i] - pred[0, i]  # The difference of speed of node i between perturbed prediction and original prediction
            speed[i] = pred[0, i]

        difference = np.array(difference)
        speed = np.array(speed)

        result1 = difference / speed  # This vector stores degradation ratio of speed of each node under attack
        result2 = difference * max_value  # This vector stores degradation of speed of each node under attack (km/h)
        # attack_speed = (params + xv) * max_value
        np.set_printoptions(precision=2)
        return result1, result2

    for i in range(len(B)):
        print('time_step:', time_step, 'attack_budget:', B[i])
        if method == 'random':
            attack_set = find_attack_set_by_random(b, B[i])
        if method == 'degree':
            attack_set = find_attack_set_by_degree(A2, b, B[i])
        elif method == 'spsa':
            attack_set = find_attack_set_by_spsa(-1, 0, 0.5, 10, b, B[i], w)
        # elif method == 'spsa_step':
        #     attack_set = find_attack_set_by_spsa_step(-1, 0, 0.5, 10, b, B[i], w)
        elif method == 'Kg_spsa':
            attack_set = find_attack_set_by_Kg_spsa(-1, 0, 0.5, 100, b, B[i], w)
        elif method == 'kmedoids':
            attack_set = find_attack_set_by_kmedoids(locations, b, B[i])
        elif method == 'pagerank':
            attack_set = find_attack_set_by_pagerank(A2, b, B[i])
        elif method == 'Kg_pagerank':
            attack_set = find_attack_set_by_Kg_pagerank(A2, b, B[i])
        elif method == 'betweenness':
            attack_set = find_attack_set_by_betweenness(A2, b, B[i])
        elif method == 'Kg_betweenness':
            attack_set = find_attack_set_by_Kg_betweenness(A2, b, B[i])

        attack_set = np.array(attack_set)
        result1, result2 = diffusion_attack(attack_set, -1, 0, 0.5, 30000)

        attack_influence1[time_step, i] = np.mean(result1)
        attack_influence2[time_step, i] = np.mean(result2)

        mytxt = open('%s_%s_%s_pre%r_epoch%r_drop%s_%r.txt' % (method, data_name, model_name, pre_len,
                                                            training_epoch, dropmode, k), mode='a', encoding='utf-8')
        print('average rate:', np.mean(result1), 'average speed change:', np.mean(result2),
              'attack set', attack_set, 'B:', B[i], 'time step:', time_sample[time_step],
              file=mytxt)
        mytxt.close()


####### record of average of attack result in different time sample
mytxt = open('%s_%s_%s_pre%r_epoch%r_drop%s_%r.txt' % (method, data_name, model_name, pre_len,
                                                       training_epoch, dropmode, k), mode='a', encoding='utf-8')
print('\n', file=mytxt)
for i in range(len(B)):
    print('final_average rate:', np.mean(attack_influence1[:, i]), 'final_average speed change:', np.mean(attack_influence2[:, i]),
          'B:', B[i], file=mytxt)
mytxt.close()


############ visualisation
axis_x = []
axis_y = []
for i in range(len(attack_set)):
    axis_x.append(locations[attack_set[i], 0])
    axis_y.append(locations[attack_set[i], 1])

axis_x = np.array(axis_x)
axis_y = np.array(axis_y)

# fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(16, 9), dpi=300)
# plt.style.use('seaborn-poster')
plt.scatter(axis_x, axis_y, c='black', s=400, alpha=0.4, marker='^')
for i in range(node_num):
    for j in range(node_num):
        if j >= i and A2[i, j] != 0:
            plt.plot([locations[i, 0], locations[j, 0]], [locations[i, 1], locations[j, 1]],
                     color='#06470c', linewidth=0.2)

result1 = (result1-abs(result1))/2.0  # degradation ratio of speed of each node, less than 0
result1[np.where(result1 <= -1)] = -1  # for convenience, if degradation of speed less than -100%, we set it -100%

if data_name == 'los':
    locations = np.delete(locations, 26, axis=0)  # delete an isolated node in dataset
    result1 = np.delete(result1, 26, axis=0)
print(result1)
sc = plt.scatter(locations[:, 0], locations[:, 1], c=-result1, s=40, cmap='RdYlGn_r', vmin=0, vmax=0.60)

cbar = plt.colorbar(sc)
ticklabels = ['0.0', '0.2', '0.4', '0.6']
cbar.set_ticks(np.linspace(0, 0.6, len(ticklabels)))
cbar.set_ticklabels(ticklabels)

plt.xlabel('Longitude', fontsize=15)
plt.ylabel('Latitude', fontsize=15)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.show()