from util import multirank, process_tweet,multirank_cuda
import csv
import os
from sklearn import metrics
import numpy as np
import pickle
from silearn.graph import GraphSparse
from silearn.optimizer.enc.partitioning.propagation import OperatorPropagation
from silearn.model.encoding_tree import Partitioning, EncodingTree
from torch_scatter import scatter_sum
import torch
import time
import matplotlib.pyplot as plt
import math
import networkx as nx


import pandas as pd
import os

def process(dataset_path, bot_filename):
    path = os.path.join(dataset_path, bot_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist.")
    
    # Load the preprocessed bot data
    bots = pd.read_csv(path)

    # Convert labels to 1 for bot
    bots['label'] = bots['label'].apply(lambda x: 1 if x == 'bot' else 0)
    
    num_bots = bots.shape[0]
    bot_ids = bots['userid'].tolist()
    bot_labels = bots['label'].tolist()
    bot_ff = bots['ff'].tolist()
    bot_types = bots[['type1', 'type2', 'type3']].values.tolist()
    bot_infs = bots['inf'].tolist()
    
    return num_bots, bot_ids, bot_labels, bot_ff, bot_types, bot_infs

def process_human(dataset_path, human_filename):
    path = os.path.join(dataset_path, human_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist.")
    
    # Load the preprocessed human data
    humans = pd.read_csv(path)

    # Convert labels to 0 for human
    humans['label'] = humans['label'].apply(lambda x: 1 if x == 'bot' else 0)
    
    num_humans = humans.shape[0]
    human_ids = humans['userid'].tolist()
    human_labels = humans['label'].tolist()
    human_ff = humans['ff'].tolist()
    human_types = humans[['type1', 'type2', 'type3']].values.tolist()
    human_infs = humans['inf'].tolist()
    
    return num_humans, human_ids, human_labels, human_ff, human_types, human_infs

def test(bot_filename, human_filename, dataset_path=''):
    num_bots, bot_ids, bot_labels, bot_ff, bot_types, bot_infs = process(dataset_path, bot_filename)
    num_humans, human_ids, human_labels, human_ff, human_types, human_infs = process_human(dataset_path, human_filename)

    num = num_bots + num_humans
    id_block = bot_ids + human_ids
    label_block = bot_labels + human_labels
    ff_block = bot_ff + human_ff
    types_block = bot_types + human_types
    infs_block = bot_infs + human_infs

    #print('num:{}, bot:{}'.format(num, np.sum(label_block)))

    graph = np.zeros((3, num, num))
    edge = np.zeros(3)

    diff_ff = abs(np.array(ff_block).reshape(-1, 1) - np.array(ff_block).reshape(1, -1))
    max_ff = np.maximum(np.array(ff_block).reshape(-1, 1), np.array(ff_block).reshape(1, -1))
    min_ff = np.minimum(np.array(ff_block).reshape(-1, 1), np.array(ff_block).reshape(1, -1))
    mask_ff = (diff_ff < (max_ff * 0.1)) & (min_ff >= 3)
    graph[0][mask_ff] = 1
    graph[0] -= np.diag(np.diag(graph[0]))

    diff_types = abs(np.array(types_block)[:, 0].reshape(-1, 1) - np.array(types_block)[:, 0].reshape(1, -1)) + \
                 abs(np.array(types_block)[:, 1].reshape(-1, 1) - np.array(types_block)[:, 1].reshape(1, -1)) + \
                 abs(np.array(types_block)[:, 2].reshape(-1, 1) - np.array(types_block)[:, 2].reshape(1, -1))
    mask_types = diff_types < 0.1
    graph[1][mask_types] = 1
    graph[1] -= np.diag(np.diag(graph[1]))

    diff_infs = abs(np.array(infs_block).reshape(-1, 1) - np.array(infs_block).reshape(1, -1))
    max_infs = np.maximum(np.array(infs_block).reshape(-1, 1), np.array(infs_block).reshape(1, -1))
    min_infs = np.minimum(np.array(infs_block).reshape(-1, 1), np.array(infs_block).reshape(1, -1))
    mask_infs = (diff_infs < (max_infs * 0.1))
    graph[2][mask_infs] = 1
    graph[2] -= np.diag(np.diag(graph[2]))

    max_ff[~mask_ff] = 1
    max_infs[~mask_infs] = 1
    adj_matrix = np.zeros((num, num))
    adj_matrix += (1 - (diff_ff / max_ff)) * mask_ff + (1 - diff_types) * mask_types + (1 - (diff_infs / max_infs)) * mask_infs
    adj_matrix -= np.diag(np.diag(adj_matrix))

    index = np.where(np.sum(adj_matrix, axis=1) == 0)
    for i in index:
        adj_matrix[i, i] = 0.01

    x, y = multirank_cuda(graph)
    edges = np.array(adj_matrix.nonzero())  # [2, E]
    ew = adj_matrix[edges[0, :], edges[1, :]]
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ew, edges = torch.tensor(ew, device=device), torch.tensor(edges, device=device).t()
    dist = scatter_sum(ew, edges[:, 1]) + scatter_sum(ew, edges[:, 0])
    dist = dist / (2 * ew.sum())
    print('construct encoding tree...')
    g = GraphSparse(edges, ew, dist)
    optim = OperatorPropagation(Partitioning(g, None))
    optim.perform(p=0.15)
    print('construct encoding tree done')
    division = optim.enc.node_id
    SE2d = optim.enc.structural_entropy(reduction='sum', norm=True)
    module_se = optim.enc.structural_entropy(reduction='module', norm=True)
    total_comm = torch.max(division) + 1
    print('total_comm:{}'.format(total_comm))
    comms = {}
    for i in range(total_comm):
        idx = division == i
        if idx.any():
            comms[i] = idx.nonzero().squeeze(1)

    bot_rate = []
    pre_bot = np.zeros(num)
    value = np.zeros(num)
    num_bot_comm = 0
    bot_list = []
    for i in comms.keys():
        comm = comms[i]
        n_bots = 0
        n_nodes = 0
        n_x = 0
        for node in comm:
            n_bots += label_block[node]
            n_nodes += 1
            n_x += x[node]
        comm_SE = module_se[i]
        n_beta = (n_x / n_nodes) / (1 / num) * 0.6 + comm_SE / (sum(module_se) / total_comm) * 0.4
        if n_beta >= 1:
            num_bot_comm += 1
            for node in comm:
                pre_bot[node] = 1
                value[node] = n_beta
        else:
            for node in comm:
                bot_list.append(node)
                value[node] = n_beta
        bot_rate.append([n_bots / n_nodes, n_bots, n_nodes, n_x / n_nodes, comm_SE, n_beta])

    acc = metrics.accuracy_score(label_block, pre_bot)
    precision = metrics.precision_score(label_block, pre_bot)
    recall = metrics.recall_score(label_block, pre_bot)
    f1 = metrics.f1_score(label_block, pre_bot)
    fpr, tpr, thresholds = metrics.roc_curve(label_block, value)
    auc_score = metrics.roc_auc_score(label_block, value)
    print('acc:{}'.format(acc))
    print('Precision:{}'.format(precision))
    print('Recall:{}'.format(recall))
    print('F1:{}'.format(f1))
    print('AUC:{}'.format(auc_score))

if __name__ == "__main__":
    start_time=time.time()
    test('output/combined_preprocessed_twibot22_bots.csv', 'output/combined_preprocessed_twibot22_humans.csv', dataset_path='')
    end_time=time.time()
    print('running time:{}s'.format((end_time-start_time)/1))