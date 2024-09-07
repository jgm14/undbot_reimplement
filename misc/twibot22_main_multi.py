import pandas as pd
import numpy as np
from sklearn import metrics
from scipy.sparse import csr_matrix, lil_matrix
from silearn.graph import GraphSparse
from silearn.optimizer.enc.partitioning.propagation import OperatorPropagation
from silearn.model.encoding_tree import Partitioning
from torch_scatter import scatter_sum
import torch
import time
import os
from util import multirank, multirank_cuda, multirank_cuda2
import gc
from joblib import Parallel, delayed
import multiprocessing

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device is {device}')

def process(dataset_path, bot_filename):
    path = os.path.join(dataset_path, bot_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist.")
    
    # Load the preprocessed bot data
    bots = pd.read_csv(path)
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
    num_humans = humans.shape[0]
    human_ids = humans['userid'].tolist()
    human_labels = humans['label'].tolist()
    human_ff = humans['ff'].tolist()
    human_types = humans[['type1', 'type2', 'type3']].values.tolist()
    human_infs = humans['inf'].tolist()
    
    return num_humans, human_ids, human_labels, human_ff, human_types, human_infs

def calculate_weights_chunk(start_idx, end_idx, num, ff_block, types_block, infs_block):
    print(f"Processing chunk from {start_idx} to {end_idx} with PID {os.getpid()}")
    
    graph_ff = lil_matrix((num, num))
    graph_types = lil_matrix((num, num))
    graph_infs = lil_matrix((num, num))
    
    for i in range(start_idx, end_idx):
        for j in range(i + 1, num):
            # Follow-to-follower ratio
            diff_ff = abs(ff_block[i] - ff_block[j])
            max_ff = max(ff_block[i], ff_block[j])
            min_ff = min(ff_block[i], ff_block[j])
            if max_ff != 0:  # Avoid division by zero
                deviation_ratio_ff = diff_ff / max_ff
                if deviation_ratio_ff <= 1:  # Ensure d_ij <= 1
                    weight_ff = 1 - deviation_ratio_ff
                    if deviation_ratio_ff < 0.1 and min_ff >= 3:  # Only add edge if below threshold
                        graph_ff[i, j] = graph_ff[j, i] = weight_ff
            
            # Posting type distribution
            diff_types = np.sum(abs(types_block[i] - types_block[j]))
            if diff_types <= 1:  # Ensure d_ij <= 1 as per the paper
                weight_types = 1 - diff_types
                if diff_types < 0.1:  # Only add edge if below threshold
                    graph_types[i, j] = graph_types[j, i] = weight_types
            
            # Posting influence
            diff_infs = abs(infs_block[i] - infs_block[j])
            max_infs = max(infs_block[i], infs_block[j])
            if max_infs != 0:  # Avoid division by zero
                deviation_ratio_infs = diff_infs / max_infs
                if deviation_ratio_infs <= 1:  # Ensure d_ij <= 1
                    weight_infs = 1 - deviation_ratio_infs
                    if deviation_ratio_infs < (max_infs * 0.1):  # Only add edge if below threshold
                        graph_infs[i, j] = graph_infs[j, i] = weight_infs
    
    return graph_ff, graph_types, graph_infs

def test(bot_filename, human_filename, dataset_path=''):
    num_bots, bot_ids, bot_labels, bot_ff, bot_types, bot_infs = process(dataset_path, bot_filename)
    num_humans, human_ids, human_labels, human_ff, human_types, human_infs = process_human(dataset_path, human_filename)

    num = num_bots + num_humans
    id_block = bot_ids + human_ids
    label_block = np.array(bot_labels + human_labels)  # Ensure label_block is a numpy array
    ff_block = np.array(bot_ff + human_ff)
    types_block = np.array(bot_types + human_types)
    infs_block = np.array(bot_infs + human_infs)

    print(f'Number of bots: {num_bots}')
    print(f'Number of humans: {num_humans}')
    if len(set(label_block)) < 2:
        raise ValueError("The dataset does not contain both classes.")

    # Logging start time for graph creation
    start_graph_creation = time.time()

    chunk_size = 1000  # Adjust chunk size based on your memory capacity and data size
    results = Parallel(n_jobs=2)(delayed(calculate_weights_chunk)(
        i, min(i + chunk_size, num), num, ff_block, types_block, infs_block
    ) for i in range(0, num, chunk_size))
    
    graph_ff = sum([res[0] for res in results], lil_matrix((num, num)))
    graph_types = sum([res[1] for res in results], lil_matrix((num, num)))
    graph_infs = sum([res[2] for res in results], lil_matrix((num, num)))

    graph_ff = graph_ff.tocsr()
    graph_types = graph_types.tocsr()
    graph_infs = graph_infs.tocsr()

    # Logging end time for graph creation
    end_graph_creation = time.time()
    print(f"Graph creation took {end_graph_creation - start_graph_creation} seconds")

    max_ff = np.maximum(ff_block.reshape(-1, 1), ff_block.reshape(1, -1))
    max_infs = np.maximum(infs_block.reshape(-1, 1), infs_block.reshape(1, -1))

    # Replace zeros in max_infs with a small value to avoid division by zero
    max_infs[max_infs == 0] = 1e-10

    # Combine graphs
    adj_matrix = (graph_ff.multiply(1 - (abs(ff_block.reshape(-1, 1) - ff_block.reshape(1, -1)) / max_ff)) +
                  graph_types.multiply(1 - abs(types_block.reshape(-1, 1, 3) - types_block.reshape(1, -1, 3)).sum(axis=2)) +
                  graph_infs.multiply(1 - (abs(infs_block.reshape(-1, 1) - infs_block.reshape(1, -1)) / max_infs))).toarray()
    np.fill_diagonal(adj_matrix, 0)

    index = np.where(adj_matrix.sum(axis=1) == 0)[0]
    for i in index:
        adj_matrix[i, i] = 0.01

    multi_graph = np.array([graph_ff.toarray(), graph_types.toarray(), graph_infs.toarray()])
    
    # Release memory from CPU variables
    del graph_ff, graph_types, graph_infs
    gc.collect()
    torch.cuda.empty_cache()

    print("Calling multirank_cuda()")
    x, y = multirank_cuda(multi_graph)
    print("Finished multirank_cuda()")

    edges = np.array(adj_matrix.nonzero())  # [2, E]
    ew = adj_matrix[edges[0, :], edges[1, :]].flatten()  # convert to 1D array
    ew, edges = torch.tensor(ew, device=device), torch.tensor(edges, device=device).t()
    edges = edges.to(torch.int64)  # Ensure edges tensor is int64
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
    print(f'total_comm after merging: {total_comm}')
    comms = {i: (division == i).nonzero().squeeze(1) for i in range(total_comm) if (division == i).any()}

    bot_rate = []
    pre_bot = np.zeros(num)
    value = np.zeros(num)
    num_bot_comm = 0
    for i in comms.keys():
        comm = comms[i].cpu().numpy()  # Ensure comm is a numpy array
        n_bots = label_block[comm].sum()
        n_nodes = len(comm)
        n_x = x[comm].sum()
        comm_SE = module_se[i].cpu().numpy()
        n_beta = (n_x / n_nodes) / (1 / num) * 0.6 + comm_SE / (sum(module_se).cpu().numpy() / total_comm.cpu().numpy()) * 0.4
        if n_beta >= 1:
            num_bot_comm += 1
            pre_bot[comm] = 1
            value[comm] = n_beta
        else:
            value[comm] = n_beta
        bot_rate.append([n_bots / n_nodes, n_bots, n_nodes, n_x / n_nodes, comm_SE, n_beta])

    # Ensure no NaN values in value
    value = np.nan_to_num(value)

    acc = metrics.accuracy_score(label_block, pre_bot)
    precision = metrics.precision_score(label_block, pre_bot, zero_division=0)
    recall = metrics.recall_score(label_block, pre_bot, zero_division=0)
    f1 = metrics.f1_score(label_block, pre_bot, zero_division=0)
    fpr, tpr, thresholds = metrics.roc_curve(label_block, value)
    auc_score = metrics.roc_auc_score(label_block, value)
    print(f'acc: {acc}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')
    print(f'AUC: {auc_score}')

if __name__ == "__main__":
    start_time = time.time()
    print(f'Start time is {time.strftime("%x %X")}')
    test('output2/combined_preprocessed_twibot22_bots.csv', 'output2/combined_preprocessed_twibot22_humans.csv', dataset_path='')
    end_time = time.time()
    print(f'running time: {(end_time - start_time) / 1} s')
    print(f'End time is {time.strftime("%x %X")}')
