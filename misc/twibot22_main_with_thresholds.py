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
from util import multirank, multirank2, multirank_cuda, multirank_cuda2
import gc
import matplotlib.pyplot as plt

# For the Multi-Relational Graph Construction:
ff_similarity_threshold = 0.1
posting_type_threshold = 0.1
posting_inf_threshold = 0.1

# For User Community Division:
# The ratio of three edge weights for user community division is implicitly present in #Combine Graphs section
max_scale_ratio = 0.15

# For Community Binary Classification:
# Stabilize threshold of the distribution tensor is in multirank functions in util.py
stabalize_threshold = 0.004 #For use as parameter if calling multirank2() instrad of multirank()
community_entropy_weight = 0.4
community_influence_weight = 0.6
evaluation_index_treshold = 1

#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f'device is {device}')

def process(dataset_path, bot_filename):
    path = os.path.join(dataset_path, bot_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist.")
    
    # Load the prep`rocessed bot data
    bots = pd.read_csv(path)

    # Convert labels to 1 for bot
    #bots['label'] = bots['label'].apply(lambda x: 1 if x == 'bot' else 0)
    
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
    #humans['label'] = humans['label'].apply(lambda x: 0 if x == 'human' else 1)
    
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
    label_block = np.array(bot_labels + human_labels)  # Ensure label_block is a numpy array
    ff_block = np.array(bot_ff + human_ff)
    types_block = np.array(bot_types + human_types)
    infs_block = np.array(bot_infs + human_infs)

    print(f'Number of bots: {num_bots}')
    print(f'Number of humans: {num_humans}')
    #print(f'Labels: {label_block}')

    if len(set(label_block)) < 2:
        raise ValueError("The dataset does not contain both classes.")

    # Logging start time for graph creation
    start_graph_creation = time.time()

    graph_ff = lil_matrix((num, num))
    graph_types = lil_matrix((num, num))
    graph_infs = lil_matrix((num, num))

    # Create the graphs using sparse matrices
    for i in range(num):
        start_row_processing = time.time()
        #print(f"Processing row {i}/{num}")
        for j in range(i + 1, num):
            # Follow-to-follower ratio
            diff_ff = abs(ff_block[i] - ff_block[j])
            max_ff = max(ff_block[i], ff_block[j])
            min_ff = min(ff_block[i], ff_block[j])
            if max_ff != 0:  # Avoid division by zero
                deviation_ratio_ff = diff_ff / max_ff
                if deviation_ratio_ff <= 1:  # Ensure d_ij <= 1
                    weight_ff = 1 - deviation_ratio_ff
                    #if deviation_ratio_ff < ff_similarity_threshold and min_ff >= 3:  # Only add edge if below threshold
                    if deviation_ratio_ff < ff_similarity_threshold:  # Only add edge if below threshold
                        graph_ff[i, j] = graph_ff[j, i] = weight_ff
            
            # Posting type distribution
            diff_types = np.sum(abs(types_block[i] - types_block[j]))
            if diff_types <= 1:  # Ensure d_ij <= 1 as per the paper
                weight_types = 1 - diff_types
                if diff_types < posting_type_threshold:  # Only add edge if below threshold
                    graph_types[i, j] = graph_types[j, i] = weight_types
            
            # Posting influence
            diff_infs = abs(infs_block[i] - infs_block[j])
            max_infs = max(infs_block[i], infs_block[j])
            if max_infs != 0:  # Avoid division by zero
                deviation_ratio_infs = diff_infs / max_infs
                if deviation_ratio_infs <= 1:  # Ensure d_ij <= 1
                    weight_infs = 1 - deviation_ratio_infs
                    #if deviation_ratio_infs < (max_infs * posting_inf_threshold) and min_infs >= 3:  # Only add edge if below threshold
                    if deviation_ratio_infs < posting_inf_threshold:  # Only add edge if below threshold
                        graph_infs[i, j] = graph_infs[j, i] = weight_infs

        end_row_processing = time.time()
        #print(f"Row processing took {end_row_processing - start_row_processing} seconds")

    graph_ff = graph_ff.tocsr()
    graph_types = graph_types.tocsr()
    graph_infs = graph_infs.tocsr()

    # Logging end time for graph creation
    end_graph_creation = time.time()
    print(f"Graph creation took {end_graph_creation - start_graph_creation} seconds")

    max_ff = np.maximum(ff_block.reshape(-1, 1), ff_block.reshape(1, -1))
    max_infs = np.maximum(infs_block.reshape(-1, 1), infs_block.reshape(1, -1))

    # Replace zeros in max_infs with a small value to avoid division by zero
    max_ff[max_ff == 0] = 1e-10
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
    print('Garbage collection done')

    print("Calling multirank_cuda()")
    x, y = multirank2(multi_graph, stabalize_threshold)
    print("Finished multirank_cuda()")

    edges = np.array(adj_matrix.nonzero())  # [2, E]
    ew = adj_matrix[edges[0, :], edges[1, :]].flatten()  # convert to 1D array
    ew, edges = torch.tensor(ew, device=device), torch.tensor(edges, device=device).t()
    edges = edges.to(torch.int64)  # Ensure edges tensor is int64
    dist = scatter_sum(ew, edges[:, 1]) + scatter_sum(ew, edges[:, 0])
    dist = dist / (2 * ew.sum())
    print('construct encoding tree...')
    g = GraphSparse(edges, ew, dist)
    
     # Release memory from CPU variables
    del edges, ew, dist
    gc.collect()
    torch.cuda.empty_cache()
    print('Garbage collection done')

    optim = OperatorPropagation(Partitioning(g, None))
    optim.perform(p=max_scale_ratio)
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
        n_beta = (n_x / n_nodes) / (1 / num) * community_influence_weight + comm_SE / (sum(module_se).cpu().numpy() / total_comm.cpu().numpy()) * community_entropy_weight
        if n_beta >= evaluation_index_treshold:
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

    # Save the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

if __name__ == "__main__":
    start_time = time.time()
    print(f'Start time is {time.strftime("%x %X")}')
    #print("device is", 'cuda' if torch.cuda.is_available() else 'cpu')
    test('preprocess/sample1/sampled_combined_preprocessed_twibot22_bots.csv', 'preprocess/sample1/sampled_combined_preprocessed_twibot22_humans.csv', dataset_path='')
    end_time = time.time()
    print(f'running time: {(end_time - start_time) / 1} s')
    print(f'End time is {time.strftime("%x %X")}')
