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
from util import multirank, multirank2, multirank_cuda
import gc
import matplotlib.pyplot as plt

#Define Device
#device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device('cpu')
print(f'device is {device}')

#Define Thresholds
# For the Multi-Relational Graph Construction:
ff_similarity_threshold = 0.1
posting_type_threshold = 0.1
posting_inf_threshold = 0.1

# For User Community Division:
max_scale_ratio = 0.15

# For Community Binary Classification:
# Stabilize threshold of the distribution tensor is in multirank functions in util.py
stabalize_threshold = 0.004 #For use as parameter if calling multirank2() instrad of multirank()
community_entropy_weight = 0.4
community_influence_weight = 1 - community_entropy_weight
evaluation_index_treshold = 1

def process(dataset_path, bot_filename):
    #Initializes bot data as needed by test()
    path = os.path.join(dataset_path, bot_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist.")
    
    bots = pd.read_csv(path)
    num_bots = bots.shape[0]
    bot_ids = bots['userid'].tolist()
    bot_labels = bots['label'].tolist()
    bot_ff = bots['ff'].tolist()
    bot_types = bots[['type1', 'type2', 'type3']].values.tolist()
    bot_infs = bots['inf'].tolist()
    
    return num_bots, bot_ids, bot_labels, bot_ff, bot_types, bot_infs

def process_human(dataset_path, human_filename):
    #Initializes human data as needed by test()
    path = os.path.join(dataset_path, human_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist.")
    
    humans = pd.read_csv(path)
    num_humans = humans.shape[0]
    human_ids = humans['userid'].tolist()
    human_labels = humans['label'].tolist()
    human_ff = humans['ff'].tolist()
    human_types = humans[['type1', 'type2', 'type3']].values.tolist()
    human_infs = humans['inf'].tolist()
    
    return num_humans, human_ids, human_labels, human_ff, human_types, human_infs

def test(bot_filename, human_filename, dataset_path=''):
    #Initial setup and checks
    num_bots, bot_ids, bot_labels, bot_ff, bot_types, bot_infs = process(dataset_path, bot_filename)
    num_humans, human_ids, human_labels, human_ff, human_types, human_infs = process_human(dataset_path, human_filename)

    num = num_bots + num_humans
    id_block = bot_ids + human_ids
    label_block = np.array(bot_labels + human_labels)
    ff_block = np.array(bot_ff + human_ff)
    types_block = np.array(bot_types + human_types)
    infs_block = np.array(bot_infs + human_infs)

    print(f'Number of bots: {num_bots}')
    print(f'Number of humans: {num_humans}')

    if len(set(label_block)) < 2:
        raise ValueError("The dataset does not contain both classes.")

    start_graph_creation = time.time()

    # Initialize the graphs using sparse matrices
    graph_ff = lil_matrix((num, num))
    graph_types = lil_matrix((num, num))
    graph_infs = lil_matrix((num, num))

    # Compute the differences and maximums for follow-to-follower ratios
    diff_ff = abs(ff_block.reshape(-1, 1) - ff_block.reshape(1, -1))
    max_ff = np.maximum(ff_block.reshape(-1, 1), ff_block.reshape(1, -1))
    min_ff = np.minimum(ff_block.reshape(-1, 1), ff_block.reshape(1, -1))

    # Compute the FF mask for valid FF edges
    mask_ff = (diff_ff <= (1 * max_ff)) & (diff_ff < (ff_similarity_threshold * max_ff)) & (min_ff >= 3) # min_ff condition present
    #mask_ff = (diff_ff <= (1 * max_ff)) & (diff_ff < (ff_similarity_threshold * max_ff)) # min_ff condition not present

    # Apply the  FF mask to create the FF edges
    rows_ff, cols_ff = np.where(mask_ff)
    graph_ff[rows_ff, cols_ff] = 1 - (diff_ff[mask_ff] / max_ff[mask_ff])

    # Compute the differences for posting type distribution
    diff_types = (abs(types_block[:, 0].reshape(-1, 1) - types_block[:, 0].reshape(1, -1)) + 
                        abs(types_block[:, 1].reshape(-1, 1) - types_block[:, 1].reshape(1, -1)) + 
                        abs(types_block[:, 2].reshape(-1, 1) - types_block[:, 2].reshape(1, -1)))

    # Compute the posting type mask for valid posting type edges
    mask_types = (diff_types <= 1) & (diff_types < posting_type_threshold)

    # Apply the posting type mask to create the posting type edges
    rows, cols = np.where(mask_types)
    graph_types[rows, cols] = 1 - diff_types[mask_types]

    # Compute the differences and maximums for posting influence
    diff_infs = abs(infs_block.reshape(-1, 1) - infs_block.reshape(1, -1))
    max_infs = np.maximum(infs_block.reshape(-1, 1), infs_block.reshape(1, -1))
    min_infs = np.minimum(infs_block.reshape(-1, 1), infs_block.reshape(1, -1))

    # Compute the user influence mask for valid euser influence dges
    #mask_infs = (diff_infs <= (1 * max_infs)) & (diff_infs < (posting_inf_threshold * max_infs)) & (min_infs >= 3)
    mask_infs = (diff_infs <= (1 * max_infs)) & (diff_infs < (posting_inf_threshold * max_infs))

    # Apply the user inflience mask to create the user influence edges
    rows_infs, cols_infs = np.where(mask_infs)
    graph_infs[rows_infs, cols_infs] = 1 - (diff_infs[mask_infs] / max_infs[mask_infs])

    #Convert graphs
    graph_ff = graph_ff.tocsr()
    graph_types = graph_types.tocsr()
    graph_infs = graph_infs.tocsr()

    end_graph_creation = time.time()
    print(f"Graph creation took {end_graph_creation - start_graph_creation} seconds")

    # Combine graphs into multi graph by summing the pre-calculated weights
    adj_matrix = (graph_ff + graph_types + graph_infs).toarray()
    np.fill_diagonal(adj_matrix, 0)

    index = np.where(adj_matrix.sum(axis=1) == 0)[0]
    for i in index:
        adj_matrix[i, i] = 0.01

    multi_graph = np.array([graph_ff.toarray(), graph_types.toarray(), graph_infs.toarray()])
    
    # Release memory
    del graph_ff, graph_types, graph_infs
    gc.collect()
    torch.cuda.empty_cache()
    print('Garbage collection done')

    #Calculate ranks using MultiRank
    print("Calling multirank_cuda()")
    x, y = multirank2(multi_graph, stabalize_threshold)
    print("Finished multirank_cuda()")

    # Identify the non-zero elements in the adjacency matrix, representing the edges in the graph
    edges = np.array(adj_matrix.nonzero())  # Edges shape [2, E]
    # Extract the edge weights from the adjacency matrix
    ew = adj_matrix[edges[0, :], edges[1, :]].flatten()  # convert to 1D array
    # Convert edge weights and edges to torch tensors and ensure edges are in int64 format
    ew, edges = torch.tensor(ew, device=device), torch.tensor(edges, device=device).t()
    edges = edges.to(torch.int64)  # Ensure correct tensor format for graph processing
    # Compute the distribution tensor
    dist = scatter_sum(ew, edges[:, 1]) + scatter_sum(ew, edges[:, 0])
    dist = dist / (2 * ew.sum())
    print('construct encoding tree...')
    # Create a sparse graph representation required by the encoding tree construction
    g = GraphSparse(edges, ew, dist)
    
    #Release memory
    del edges, ew, dist
    gc.collect()
    torch.cuda.empty_cache()
    print('Garbage collection done')

    # Perform community division using the encoding tree
    optim = OperatorPropagation(Partitioning(g, None))
    optim.perform(p=max_scale_ratio)
    print('construct encoding tree done')
    # Calculate the structural entropy
    division = optim.enc.node_id
    SE2d = optim.enc.structural_entropy(reduction='sum', norm=True)
    module_se = optim.enc.structural_entropy(reduction='module', norm=True)
    # Determine the total number of communities after merging
    total_comm = torch.max(division) + 1
    print(f'total_comm after merging: {total_comm}')
    # Store the nodes belonging to each community in a dictionary
    comms = {i: (division == i).nonzero().squeeze(1) for i in range(total_comm) if (division == i).any()}

    # Initialize variables for tracking bot predictions and community evaluation values
    bot_rate = []
    pre_bot = np.zeros(num)
    value = np.zeros(num)
    num_bot_comm = 0
    # Iterate through each community to evaluate its likelihood of being a bot community
    for i in comms.keys():
        comm = comms[i].cpu().numpy()
        n_bots = label_block[comm].sum()
        n_nodes = len(comm)
        n_x = x[comm].sum()
        # Calculate the community entropy and the community evaluation index n_beta
        comm_SE = module_se[i].cpu().numpy()
        n_beta = (n_x / n_nodes) / (1 / num) * community_influence_weight + comm_SE / (sum(module_se).cpu().numpy() / total_comm.cpu().numpy()) * community_entropy_weight
        # Determine whether the community should be classified as a bot community based on n_beta
        if n_beta >= evaluation_index_treshold:
            num_bot_comm += 1
            pre_bot[comm] = 1
            value[comm] = n_beta
        else:
            value[comm] = n_beta
        # Record the community's bot rate and other metrics
        bot_rate.append([n_bots / n_nodes, n_bots, n_nodes, n_x / n_nodes, comm_SE, n_beta])

    value = np.nan_to_num(value)

    # Derive performance metrics
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

    # Create ROC Curve for this test
    plt.figure()
    plt.plot(fpr, tpr, color='navy', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
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
    test('twibot-22/sample1/sampled_combined_preprocessed_twibot22_bots.csv', 'twibot-22/sample1/sampled_combined_preprocessed_twibot22_humans.csv', dataset_path='')
    end_time = time.time()
    print(f'running time: {(end_time - start_time) / 1} s')
    print(f'End time is {time.strftime("%x %X")}')
