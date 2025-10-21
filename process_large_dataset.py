from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import scipy.io
import pickle
import pandas as pd
from sklearn.preprocessing import label_binarize
import gdown
from os import path
import os
#from utils import *
from generateSimFromLargeData import *
# from load_data import load_twitch, load_fb100, load_twitch_gamer, DATAPATH
# from data_utils import rand_train_test_idx, even_quantile_labels, to_sparse_tensor, dataset_drive_url

# from homophily import our_measure, edge_homophily_edge_idx

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
# from torch_sparse import SparseTensor
from ogb.nodeproppred import NodePropPredDataset
import dgl
import networkx as nx
import random
import json
from scipy import sparse as sp
DATAPATH = path.dirname(path.abspath(__file__)) + '/data/'
dataset_drive_url = {
    'twitch-gamer_feat' : '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
    'twitch-gamer_edges' : '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
    'snap-patents' : '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia', 
    'pokec' : '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y', 
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ', 
    'wiki_views': '1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP', # Wiki 1.9M 
    'wiki_edges': '14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u', # Wiki 1.9M 
    'wiki_features': '1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK' # Wiki 1.9M
}

splits_drive_url = {
    'snap-patents' : '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N', 
    'pokec' : '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_', 
}
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_genius():
    fulldata = scipy.io.loadmat(f'data/genius/genius.mat')

    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat'], dtype=torch.float)
    label = torch.tensor(fulldata['label'], dtype=torch.long).squeeze()
    num_nodes = label.shape[0]
    print('num_nodes:', num_nodes)

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(num_nodes))
    edge_index_np = edge_index.numpy()
    nx_graph.add_edges_from(edge_index_np.T)
    
    
    graph = dgl.graph((edge_index[0], edge_index[1]))

    
    
    
    # Calculate the diameter of the graph
    if nx.is_connected(nx_graph):
        graph_diameter = nx.diameter(nx_graph)
        print(f"Graph Diameter: {graph_diameter}")
    else:
        print("The graph is not connected; diameter is undefined.")

    return graph,nx_graph, node_feat, label
def load_penn94():
    mat = scipy.io.loadmat(DATAPATH + 'penn94/Penn94.mat')
    A = mat['A']
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    # Convert A to a symmetric sparse matrix and set diagonal to 1
    A = scipy.sparse.csr_matrix(A)
    
    A = A + A.T  # Make symmetric
    A.data = np.minimum(A.data, 1) 
    A=normalize(A)# Ensure values do not exceed 1
    A.setdiag(1)  # Set diagonal to 1
    A = torch.sparse_coo_tensor(
        torch.tensor(A.nonzero(), dtype=torch.long),
        torch.tensor(A.data, dtype=torch.float),
        size=A.shape
    )
    
    metadata = mat['local_info']
    
    # Convert edge_index to a symmetric sparse adjacency matrix with diagonal set to 1
    
    metadata = metadata.astype(int)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled

    # make features into one-hot encodings
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    node_feat = torch.tensor(features, dtype=torch.float)
    
    #graph = dgl.graph((edge_index[0], edge_index[1]))
    label = torch.tensor(label)
    
    num_nodes = metadata.shape[0]
    masks=generate_or_load_masks(num_nodes,mask_file='data/penn94/masks.pt')
    train_mask = masks['train_mask']
    val_mask = masks['val_mask']
    test_mask = masks['test_mask']
    
    num_classes = len(np.unique(label))
    label = torch.clamp(label, min=0, max=num_classes - 1)
    
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(num_nodes))
    edge_index_np = edge_index.numpy()
    nx_graph.add_edges_from(edge_index_np.T)
    return A, nx_graph, node_feat, label, train_mask, val_mask, test_mask,num_classes

def generate_or_load_masks(num_nodes, mask_file='masks.pt', train_ratio=0.6, val_ratio=0.2):
    """
    Generate or load train, validation, and test masks for nodes.

    Args:
        num_nodes (int): Total number of nodes.
        mask_file (str): Path to the file where masks are saved.
        train_ratio (float): Proportion of nodes for training.
        val_ratio (float): Proportion of nodes for validation.

    Returns:
        dict: A dictionary containing train, validation, and test masks as tensors.
    """
    if path.exists(mask_file):
        print(f"Loading masks from {mask_file}...")
        masks = torch.load(mask_file)  # 使用 torch.load 加载张量
    else:
        print(f"Generating new masks and saving to {mask_file}...")
        indices = list(range(num_nodes))
        random.shuffle(indices)

        train_size = int(train_ratio * num_nodes)
        val_size = int(val_ratio * num_nodes)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True

        masks = {
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask
        }

        torch.save(masks, mask_file)  # 使用 torch.save 保存张量

    return masks

def load_snap_patents_mat(nclass=5):
    datapath=DATAPATH+'snap_patents/'
    if not path.exists(f'{datapath}snap_patents.mat'):
        p = dataset_drive_url['snap-patents']
        print(f"Snap patents url: {p}")
        gdown.download(id=dataset_drive_url['snap-patents'], \
            output=f'{datapath}snap_patents.mat', quiet=False)

    fulldata = scipy.io.loadmat(f'{datapath}snap_patents.mat')
    
    # Convert A to a symmetric sparse matrix and set diagonal to 1
    
    num_nodes = int(fulldata['num_nodes'])
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    # Convert edge_index to a NetworkX graph
    edge_index_np = fulldata['edge_index']
    
    # Create a sparse adjacency matrix A from edge_index_np
    row, col = edge_index_np
    data = np.ones(row.shape[0])  # All edges have weight 1
    A = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    # Ensure A is symmetric and set diagonal to 1
    A = A + A.T  # Make symmetric
    A.data = np.minimum(A.data, 1) 
    A.setdiag(1)
    A=normalize(A)# Ensure values do not exceed 1
      # Set diagonal to 1
    A = torch.sparse_coo_tensor(
        torch.tensor(A.nonzero(), dtype=torch.long),
        torch.tensor(A.data, dtype=torch.float),
        size=A.shape
    )
    
    graph = dgl.graph((edge_index[0], edge_index[1]))
    graph = dgl.add_self_loop(graph)
    node_feat = torch.tensor(
        fulldata['node_feat'].todense(), dtype=torch.float)
    
    
    masks=generate_or_load_masks(num_nodes,mask_file='data/snap_patents/masks.pt')
    train_mask = masks['train_mask']
    val_mask = masks['val_mask']
    test_mask = masks['test_mask']
    print('dataset:snap_patents')
    print('num_nodes:', num_nodes)
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(num_nodes))
    edge_index_np = edge_index.numpy()
    nx_graph.add_edges_from(edge_index_np.T)
    
    years = fulldata['years'].flatten()
    label = even_quantile_labels(years, nclass, verbose=False)
    label = torch.tensor(label, dtype=torch.long)
    num_classes = len(np.unique(label))
    # Calculate the diameter of the graph
    if nx.is_connected(nx_graph):
        graph_diameter = nx.diameter(nx_graph)
        print(f"Graph Diameter: {graph_diameter}")
    else:
        print("The graph is not connected; diameter is undefined.")

    return A,nx_graph,node_feat, label, train_mask, val_mask, test_mask, num_classes,graph


def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    
    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label

def generateMatrix(dataset='snap_patents'):
    print('generating graph...')
    #graph,nx_graph,_,_=load_snap_patents_mat()
    #graph,nx_graph,_,_=load_penn94()
    # Convert DGL graph to NetworkX graph
    # print('converting graph...')
    # nx_graph = graph.to_networkx()

    # # Ensure the graph is undirected (if needed)
    if dataset=='snap_patents':
        adj, nx_graph,features, labels, train_mask, val_mask, test_mask, num_classes,graph=load_snap_patents_mat()
    elif dataset=='penn94':
        graph,nx_graph,_,_=load_penn94()
    elif dataset=='genius':
        graph,nx_graph,_,_=load_genius()
    # nx_graph = nx_graph.to_undirected()
    
    filedir = 'data/'+dataset+'/'
    chooserate=1
    print ('generating matrix...')
    print('graph.num_nodes():',nx_graph.number_of_nodes())
    
    #generateSimMatrix_Pagerank_optimized(nx_graph,filedir,nx_graph.number_of_nodes(),select_rate=0.01,match_rate=1,cache_path='data/'+dataset+'/community_cache.npy',percentile=99.999)
    generateSimMatrix_Pagerank_optimized(nx_graph,filedir,nx_graph.number_of_nodes(),select_rate=0.5,match_rate=1,cache_path='data/'+dataset+'/community_cache.npy',percentile=99.9)

    
if __name__=='__main__':
    dataset='snap_patents'
    generateMatrix(dataset)