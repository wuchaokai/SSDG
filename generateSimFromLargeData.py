import networkx as nx
from collections import defaultdict
import heapq
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import os
from math import exp
#from Pagerank import pr
from scipy.sparse import lil_matrix
import dgl
from scipy.sparse import save_npz
from scipy.sparse import load_npz
from scipy.sparse import csr_matrix,coo_matrix
from concurrent.futures import ThreadPoolExecutor
import community as community_louvain  # pip install python-louvain
from networkx.algorithms.community import label_propagation_communities
import torch
from torch_sparse import SparseTensor
from scipy.sparse import triu
from concurrent.futures import ProcessPoolExecutor
import h5py

def get_second_order_neighbors(graph, node):
    first_order_neighbors = set(graph.neighbors(node))
    second_order_neighbors = set()

    for neighbor in first_order_neighbors:
        second_order_neighbors.update(graph.neighbors(neighbor))

    second_order_neighbors.update(first_order_neighbors)
    return set(second_order_neighbors)


def calculate_in_out_degree(graph, second_order_neighbors):

    in_deg = 0
    out_deg = 0
    for neighbor in second_order_neighbors:
        for n in graph.neighbors(neighbor):
            if n in second_order_neighbors:
                in_deg += 1
            else:
                out_deg += 1

    if in_deg==0:
        print(in_deg)
        print(out_deg)

    return out_deg/in_deg

def get_top_k_nodes(degrees_dict, k):
    #return [node for node, degree in heapq.nsmallest(k, degrees_dict.items(), key=lambda x: x[1])]
    return [node for node, degree in heapq.nsmallest(k, degrees_dict.items(), key=lambda x: x[1])]


def precompute_neighbors(graph):
    """
    预计算所有节点的一阶和二阶邻居。

    参数:
    graph (networkx.Graph): 输入的图。

    返回:
    dict: 每个节点对应的一阶和二阶邻居集合。
    """
    neighbors_cache = {}
    for node in graph:
        first_order_neighbors = set(graph.neighbors(node))
        second_order_neighbors = set()
        for neighbor in first_order_neighbors:
            second_order_neighbors.update(graph.neighbors(neighbor))
        second_order_neighbors.update(first_order_neighbors)
        neighbors_cache[node] = second_order_neighbors
    return neighbors_cache

def compute_score(graph, node, neighbors_cache):
    """
    计算节点的分数。

    参数:
    graph (networkx.Graph): 输入的图。
    node (int): 节点 ID。
    neighbors_cache (dict): 节点的邻居缓存。

    返回:
    tuple: 节点 ID 和分数。
    """
    if graph.degree(node) == 0:
        return node, float('inf')  # 使用无穷大代替 1000000
    second_order_neighbors = neighbors_cache[node]
    score = calculate_in_out_degree(graph, second_order_neighbors)
    return node, score

def choose_oversquashed_nodes(graph, select_rate):
    """
    选择过度压缩的节点（oversquashed nodes）。

    参数:
    graph (networkx.Graph): 输入的图。
    select_rate (float): 选择节点的比例，范围为 (0, 1]。

    返回:
    list: 选择的节点列表。
    """
    if select_rate == 1:
        return list(graph.nodes())

    # 预计算所有节点的邻居
    print("Precomputing neighbors...")
    neighbors_cache = precompute_neighbors(graph)

    # 并行计算分数
    print("Computing scores...")
    score_list = {}
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda node: compute_score(graph, node, neighbors_cache), graph.nodes())
        for node, score in results:
            score_list[node] = score

    # 选择前 k 个节点
    k = max(int(select_rate * len(score_list)), 1)
    node_list = get_top_k_nodes(score_list, k)

    return node_list


def find_anchor_points_spectral(G, select_rate=0.1):
    if select_rate==1:
        return list(G.nodes())
    adjacency_matrix = nx.to_numpy_array(G)
    n_clusters = max(1, int(select_rate*len(G.nodes)))
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    labels = spectral.fit_predict(adjacency_matrix)
    clusters = {}
    for node, label in zip(G.nodes, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(node)
    anchor_points = []
    for cluster_nodes in clusters.values():
        representative_node = max(cluster_nodes, key=lambda node: G.degree(node))
        #representative_node=sorted(cluster_nodes, key=lambda node: G.degree(node), reverse=True)
        anchor_points.append(representative_node)
    return anchor_points
def compute_external_connections_sparse(G, clusters,num_nodes,cache_dir=None,oversquashed_nodes=None):
    """
    使用稀疏矩阵计算每个节点的外部连接比例。

    参数:
    G (networkx.Graph): 输入的图。
    clusters (dict): 每个社区的节点分组。

    返回:
    dict: 每个节点的外部连接比例。
    
    """
    try:
        adjacency_matrix = nx.to_scipy_sparse_matrix(G, format='csr')
    except AttributeError:
        try:
            adjacency_matrix = nx.to_scipy_sparse_array(G, format='csr')
        except AttributeError:
            # 回退：先得到密集邻接矩阵再转换为 CSR（内存敏感时注意）
            adjacency_matrix = csr_matrix(nx.to_numpy_array(G))
    # 确保返回类型为 scipy.sparse.csr_matrix（统一后续处理）
    adjacency_matrix = csr_matrix(adjacency_matrix)
    external_connections = {}
    print(num_nodes,adjacency_matrix.shape)
    def process_cluster(cluster_nodes):
        cluster_mask = np.zeros(num_nodes, dtype=bool)
        cluster_mask[cluster_nodes] = True
        
        internal_connections = adjacency_matrix[cluster_mask][:, cluster_mask].sum(axis=1).A.flatten()
        total_connections = adjacency_matrix[cluster_mask].sum(axis=1).A.flatten()
        external_ratios = (total_connections - internal_connections) / total_connections
        external_ratios[np.isnan(external_ratios)] = float('inf')  # 处理除以 0 的情况

        return {node: ratio for node, ratio in zip(cluster_nodes, external_ratios)}

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_cluster, clusters.values())
        for result in results:
            external_connections.update(result)

    return external_connections
def process_subgraph(subgraph):
    """优化的子图处理函数"""
    print('subgraph',len(subgraph))
    
    # 使用更高效的标签传播实现
    labels = {node: i for i, node in enumerate(subgraph.nodes())}
    max_iter = 5  # 限制迭代次数
    
    for _ in range(max_iter):
        # 随机打乱节点顺序
        nodes = list(subgraph.nodes())
        random.shuffle(nodes)
        
        # 更新标签
        for node in nodes:
            neighbor_labels = [labels[neighbor] for neighbor in subgraph.neighbors(node)]
            if not neighbor_labels:
                continue
            # 选择最常见的标签
            label_count = {}
            for label in neighbor_labels:
                label_count[label, 0] = label_count.get(label, 0) + 1
            labels[node] = max(label_count.items(), key=lambda x: x[1])[0]
    
    # 将结果转换为社区格式
    communities = defaultdict(list)
    for node, label in labels.items():
        communities[label].append(node)
    
    return list(communities.values())
def parallel_label_propagation(G, num_partitions=None,cache_path=None):
    """
    使用并行标签传播算法划分社区，并处理节点数不超过 10 的子图。
    
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading communities from cache: {cache_path}")
        return load_communities(cache_path)
    if num_partitions is None:
        num_partitions = max(1, int(os.cpu_count() * 0.75))
    else:
        num_partitions = int(num_partitions)
    # 将图分成多个子图
    print('split subgraphs')
    subgraphs = list(nx.connected_components(G))
    print('subgraphs', len(subgraphs))
    small_subgraphs = []
    large_subgraphs = []
    
    for nodes in subgraphs:
        if len(nodes) <= 10 and len(nodes) > 1:
            small_subgraphs.append(nodes)
        elif len(nodes) > 10:
            large_subgraphs.append(G.subgraph(nodes))
    
    print('large subgraphs', len(large_subgraphs))
    print('parallel label propagation')
    # 并行运行标签传播算法
    communities = []
    small_subgraph_nodes = []  # 用于存储小子图的节点
    batch_size = max(1, len(large_subgraphs) // (num_partitions * 2))
    with ProcessPoolExecutor(max_workers=num_partitions) as executor:
        # 使用批处理来减少进程间通信开销
        for i in range(0, len(large_subgraphs), batch_size):
            batch = large_subgraphs[i:i + batch_size]
            results = list(executor.map(process_subgraph, batch))
            for result in results:
                if result:
                    communities.extend(result)
    
    # 处理小型子图
    print('process small subgraphs')
    for nodes in small_subgraphs:
        small_subgraph_nodes.append(random.choice(list(nodes)))
    
    print('merge communities')
    # 合并社区
    clusters = {i: list(community) for i, community in enumerate(communities)}
    
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        save_communities(clusters, small_subgraph_nodes, cache_path)
        print(f"Saved communities to cache: {cache_path}")
    
    return clusters, small_subgraph_nodes
def find_anchor_points_louvain(G, num_nodes,select_rate=0.1,cache_path=None):
    """
    使用 Louvain 算法选择锚点节点。

    参数:
    G (networkx.Graph): 输入的图。
    select_rate (float): 选择节点的比例，范围为 (0, 1]。

    返回:
    list: 锚点节点列表。
    """
    if select_rate == 1:
        return list(G.nodes()), list(G.nodes())

    # 使用 Louvain 算法进行社区检测
    clusters,small_subgraph_nodes=parallel_label_propagation(G,cache_path=cache_path)
    print('clusters',len(clusters))
    print('find over-squashed nodes')

    # 调用 compute_external_connections_sparse 计算外部连接比例
    external_connections = compute_external_connections_sparse(G, clusters,num_nodes)

    # 按照外部连接比例排序，选出全图前 K 个节点
    k = max(1, int(select_rate * len(G.nodes())))
    print('select_rate',select_rate)
    print('k',k)
    oversquashed_nodes = sorted(external_connections, key=external_connections.get, reverse=False)[:k]
    oversquashed_nodes.extend(small_subgraph_nodes)
    print('find anchor nodes')

    # 从每个社区中选择一个代表节点（度数最大的节点）
    anchor_points = []
    for cluster_nodes in clusters.values():
        if (len(cluster_nodes)>1):
            representative_node = max(cluster_nodes, key=lambda node: G.degree(node))
            anchor_points.append(representative_node)

    return oversquashed_nodes, anchor_points

def degree_matrix(G,num_nodes):
    """
    计算图中每个节点的度数，并返回一个度数矩阵。

    参数:
    G (networkx.Graph): 输入的图。

    返回:
    degree_matrix (numpy.ndarray): 度数矩阵，每行一个节点的度数。
    """
    nodes = list(G.nodes())
    degree_matrix = np.zeros((num_nodes, 1), dtype=int)

    for node in nodes:
        degree_matrix[node] = G.degree(node)

    return degree_matrix


def complete_graph_with_all_nodes(G, num_nodes):
    """
    输入一个图和节点个数，确保输出一个包含所有指定节点的完整图。

    参数:
    G (networkx.Graph): 输入的图。
    num_nodes (int): 节点的总个数。

    返回:
    complete_G (networkx.Graph): 包含所有指定节点的完整图。
    """
    # 获取已有的所有节点
    existing_nodes = set(G.nodes())

    # 获取孤立的节点
    isolated_nodes = [node for node in range(num_nodes) if node not in existing_nodes]

    # 创建一个新的图
    complete_G = G.copy()

    # 添加孤立节点到图中
    for node in isolated_nodes:
        complete_G.add_node(node)

    return complete_G



def cacDistance(a, b):
    return torch.min(a, b) / torch.max(a, b)

    
def pr(graph, isDirected=False):
    """
    使用 NetworkX 的 PageRank 实现。
    """
    #nx_graph = graph if isDirected else graph.to_undirected()
    ranks = nx.pagerank(graph, alpha=0.85)  # alpha 是阻尼系数
    return ranks


from concurrent.futures import ThreadPoolExecutor
import time
import random

def process_node_pair(nodei, graph, matched_nodes, ranks, simMatrix, path):
    """
    处理单个节点对的逻辑。
    """
    print(nodei, path)
    shortest_paths = nx.single_source_shortest_path_length(graph, source=nodei)
    for nodej in matched_nodes:
        if nodej in shortest_paths and shortest_paths[nodej] > 3 and graph.degree(nodei) != 0 and graph.degree(nodej) != 0:
            simMatrix[nodei, nodej] = cacDistance(ranks[nodei], ranks[nodej])
            simMatrix[nodej, nodei] = simMatrix[nodei, nodej]
            

def parallel_process(graph, oversquashed_nodes, matched_nodes, ranks, simMatrix, path):
    """
    并行处理 oversquashed_nodes 中的每个节点。
    """
    with ThreadPoolExecutor() as executor:
        executor.map(lambda nodei: process_node_pair(nodei, graph, matched_nodes, ranks, simMatrix, path), oversquashed_nodes)

def generateSimMatrix_Pagerank(graph, path, node_num, select_rate=1, match_rate=1):
    os.makedirs(path, exist_ok=True)
    print('find nodes start...')
    oversquashed_nodes, matched_nodes = find_anchor_points_louvain(graph, select_rate)
    print('find nodes end...')
    # Randomly select 1% of oversquashed_nodes and matched_nodes
    # oversquashed_nodes = random.sample(oversquashed_nodes, max(1, int(0.01 * len(oversquashed_nodes))))
    # matched_nodes = random.sample(matched_nodes, max(1, int(0.01 * len(matched_nodes))))
    simMatrix = lil_matrix((node_num, node_num), dtype=np.float32)

    print('calculate pagerank')
    ranks = pr(graph)
    print('calculate pagerank done')

    # 并行处理
    start_time = time.time()
    parallel_process(graph, oversquashed_nodes, matched_nodes, ranks, simMatrix, path)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time / 60:.2f} minutes")

    simMatrix_csr = simMatrix.tocsr()
    save_npz(path + 'simMatrixHop_{}{}.npz'.format(str(select_rate), str(match_rate)), simMatrix_csr)
    print('save simMatrixHop_{}{}.npy'.format(str(select_rate), str(match_rate)))

def randomSimMatrix(simMatrix):
    rows, cols = simMatrix.shape
    density = 0.0001  # Adjust the density of the sparse matrix
    num_elements = int(rows * cols * density)

    # Randomly select indices to fill
    row_indices = np.random.choice(rows, num_elements, replace=True)
    col_indices = np.random.choice(cols, num_elements, replace=True)
    values = np.random.rand(num_elements)

    # Fill the selected indices with random values
    for r, c, v in zip(row_indices, col_indices, values):
        simMatrix[r, c] = v
    return simMatrix
def retain_values(matrix, row_indices, col_indices):
    """
    保持行索引和列索引同时覆盖到的数值不变，其他值清0。

    参数:
    matrix (np.ndarray): 原始矩阵
    row_indices (list of int): 行索引列表
    col_indices (list of int): 列索引列表

    返回:
    np.ndarray: 新矩阵，其中只有指定位置的值保留，其余值为零
    """

    new_matrix = lil_matrix(matrix.shape, dtype=matrix.dtype)

    # 遍历指定的行和列索引
    for i in row_indices:
        for j in col_indices:
            new_matrix[i, j] = matrix[i, j]

    return new_matrix
def save_sparse_h5(filename, matrix):
    """将稀疏矩阵保存为 HDF5 格式"""
    with h5py.File(filename, 'w') as f:
        g = f.create_group('matrix')
        g.create_dataset('data', data=matrix.data, chunks=True)  # 启用分块存储
        g.create_dataset('indices', data=matrix.indices)
        g.create_dataset('indptr', data=matrix.indptr)
        g.attrs['shape'] = matrix.shape
def cacWeight(graph, path, select_rate=1,percentile=0,highmemoryutilization=False):
    """从 HDF5 文件加载并返回 PyTorch 稀疏张量"""
    h5_filename = path + f'simMatrixHop_{str(select_rate)}{str(percentile)}.h5'
    npz_filename = path + f'simMatrixHop_{str(select_rate)}{str(percentile)}.npz'
    print('cacWeight:',percentile)
    if not os.path.exists(h5_filename):
        print("Converting NPZ to HDF5 format...")
        matrix = load_npz(npz_filename)
        save_sparse_h5(h5_filename, matrix)
        print("Conversion complete")
        indices = torch.from_numpy(
            np.vstack((matrix.nonzero()[0], matrix.nonzero()[1]))
        ).long()
        values = torch.from_numpy(matrix.data).float()
        # 将 matrix.shape 转换为 tuple
        shape = tuple(matrix.shape)
        return torch.sparse_coo_tensor(
            indices, values, 
            size=shape  # 使用 tuple 类型的 shape

        )
    
    # 从 HDF5 文件加载
    if not highmemoryutilization:
        with h5py.File(h5_filename, 'r') as f:
            g = f['matrix']
            indices = torch.from_numpy(
                np.vstack((g['indices'][:], np.repeat(np.arange(len(g['indptr'])-1), 
                    np.diff(g['indptr']))))
            ).long()
            values = torch.from_numpy(g['data'][:]).float()
            # 将 shape 转换为 tuple
            shape = tuple(g.attrs['shape'])
            
            return torch.sparse_coo_tensor(
                indices, values, 
                size=shape  # 使用 tuple 类型的 shape

            )
    elif highmemoryutilization:
        import torch_scatter
        k=20
        with h5py.File(h5_filename, 'r') as f:
            g = f['matrix']
            
            # 直接读取 CSR 格式的数据
            indptr = g['indptr'][:]
            indices_data = g['indices'][:]
            data = g['data'][:]
            shape = tuple(g.attrs['shape'])
            
            n_rows = shape[0]
            
            # 收集要保留的索引和数据
            new_indices_list = []
            new_data_list = []
            
            # 逐行处理（但使用 NumPy 向量化操作）
            for i in range(n_rows):
                start, end = indptr[i], indptr[i+1]
                if end > start:
                    row_data = data[start:end]
                    row_indices = indices_data[start:end]
                    
                    # 取 top-k
                    if k == 1:
                        topk_idx = np.argmax(row_data)
                    else:
                        topk_idx = np.argpartition(-row_data, min(k, len(row_data))-1)[:k]
                    
                    new_indices_list.extend([(i, idx) for idx in row_indices[topk_idx]])
                    new_data_list.extend(row_data[topk_idx])
            
            if not new_data_list:
                return torch.sparse_coo_tensor(size=shape)
            
            # 转换为 COO 格式
            new_indices = np.array(new_indices_list).T
            new_values = np.array(new_data_list)
            
            return torch.sparse_coo_tensor(
                torch.from_numpy(new_indices).long(),
                torch.from_numpy(new_values).float(),
                size=shape
            )
    # if select_rate!=1 or match_rate!=1:
    #     oversquashedNodes=sorted(choose_oversquashed_nodes(graph,select_rate))
    #     matchNodes=sorted(find_anchor_points_spectral(graph,match_rate))
    #     simMatrix=retain_values(simMatrix,oversquashedNodes,matchNodes)
    

def generateStructWeight(graph,path,select_rate=1,percentile=0,highmemoryutilization=False):
    print('generateStructWeight',percentile)
    Weight = cacWeight(graph, path,select_rate,percentile,highmemoryutilization)
    
    print('normalize start...')
    Weight = normalize(Weight)
    print('normalize end...')
    return Weight

def calculateThreshold(weights,threshold):
    non_zero_weights = weights[weights != 0]
    return np.percentile(non_zero_weights, threshold)

def normalize(mx):
    if isinstance(mx, torch.Tensor):
        # 获取稀疏张量的索引和值
        indices = mx._indices()
        values = mx._values()
        shape = mx.shape

        # 计算每行的和
        rowsum = torch.zeros(shape[0], device=values.device)
        rowsum.scatter_add_(0, indices[0], values)
        
        # 避免除以零
        rowsum[rowsum == 0] = 1
        
        # 计算归一化值
        norm_values = values / rowsum[indices[0]]
        
        # 构建新的归一化后的稀疏张量
        return torch.sparse_coo_tensor(indices, norm_values, shape)
    else:
        # 如果是 scipy 稀疏矩阵，使用原来的方法
        rowsum = np.array(mx.sum(axis=1)).flatten()
        rowsum[rowsum == 0] = 1
        inv_rowsum = 1.0 / rowsum
        inv_diag = csr_matrix(
            (inv_rowsum, (np.arange(len(rowsum)), np.arange(len(rowsum)))), 
            shape=(len(rowsum), len(rowsum))
        )
        return inv_diag.dot(mx)



def visualize_graph_with_anchor_points(G, anchor_points):
    """
    可视化图和锚点。

    参数:
    G (networkx.Graph): 输入的图。
    anchor_points (list): 锚点列表。
    """
    pos = nx.spring_layout(G)
    labels = {node: G.nodes[node].get('label', node) for node in G.nodes}
    colors = ['black' if node in anchor_points else 'red' for node in G.nodes]

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, node_color=colors, with_labels=True, node_size=300, cmap=plt.cm.rainbow)
    nx.draw_networkx_nodes(G, pos, nodelist=anchor_points, node_color='blue', node_size=500)
    plt.show()


def compute_neighbor_degrees(adj):
    """
    计算每个节点的度数列表（包括自身、一阶和二阶邻居）。

    参数：
    adj -- 稀疏邻接矩阵（scipy.sparse 矩阵）

    返回：
    zero_order_degrees -- 自身度数列表
    first_order_degrees -- 一阶邻居度数列表
    second_order_degrees -- 二阶邻居度数列表
    """

    G = nx.from_scipy_sparse_matrix(adj)


    zero_order_degrees = []
    first_order_degrees = []
    second_order_degrees = []


    for node in G.nodes():

        zero_degree = G.degree(node)
        zero_order_degrees.append(zero_degree)


        first_neighbors = list(G.neighbors(node))
        first_degrees = [G.degree(neighbor) for neighbor in first_neighbors]
        first_order_degrees.append(first_degrees)


        second_neighbors = set()
        for neighbor in first_neighbors:
            second_neighbors.update(G.neighbors(neighbor))
        #second_neighbors.discard(node)
        second_degrees = [G.degree(neighbor) for neighbor in second_neighbors]
        second_order_degrees.append(second_degrees)

    return zero_order_degrees, first_order_degrees, second_order_degrees

def compute_similarity_gpu_batch(graph, oversquashed_nodes, matched_nodes, ranks, simMatrix):
    """
    使用 GPU 和稀疏矩阵批量计算节点对的相似度。
    """

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    # 将 oversquashed_nodes 和 matched_nodes 转换为张量
    oversquashed_nodes_tensor = torch.tensor(oversquashed_nodes, dtype=torch.long).to(device)
    matched_nodes_tensor = torch.tensor(matched_nodes, dtype=torch.long).to(device)
    block_size = 30000
    ranks=ranks.to(device)
    print('start calculate similarity')
    for i in range(0, len(oversquashed_nodes_tensor), block_size):
        for j in range(0, len(matched_nodes_tensor), block_size):
            # 获取当前块
            oversquashed_block = oversquashed_nodes_tensor[i:i + block_size]
            matched_block = matched_nodes_tensor[j:j + block_size]

            # 计算当前块的笛卡尔积
            node_pairs_block = torch.cartesian_prod(oversquashed_block, matched_block)

            # 提取节点对的索引
            nodei_indices = node_pairs_block[:, 0]
            nodej_indices = node_pairs_block[:, 1]

            # 计算相似度
            similarities = cacDistance(ranks[nodei_indices], ranks[nodej_indices])

            # 将结果写入 simMatrix
            nodei_indices_np = nodei_indices.cpu().numpy()
            nodej_indices_np = nodej_indices.cpu().numpy()
            similarities_np = similarities.cpu().numpy()

            # 写入稀疏矩阵
            simMatrix += coo_matrix((similarities_np, (nodei_indices_np, nodej_indices_np)), shape=simMatrix.shape)
            print('block',i,j)
    simMatrix = triu(simMatrix)

    simMatrix = simMatrix + simMatrix.T
    return simMatrix


def generateSimMatrix_Pagerank_optimized(graph, path, node_num, select_rate=1, match_rate=1,cache_path=None,percentile=99):
    """
    使用 PageRank 和 GPU 加速生成相似度矩阵。
    """
    os.makedirs(path, exist_ok=True)
    print('find nodes start...')
    oversquashed_nodes, matched_nodes = find_anchor_points_louvain(graph, node_num,select_rate,cache_path)
    print('oversquashed_nodes',len(oversquashed_nodes))
    print('matched_nodes',len(matched_nodes))
    print('find nodes end...')
    simMatrix = coo_matrix((node_num, node_num), dtype=np.float32)

    print('calculate pagerank')
    ranks = pr(graph)
    ranks_array = np.zeros(node_num, dtype=np.float32)
    for node, rank in ranks.items():
        ranks_array[node] = rank
    ranks=torch.tensor(ranks_array, dtype=torch.float32)
    print('calculate pagerank done')

    
    print("Using GPU for similarity computation...")
    simMatrix=compute_similarity_gpu_batch_with_threshold(graph, oversquashed_nodes, matched_nodes, ranks, simMatrix,percentile=percentile)


    simMatrix_csr = simMatrix.tocsr()
    save_sparse_h5(path + f'simMatrixHop_{str(select_rate)}{str(percentile)}.h5', simMatrix_csr)
    print(f'save simMatrixHop_{str(select_rate)}.h5')
    
def save_communities(clusters, small_subgraph_nodes, filepath):
    """保存社区分区结果到文件"""
    data = {
        'clusters': clusters,
        'small_subgraph_nodes': small_subgraph_nodes
    }
    np.save(filepath, data, allow_pickle=True)    
def load_communities(filepath):
    """从文件加载社区分区结果"""
    data = np.load(filepath, allow_pickle=True).item()
    return data['clusters'], data['small_subgraph_nodes']
def compute_approximate_quantile(values, percentile, chunk_size=1000000):
    """
    计算大规模数据的近似分位数
    """
    print("计算近似分位数...")
    
    # 收集样本
    sample_size = min(chunk_size, len(values))
    indices = torch.randperm(len(values))[:sample_size]
    sampled_values = values[indices]
    
    print(f"原始数据大小: {len(values)}")
    print(f"采样数量: {len(sampled_values)}")
    return torch.quantile(sampled_values, percentile/100)

def compute_similarity_gpu_batch_with_threshold(graph, oversquashed_nodes, matched_nodes, ranks, simMatrix, percentile):
    """使用 GPU 计算节点对的相似度，即时处理阈值和矩阵构建"""
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    oversquashed_nodes_tensor = torch.tensor(oversquashed_nodes, dtype=torch.long).to(device)
    matched_nodes_tensor = torch.tensor(matched_nodes, dtype=torch.long).to(device)
    block_size = 40000
    ranks = ranks.to(device)
    
    def process_block(oversquashed_block, matched_block):
        """处理单个数据块"""
        # 在 GPU 上计算节点对和相似度
        node_pairs_block = torch.cartesian_prod(oversquashed_block, matched_block)
        similarities = cacDistance(
            ranks[node_pairs_block[:, 0]], 
            ranks[node_pairs_block[:, 1]]
        )
        
        # 筛选非零值
        non_zero_mask = similarities > 0
        if not non_zero_mask.any():
            return None
        
        # 计算当前块的阈值
        block_values = similarities[non_zero_mask]
        block_threshold = compute_approximate_quantile(block_values,percentile=percentile)
        
        # 应用阈值
        threshold_mask = similarities > block_threshold
        if not threshold_mask.any():
            return None
        
        # 获取满足条件的节点对和值
        filtered_pairs = node_pairs_block[threshold_mask]
        filtered_values = similarities[threshold_mask]
        
        # 转移到 CPU 并更新稀疏矩阵
        return coo_matrix(
            (filtered_values.cpu().numpy(), 
             (filtered_pairs[:, 0].cpu().numpy(), filtered_pairs[:, 1].cpu().numpy())),
            shape=simMatrix.shape
        )
    
    print('计算相似度并构建矩阵...')
    for i in range(0, len(oversquashed_nodes_tensor), block_size):
        for j in range(0, len(matched_nodes_tensor), block_size):
            oversquashed_block = oversquashed_nodes_tensor[i:i + block_size]
            matched_block = matched_nodes_tensor[j:j + block_size]
            
            # 处理当前块
            block_matrix = process_block(oversquashed_block, matched_block)
            if block_matrix is not None:
                simMatrix += block_matrix
            
            print(f'处理块 {i},{j}')
            torch.cuda.empty_cache()
    
    # 处理对称性
    simMatrix = triu(simMatrix)
    simMatrix = simMatrix + simMatrix.T
    
    return simMatrix
# def merge_blocks(path, simMatrix_shape):
#     """
#     从磁盘读取所有块并合并为一个完整的稀疏矩阵。
#     """
#     import os
#     files = [f for f in os.listdir(path) if f.startswith("block_") and f.endswith(".npz")]
#     merged_matrix = coo_matrix(simMatrix_shape, dtype=np.float32)

#     for file in files:
#         block_matrix = load_npz(os.path.join(path, file))
#         merged_matrix += block_matrix

#     return merged_matrix

# def compute_similarity_gpu_batch(graph, oversquashed_nodes, matched_nodes, ranks, path, simMatrix_shape):
#     """
#     使用 GPU 和稀疏矩阵批量计算节点对的相似度，并将每个块写入磁盘。
#     """
#     device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
#     oversquashed_nodes_tensor = torch.tensor(oversquashed_nodes, dtype=torch.long).to(device)
#     matched_nodes_tensor = torch.tensor(matched_nodes, dtype=torch.long).to(device)
#     block_size = 30000
#     ranks = ranks.to(device)

#     print('start calculate similarity')
#     for i in range(0, len(oversquashed_nodes_tensor), block_size):
#         for j in range(0, len(matched_nodes_tensor), block_size):
#             # 获取当前块
#             oversquashed_block = oversquashed_nodes_tensor[i:i + block_size]
#             matched_block = matched_nodes_tensor[j:j + block_size]

#             # 计算当前块的笛卡尔积
#             node_pairs_block = torch.cartesian_prod(oversquashed_block, matched_block)

#             # 提取节点对的索引
#             nodei_indices = node_pairs_block[:, 0]
#             nodej_indices = node_pairs_block[:, 1]

#             # 计算相似度
#             similarities = cacDistance(ranks[nodei_indices], ranks[nodej_indices])

#             # 将当前块写入磁盘
#             block_matrix = coo_matrix((similarities.cpu().numpy(),
#                                        (nodei_indices.cpu().numpy(), nodej_indices.cpu().numpy())),
#                                       shape=simMatrix_shape)
#             save_npz(f"{path}/block_{i}_{j}.npz", block_matrix)
#             print(f"Saved block ({i}, {j}) to disk")