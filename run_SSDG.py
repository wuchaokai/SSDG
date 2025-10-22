import argparse
import copy
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
#from PreProcss_cora import PreProcessCora as cora
from sklearn.metrics import f1_score,precision_score,recall_score
import csv
import random
from GCNmodel import GCN
#from utils import generateFOSRGraph,generateBORFGraph,generateSDRFGraph
import scipy.sparse as sp
from SDRF import SDRF
import utils
import os
import dgl
from LPGNN_ours import Ours
#from LPGNN_ours_dense import Ours as Ours_dense
#from PreProcess_new_data import Process_new_data as new_data
#from PreProcess_airports import Process_airports as airports
from generateSimFromLargeData import generateStructWeight
#from utils import generateGraph2
# from GIN import GIN
# from GAT import GAT
# from GATv2 import GATv2
# from DaGNN import DAGNN
# from APPNP import APPNP
# from GraphSAGE import SAGE
# from GCNII import GCNII
import numpy as np
import warnings
import pandas as pd
import time
import math
from process_large_dataset import load_penn94,load_snap_patents_mat
import pynvml

def get_real_gpu_memory_usage(gpu_index=0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    
    used_mb = mem_info.used / 1024**2
    total_mb = mem_info.total / 1024**2
    return used_mb/total_mb

def save_metrics_to_csv(file_path, mean_acc, mean_f1, description):

    header = ['mean_acc', 'mean_f1', 'description']
    data = [mean_acc, mean_f1, description]


    try:
        with open(file_path, 'x', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerow(data)
    except FileExistsError:
        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)
# def print_info(dataset,label,des):
#     df = pd.read_csv('data/{}/class{}_results.csv'.format(dataset,label))[-10:]
#     
#     mean_acc = df['acc'].mean()
#     std_acc = np.std(df['acc'], ddof=1)
#     mean_f1 = df['macro_f1'].mean()
#     std_f1 = np.std(df['macro_f1'], ddof=1)
#
#     # 打印结果
#     print(f"acc: {mean_acc:.4f}±{std_acc:.4f}")
#     print(f"macro_f1: mean = {mean_f1:.4f}±{std_f1:.4f}")
#
#     mean_acc = f"{mean_acc:.4f}±{std_acc:.4f}"
#     mean_f1 = f"{mean_f1:.4f}±{std_f1:.4f}"
#     save_metrics_to_csv('data/{}/class{}_mean_result.csv'.format(dataset,label),mean_acc,mean_f1,des)
def print_info(dataset,label,des):
    df = pd.read_csv('data/{}/class{}_results.csv'.format(dataset,label))[-10:]
    
    mean_acc = df['acc'].mean()
    std_acc = np.std(df['acc'], ddof=1)
    mean_f1 = df['macro_f1'].mean()
    std_f1 = np.std(df['macro_f1'], ddof=1)

    
    print(f"acc: {mean_acc:.4f}±{std_acc:.4f}")
    print(f"macro_f1: mean = {mean_f1:.4f}±{std_f1:.4f}")

    mean_acc = f"{mean_acc:.4f}±{std_acc:.4f}"
    mean_f1 = f"{mean_f1:.4f}±{std_f1:.4f}"
    save_metrics_to_csv('data/{}/class{}_mean_result.csv'.format(dataset,label),mean_acc,mean_f1,des)
warnings.filterwarnings('ignore')

def print_info2(dataset,des):
    df = pd.read_csv('data/{}/results.csv'.format(dataset))[-10:]
    mean_acc = df['acc'].mean()
    mean_f1 = df['macro_f1'].mean()

    save_metrics_to_csv('data/{}/mean_result.csv'.format(dataset),mean_acc,mean_f1,des)
warnings.filterwarnings('ignore')
def evaluate(features,adj,structMatrix,featureMatrix, labels, mask, model,graph):
    model.eval()
    with torch.no_grad():
        start_time=time.time()
        if args.model=='SSDG':
            logits,_ = model(features, adj, structMatrix,featureMatrix)
        elif args.model in ['GCN','GCNII']:
            logits= model(features, adj, structMatrix)
        elif args.model in ['GIN','GAT','GATv2','SAGE','DaGNN','APPNP',]:
            logits = model(features, graph)
        # end_time = time.time()
        # print(f"torch.mm time: {(end_time - start_time):.6f} seconds")
        # new_data = {
        #     "time": end_time - start_time,
        # }
        # save_to_csv('data/time.csv'.format(args.dataset), new_data)
        # exit()
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        #print(correct)
        micro = f1_score(labels.cpu(), indices.cpu(), average='micro')
        macro = f1_score(labels.cpu(), indices.cpu(), average='macro')
        return correct.item() * 1.00 / len(labels),micro,macro


def train(adj,structMatrix,featureMatrix,features, labels, masks, model,graph,args, loss_rate=0.5):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=5e-4)
    best_acc=0
    bestmodel=model
    model.train()
    # training loop
    sum_time=0
    start_time=time.time()
    record_time=0
    for epoch in range(100000000):
        # Training logic here
        epoch_start_time = time.time()
        if args.model=='SSDG':
            logits,prelabels = model(features,adj,structMatrix,featureMatrix)
            loss = loss_fcn(logits[train_mask], labels[train_mask]) + loss_rate * loss_fcn(prelabels[train_mask],
                                                                                           labels[train_mask])
        elif args.model in ['GCN','GCNII']:
            logits= model(features, adj, structMatrix)
            loss = loss_fcn(logits[train_mask], labels[train_mask])
        elif args.model in ['GIN','GAT','GATv2','SAGE','DaGNN','APPNP']:
            logits = model(features, graph)
            loss = loss_fcn(logits[train_mask], labels[train_mask])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_end_time = time.time()
        sum_time += (epoch_end_time - epoch_start_time)
    # epoch_end_time = time.time()
    # print(f"took {(epoch_end_time - start_time):.4f} seconds")
    # new_data = {
    #     'model': args.model,
    #     "time": epoch_end_time - start_time,
    # }
    # save_to_csv('data/{}/time.csv'.format(args.dataset), new_data)
    # exit()
        
        acc,micro,macro = evaluate(features, adj,structMatrix,featureMatrix,labels, val_mask, model,graph)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )
        if best_acc<acc:
            best_acc=acc
            best_epoch=epoch
            # best_time=sum_time
            # bestmodel=copy.deepcopy(model)
            # if sum_time>record_time+1:
            #     record_time=sum_time
            #     new_data = {
            #         'model': args.model,
            #         "time": sum_time,
            #         'acc': best_acc,
            #     }
            #     save_to_csv('data/{}/time_record.csv'.format(args.dataset), new_data)
        if epoch - best_epoch >= 100:
            print("Early stopping at epoch {:05d}".format(epoch))
            break
        
    # new_data = {
    #     "best_epoch": best_epoch,
    #     "best_time": best_time,
    #     "best_acc": best_acc,
    #     'acc_per_time': best_acc/best_time,
    #     'model': args.model
    # }
    # save_to_csv('data/{}/best_time.csv'.format(args.dataset), new_data)
    #exit()
    return bestmodel

        # if acc > 0.63:
        #     model_name='GCN'
        #     utils.visual(model(features, adj, WeightAdj)[train_mask], labels[train_mask], model_name, dataset)
def choosedataset(name):
    if name in ['penn94']:
        return load_penn94()
    elif name in ['cornell','wisconsin','chameleon','texas','squirrel']:
        return new_data().LoadData(name)
    elif name in ['brazil_airports','eu_airports','usa_airports']:
        return airports().LoadData(name)
    elif name in ['snap_patents']:
        return load_snap_patents_mat()
    elif name in ['citeseer']:
        return cora().LoadData(name)
    
def label2onehot(labels):
    num_classes = max(labels) + 1
    one_hot = torch.zeros(len(labels), num_classes)
    for i, label in enumerate(labels):
        one_hot[i, label] = 1
    return one_hot
def get_class_mask(labels,num_classes,test_mask):
    tensor_list=[]
    for i in range(num_classes):
        tensor = torch.where(labels == i, torch.tensor(1), torch.tensor(0)) & test_mask
        tensor=tensor.byte()
        tensor_list.append(tensor)
    return tensor_list
def get_class_num(labels,num_classes):
    tensor_list = []
    for i in range(num_classes):
        tensor = torch.where(labels == i, torch.tensor(1), torch.tensor(0))
        print('class{}:{}'.format(i,torch.sum(tensor)))

def save_to_csv(file_name, new_data):
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        new_row = pd.DataFrame([new_data])
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = pd.DataFrame([new_data])

    df.to_csv(file_name, index=False)
    
def count_sparse_nonzero(sparse_tensor):
    if hasattr(sparse_tensor, '_nnz'):
        return sparse_tensor._nnz()
    elif hasattr(sparse_tensor, 'nnz'):
        return sparse_tensor.nnz
    else:
        return torch.sum(sparse_tensor != 0).item()
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def main(args):

    seed=args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    dataset=args.dataset
    model=args.model
    is_res=True

    acus=[]
    macros=[]
    rec = []
    rec2=[]
    sum=0
    if args.dataset in ['snap_patents']:
        adj, nx_graph,features, labels, train_mask, val_mask, test_mask, num_classes,graph= choosedataset(dataset)
    else:
        adj, features, labels, train_mask, val_mask, test_mask, num_classes,graph= choosedataset(dataset)
        
    
    #utils.generateSpilt(dataset,train_mask, val_mask, test_mask,args.seed%10)
    if dataset not in ['snap_patents','citeseer']:
        train_mask,val_mask,test_mask=utils.readSpilt(dataset,args.index)
    print('generate struct weight')
    #tensor_list=get_class_mask(labels,num_classes,test_mask)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    if args.model=='SSDG':
        if args.dataset in ['snap_patents']:
            structMatrix = generateStructWeight(nx_graph,'data/' + dataset + '/',select_rate=args.select_rate,percentile=args.percentile,highmemoryutilization=args.highmemoryutilization)
            featureMatrix = sp.eye(adj.shape[0])
            print(f"structMatrix has {count_sparse_nonzero(structMatrix)} non-zero elements.")
            print('generate struct weight done')
            print('constructing structMatrix start')
        else:
            featureMatrix, structMatrix = generateGraph2(args.hop, 'data/' + dataset + '/',chooserate=args.select_rate,threshold=args.percentile,aggregate=args.aggregate,threshold2=args.threshold)
            
            print('generate struct weight done')
            print('constructing WeightAdj start')
            structMatrix=sp.coo_matrix(structMatrix)
            structMatrix=utils.normalize(structMatrix)
            structMatrix=utils.sparse_mx_to_torch_sparse_tensor(structMatrix)
        featureMatrix=sp.coo_matrix(featureMatrix)
        featureMatrix=utils.normalize(featureMatrix)
        featureMatrix=utils.sparse_mx_to_torch_sparse_tensor(featureMatrix)
        structMatrix = structMatrix.to(device)
        featureMatrix = featureMatrix.to(device)
        print('constructing structMatrix done')
    else:
        structMatrix = torch.tensor(1)
        featureMatrix = torch.tensor(1)
    if graph is not None and args.model in ['GIN','GAT','GATv2','SAGE','DaGNN','APPNP']:
        graph = graph.to(device)
    
    rewiring_type=args.rewiring_type
    if rewiring_type == 'sdrf':
        print('rewiring type is sdrf')
        data = utils.genGeoData(features, adj)
        adj = generateSDRFGraph(data, adj.shape[0])
    elif rewiring_type == 'fosr':
        print('rewiring type is fosr')
        data = utils.genGeoData(features, adj)
        adj = utils.generateFOSRGraph(data, adj.shape[0])
    elif rewiring_type == 'borf':
        print('rewiring type is borf')
        data = utils.genGeoData(features, adj)
        adj = utils.generateBORFGraph(data, adj.shape[0], dataset)
        
    if dataset not in ['snap_patents']:
        adj=normalize(adj+sp.eye(adj.shape[0]))
        adj=sparse_mx_to_torch_sparse_tensor(adj)

    features=features.to(device)
    labels=labels.to(device)
    masks = train_mask.to(device), val_mask.to(device), test_mask.to(device)
    # create GCN model
    in_size = features.shape[1]
    out_size = num_classes
    adj=adj.to(device)
    #fullgraph=fullgraph.int().to(device)
    structMatrix = torch.tensor(structMatrix, dtype=torch.float32).to(device)
    
    
    #model =Ours(in_size, 16 , out_size,0.5,is_res).to(device)
    if args.model=='GCN':
        model=GCN(in_size,16,out_size,0.5).to(device)
    elif args.model=='SSDG':
        if args.dataset in ['snap_patents']:
            model =Ours(in_size,16, out_size,is_res,args).to(device)
        else:
            model =Ours_dense(in_size,16, out_size,is_res,args).to(device)
    elif args.model=='GIN':
        model=GIN(in_size,16,out_size).to(device)
    elif args.model=='GAT':
        model=GAT(in_size,16,out_size,[8,1]).to(device)
    elif args.model=='GATv2':
        model=GATv2(in_size,16,out_size,[8,1]).to(device)
    elif args.model=='SAGE':
        model=SAGE(in_size,16,out_size).to(device)
    elif args.model=='APPNP':
        model=APPNP(in_size,[16],out_size).to(device)
    elif args.model=='DaGNN':
        model=DAGNN(in_size,16,out_size).to(device)
    elif args.model=='GCNII':
        model=GCNII(in_size,16,out_size,0.5).to(device)
    #model=SSDG_W(in_size,8, out_size,is_res).to(device)
    #model=LPGNN(in_size,8, out_size,N).to(device)
    # model training
    #print("Training...")
    
    train(adj,structMatrix,featureMatrix,features, labels, masks, model,graph,args,loss_rate=args.beta)

    # test the model
    print("Testing...")
    acc,micro,macro = evaluate(features,adj,structMatrix,featureMatrix, labels, masks[2], model,graph)


    # if acc > 0:
    #     model_name='gcn_base'
    #     utils.visual(model(features, adj, WeightAdj), labels, model_name, dataset)

    sum=acc+sum
    rec.append(acc)
    rec2.append(macro)
    print("Test accuracy {:.4f},micro {:.4f},macro {:.4f}".format(acc,micro,macro))

    # for i,tensor in enumerate(tensor_list):
    #     if torch.sum(tensor)==0: continue
    #     acc, micro, macro = evaluate(features, adj, WeightAdj, FeatureAdj, labels, tensor, model)
    #     print('class {},acc:{},macro:{}'.format(i, acc, macro))
    #     new_data = {
    #         "acc": acc,
    #         "macro_f1": macro,
    #     }
    #     save_to_csv('data/{}/class{}_results.csv'.format(args.dataset,i), new_data)

    new_data = {
        "acc": acc,
        "macro_f1": macro,
    }
    save_to_csv('data/{}/{}_results.csv'.format(args.dataset,args.model), new_data)

    # average=sum / num
    # average2=np.sum(rec2)/len(rec2)
    # rec2=[]
    # print('accuracy: '+str(average))
    # print('macro:'+str(average2))
    # # for i in range(num_classes):
    # #     print_info(dataset,i,des='model:{} iscompensate:{}'.format(args.model,str(args.compensate)))
    # macros.append(average2)
    # #print(rec)
    # acus.append(average)
    # print_info2(dataset, des='model:{}'.format(args.model))
    # print(acus)
    # print('accuracy:'+str(np.sum(acus)/5))
    # print(macros)
    # filename = 'data/'+dataset+'/indicators.csv'
    #
    # with open(filename, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #
    #     # 写入标题
    #     writer.writerow(['Accuracy', 'Macro-F1'])
    #
    #     # 写入数据
    #     for i1, i2 in zip(acus, macros):
    #         writer.writerow([i1, i2])
    # print('average macros is:')
    # print(np.sum(macros) / len(macros))


if __name__ == "__main__":
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()

    # 添加参数
    parser.add_argument('--dataset', type=str,help='dataset',default='snap_patents')
    parser.add_argument('--numrun',type=int,default=1)
    parser.add_argument('--model',type=str,default='SSDG')
    parser.add_argument('--compensate',type=bool,default=True)
    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--select_rate',type=float,default=0.5)
    parser.add_argument('--percentile',type=float,default=99.9)
    parser.add_argument('--seed',type=int,default=2025)
    parser.add_argument('--hop',type=int,default=2)
    parser.add_argument('--rewiring_type', type=str, help='rewiring_type', default='base')
    parser.add_argument('--aggregate', type=str, help='aggregate', default='weight')
    parser.add_argument('--without', type=str, help='without', default='base')
    parser.add_argument('--threshold', type=float, help='threshold', default=1)
    parser.add_argument('--beta', type=float, help='beta', default=1)
    parser.add_argument('--gpu', type=int, help='gpu', default=0)
    parser.add_argument('--highmemoryutilization', type=bool, help='highmemoryutilization', default=False)
    args = parser.parse_args()
    print(args)
    
    main(args)