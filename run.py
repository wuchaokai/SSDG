import subprocess
import sys
import pandas as pd
import numpy as np
import torch
import csv

def save_metrics_to_csv(file_path, acc,f1, description):
    """
    保存五个字符串和描述信息到 CSV 文件中，支持重复写入。

    参数:
    file_path (str): CSV 文件路径
    mean_auc_test (str): mean_auc_test 字符串
    mean_ap (str): mean_ap 字符串
    mean_mrr (str): mean_mrr 字符串
    mean_recall (str): mean_recall 字符串
    mean_f1 (str): mean_f1 字符串
    description (str): 描述信息
    """
    header = ['acc','f1', 'description']
    data = [acc,f1, description]

    # 检查文件是否存在以决定是否写入表头
    try:
        with open(file_path, 'x', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerow(data)
    except FileExistsError:
        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)
def save_metrics_to_csv_linkprediction(file_path, mean_auc_test,mean_ap,mean_mrr,mean_recall, mean_f1, description):
    """
    保存五个字符串和描述信息到 CSV 文件中，支持重复写入。

    参数:
    file_path (str): CSV 文件路径
    mean_auc_test (str): mean_auc_test 字符串
    mean_ap (str): mean_ap 字符串
    mean_mrr (str): mean_mrr 字符串
    mean_recall (str): mean_recall 字符串
    mean_f1 (str): mean_f1 字符串
    description (str): 描述信息
    """
    header = ['mean_auc_test', 'mean_ap', 'mean_mrr', 'mean_recall', 'mean_f1', 'description']
    data = [mean_auc_test, mean_ap, mean_mrr, mean_recall, mean_f1, description]

    # 检查文件是否存在以决定是否写入表头
    try:
        with open(file_path, 'x', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerow(data)
    except FileExistsError:
        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)
def load_column_from_csv(file_name, column_name):
    df = pd.read_csv(file_name)
    return df[column_name].tolist()
def print_info(dataset,des,model,filename):
    df = pd.read_csv('data/{}/{}_results.csv'.format(dataset,model))[-10:]
    # 计算每一列的均值和标准差
    mean_acc = df['acc'].mean()
    std_acc = np.std(df['acc'], ddof=1)
    mean_f1 = df['macro_f1'].mean()
    std_f1 = np.std(df['macro_f1'], ddof=1)

    # 打印结果
    print(f"acc: {mean_acc:.4f}±{std_acc:.4f}")
    print(f"macro_f1: mean = {mean_f1:.4f}±{std_f1:.4f}")

    mean_acc = f"{mean_acc:.4f}±{std_acc:.4f}"
    mean_f1 = f"{mean_f1:.4f}±{std_f1:.4f}"
    save_metrics_to_csv('results/{}_{}.csv'.format(dataset,filename),mean_acc,mean_f1,des)
def print_info_linkprediction(dataset,des,model,filename):
    df = pd.read_csv('data/{}/{}_results.csv'.format(dataset,model))[-10:]
    # 计算每一列的均值和标准差
    mean_auc = df['roc_auc'].mean()
    std_auc = np.std(df['roc_auc'], ddof=1)
    mean_ap = df['ap'].mean()
    std_ap = np.std(df['ap'], ddof=1)
    mean_mrr= df['mrr'].mean()
    std_mrr = np.std(df['mrr'], ddof=1)
    mean_recall = df['recall'].mean()
    std_recall = np.std(df['recall'], ddof=1)
    mean_f1 = df['f1'].mean()
    std_f1 = np.std(df['f1'], ddof=1)

    # 打印结果
    print(f"auc: {mean_auc:.4f}±{std_auc:.4f}")
    print(f"ap: mean = {mean_ap:.4f}±{std_ap:.4f}")
    print(f"mrr: mean = {mean_mrr:.4f}±{std_mrr:.4f}")
    print(f"recall: mean = {mean_recall:.4f}±{std_recall:.4f}")
    print(f"f1: mean = {mean_f1:.4f}±{std_f1:.4f}")

    mean_auc = f"{mean_auc:.4f}±{std_auc:.4f}"
    mean_ap = f"{mean_ap:.4f}±{std_ap:.4f}"
    mean_mrr = f"{mean_mrr:.4f}±{std_mrr:.4f}"
    mean_recall = f"{mean_recall:.4f}±{std_recall:.4f}"
    mean_f1 = f"{mean_f1:.4f}±{std_f1:.4f}"
    save_metrics_to_csv_linkprediction('results/{}_{}.csv'.format(dataset,filename),mean_auc,mean_ap,mean_mrr,mean_recall,mean_f1,des)
def run_link_prediction(seed,model_name,wo,compensate,dataset,select_rate,match_rate):
    #subprocess.run(['python', 'link_prediction.py'])
    cmd=[sys.executable, 'link_prediction.py',
                    '--seed', str(seed),
                    '--model', str(model_name),
                    '--without',str(wo),
                    '--dataset',str(dataset),
                    '--select_rate',str(select_rate),
                    '--match_rate',str(match_rate)
                    ]
    if compensate==True:
        cmd.append('--compensate')
    subprocess.run(cmd)


def run_model(dataset,model,index,select_rate,percentile,hop,seed,rewiring_type,aggregate,without,threshold,beta=0.5):
    if model =='PD_GAT':
        cmd=[sys.executable, 'run_PD_GAT.py',
         '--model', str(model),
         '--dataset', str(dataset),
         '--index', str(index),
         '--select_rate', str(select_rate),
         '--percentile', str(percentile),
         '--hop', str(hop),
         '--seed', str(seed),
         '--rewiring_type', str(rewiring_type)
         ]
    else:
        cmd=[sys.executable, 'run_SSDG.py',
            '--model', str(model),
            '--dataset', str(dataset),
            '--index', str(index),
            '--select_rate', str(select_rate),
            '--percentile', str(percentile),
            '--hop', str(hop),
            '--seed', str(seed),
            '--rewiring_type', str(rewiring_type),
            '--aggregate', str(aggregate),
            '--without', str(without),
            '--threshold', str(threshold),
            '--beta', str(beta)
            ]
    subprocess.run(cmd)

def run_model_linkprediction(dataset,model,seed,beta,hop):
    if model =='PD_GAT':
        cmd=[sys.executable, 'run_PD_GAT.py',
         '--model', str(model),
         '--dataset', str(dataset),
         '--index', str(index),
         '--select_rate', str(select_rate),
         '--percentile', str(percentile),
         '--hop', str(hop),
         '--seed', str(seed),
         '--rewiring_type', str(rewiring_type)
         ]
    else:
        cmd=[sys.executable, 'link_prediction.py',
            '--model', str(model),
            '--dataset', str(dataset),
            '--seed', str(seed),
            '--beta', str(beta),
            '--hop', str(hop),
            '--gpu',str(7)
            ]
    subprocess.run(cmd)
if __name__=='__main__':

    
        # for model in ['GCN','SAGE','GAT','GATv2','DaGNN','GCNII','APPNP','SSDG']:
        #     for dataset in ['brazil_airports','eu_airports','wisconsin','texas','chameleon','squirrel']:
        #         if dataset in ['chameleon','squirrel']:
        #             hop=1
        #         else:
        #             hop=2
        #         for select_rate in [1]:
        #             for percentile in [0]:
        #                 for seed in range(10):
        #                     index=0
        #                     run_model(dataset,model,index,select_rate,percentile,hop,seed)
        #                     des='model:{} dataset:{} index:{} select_rate:{} percentile:{} hop:{} seed:{}'.format(model,dataset,index,select_rate,percentile,hop,seed)
        #                 print_info(dataset,des,model,'baseline')
    # for model in ['SSDG']:
    #     for dataset in ['wisconsin','texas','brazil_airports','eu_airports']:
    #         for rewiring_type in ['base']:
    #             if dataset in ['chameleon','squirrel']:
    #                 hop=1
    #             else:
    #                 hop=2
    #             for aggregate in ['structure']:
    #                 for without in ['base']:
    #                     for select_rate in [0.1,0.2,0.3,0.5,0.6,0.7,0.8]:
    #                         for percentile in [0]:
    #                             for threshold in [1]:
    #                                 if dataset=='snap_patents':
    #                                     select_rate=0.5
    #                                     percentile=99.9
    #                                 for index in range(10):
    #                                     seed=2025
    #                                     if dataset in ['wisconsin','snap_patents','citeseer']:
    #                                         seed=index
    #                                     run_model(dataset,model,index,select_rate,percentile,hop,seed,rewiring_type,aggregate,without,threshold)
    #                                     des='model:{} dataset:{} index:{} select_rate:{} percentile:{} hop:{} seed:{} rewiring_type:{} aggregate:{} without:{} threshold:{}'.format(model,dataset,index,select_rate,percentile,hop,seed,rewiring_type,aggregate,without,threshold)
    #                               print_info(dataset,des,model,'select_rate')

    # for model in ['SSDG']:
    #     for dataset in ['wisconsin','texas','brazil_airports','eu_airports']:
    #         for rewiring_type in ['base']:
    #             if dataset in ['chameleon','squirrel']:
    #                 hop=1
    #             else:
    #                 hop=2
    #             for aggregate in ['structure']:
    #                 for without in ['base']:
    #                     for select_rate in [1]:
    #                         for percentile in [0]:
    #                             for threshold in [0.1,0.2,0.3,0.5,0.6,0.7,0.8]:
    #                                 if dataset=='snap_patents':
    #                                     select_rate=0.5
    #                                     percentile=99.9
    #                                 for index in range(10):
    #                                     seed=2025
    #                                     if dataset in ['wisconsin','snap_patents','citeseer']:
    #                                         seed=index
    #                                     run_model(dataset,model,index,select_rate,percentile,hop,seed,rewiring_type,aggregate,without,threshold)
    #                                     des='model:{} dataset:{} index:{} select_rate:{} percentile:{} hop:{} seed:{} rewiring_type:{} aggregate:{} without:{} threshold:{}'.format(model,dataset,index,select_rate,percentile,hop,seed,rewiring_type,aggregate,without,threshold)
    #                                 print_info(dataset,des,model,'threshold')
    # for model in ['SSDG']:
    #     for dataset in ['wisconsin','texas','brazil_airports','eu_airports']:
    #         for rewiring_type in ['base']:
    #             if dataset in ['chameleon','squirrel']:
    #                 hop=1
    #             else:
    #                 hop=2
    #             for aggregate in ['structure']:
    #                 for without in ['base']:
    #                     for select_rate in [1]:
    #                         for percentile in [0]:
    #                             for threshold in [1]:
    #                                 for beta in [0,0.1,0.3,0.5,0.7,0.8,1]:
    #                                     if dataset=='snap_patents':
    #                                         select_rate=0.5
    #                                         percentile=99.9
    #                                     for index in range(10):
    #                                         seed=2025
    #                                         if dataset in ['wisconsin','snap_patents','citeseer']:
    #                                             seed=index
    #                                         run_model(dataset,model,index,select_rate,percentile,hop,seed,rewiring_type,aggregate,without,threshold,beta)
    #                                         des='model:{} dataset:{} index:{} select_rate:{} percentile:{} hop:{} seed:{} rewiring_type:{} aggregate:{} without:{} threshold:{} beta:{}'.format(model,dataset,index,select_rate,percentile,hop,seed,rewiring_type,aggregate,without,threshold,beta)
    #                                     print_info(dataset,des,model,'beta')
    for dataset in ['eu_airports','chameleon','squirrel']:
        for model in ['SAGE','GAT','GATv2','DaGNN','GCNII','APPNP','SSDG']:
            for beta in [0]:
                if dataset in ['chameleon','squirrel']:
                    hop=1
                else:
                    hop=2
                for seed in range(10):
                        run_model_linkprediction(dataset,model,seed,beta,hop)
                        des='model:{} dataset:{} seed:{} beta:{} hop:{}'.format(model,dataset,seed,beta,hop)
                print_info_linkprediction(dataset,des,model,'linkprediction_only')