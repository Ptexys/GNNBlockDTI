import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import pickle
import os
import json
import time
import random
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
from datetime import datetime

from Trainer import *
from models import *

class mydataset(Dataset):
    def __init__(self, data, drug_graph, target_embedding, target_graph):  #drug_data 图数据or序列数据
        super(mydataset, self).__init__()
        D, T, Tg, Y = [], [], [], []
        for ent in data:  # ent:[drug_id, prot_id, Y]
            try:
                D.append(drug_data[str(ent[0])])
                T.append(target_embedding[ent[1]])
                Tg.append(target_graph[ent[1]])
                Y.append(int(ent[2]))
            except:
                pass
        self.D = D
        self.T = T
        self.Tg = Tg
        self.Y = Y
        
    def __getitem__(self, idx):
        return self.D[idx], self.T[idx], self.Tg[idx], self.Y[idx]
    
    def __len__(self):
        return len(self.Y)
      
def collate_fn(batch):
    D_graphs, T0, T_graphs, Y0 = map(list, zip(*batch))
    D_G = dgl.batch(D_graphs)
    T = torch.cat([T0[idx].unsqueeze(0) for idx in range(len(T0))], 0)
    T_G = dgl.batch(T_graphs)
    Y = torch.tensor(Y0)
    
    return D_G, T, T_G, Y

def main_cross_validation(Task, args, K=5):  
    device = torch.device("cuda:0")
    
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    path = os.path.join("./dataset", Task)
    with open(path + "/drug_graph.pkl", 'rb') as f:
        drug_graph = pickle.load(f)
    with open(path + "/target_embedding.pkl", 'rb') as f:
        target_embedding = pickle.load(f)
    with open(path + "/target_graph.pkl", 'rb') as f:
        target_graph = pickle.load(f)

    epochs = args.epochs
    batch_size = args.batch_size
    lr =  args.lr
    dropout_d = args.dropout_d 
    dropout_DT = args.dropout_DT 
    n_layer = args.n_layer
    n_gnn = args.n_gnn
    
    hidden_size = 128
    embed_dim = 128
    hid_size = 384
    
    results = []
    metrs = []
    Name_net = "GNNBlockDTI"
    for n in range(K):
        f_path = f"{path}/random_CV5/data_CV{n}.pkl"
        with open(f_path, 'rb') as f:
            data = pickle.load(f)
        train_data, valid_data, test_data = data
        '''
        with open(path + "full_data.pkl", "rb") as f:
            data = pickle.load(f)
        train_data0, test_data = train_test_split(data, test_size=0.2)
        train_data, valid_data = train_test_split(train_data0, test_size=1/8)
        '''
        Name = Name_net + '_' + Task + '_CV' + str(n)
        
        # data
        dataset_train = mydataset(train_data, drug_graph, prot_seq, prot_data)
        train_iter = DataLoader(dataset_train, batch_size, shuffle=True, collate_fn=collate_fn)
        dataset_test = mydataset(test_data, drug_graph, prot_seq, prot_data)
        test_iter = DataLoader(dataset_test, batch_size, shuffle=False, collate_fn=collate_fn)
        dataset_valid = mydataset(valid_data, drug_graph, prot_seq, prot_data)
        valid_iter = DataLoader(dataset_valid, batch_size, shuffle=False, collate_fn=collate_fn)

        print("Data total: ", len(dataset_train)+len(dataset_test)+len(dataset_valid))
        print("Data train: ", len(dataset_train))
        print("Data test: ", len(dataset_test))
        print("Data valid: ", len(dataset_valid))

        # model
        net_T = MultiscaleCNN(inputs=30, embed_dim=embed_dim, nums_conv=[32, 64, 96], ksize=[5,7,13], dropout=0.2)
        net_T1 = WGCN(input_size=30, hidden_size=hidden_size, dropout=0.2)
        net_D = GNNBlocks(input_size=64, hidden_size=96, n_layer=n_layer, n_gnn=n_gnn, dropout=dropout_d)
        model = GNNBlockDTI(net_D, net_T, net_T1, Drug_len=96*2, Target_len=192, Target_len1=hidden_size*2,
                            hid_size=hid_size, dropout=dropout_DT)

        # training
        loss = nn.CrossEntropyLoss(weight=None)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        compare = "max"
        train_L, test_L, test_metric, best_metric = train(model, epochs, train_iter, valid_iter, test_iter, 
                                                    loss, optimizer, compare, 
                                                    device, Name, )
        results.append((train_L, test_L, test_metric))
        metrs.append(best_metric)
        
        torch.cuda.empty_cache()

        print("Best result: ", best_metric)
        if not os.path.exists('results'):
            os.makedirs('results')
        '''
        with open("results/" + Name + ".pkl", 'wb') as f:
            pickle.dump((train_L, test_L, test_metric, best_metric), f)
        '''

    with open("results/" + Name_net + '_' + Task + ".pkl", 'wb') as f:
        pickle.dump(results, f)
    
    return metrs

class Arguments():
    def __init__(self):
        self.epochs = 100
        
        self.batch_size = 64
        self.lr = 0.0005
        self.dropout_d = 0.2
        self.dropout_DT = 0.2
        
        self.n_layer = 5
        self.n_gnn = 2
        
#--------------------------------------------------------------------------------
if __name__ == "__main__":
    args = Arguments()

    tasks = ["BIOSNAP", ]  #"BIOSNAP", "DrugBank", "Human", 
    rs = []
    for Task in tasks:
        metrs = main_cross_validation(Task, args)
        df = pd.DataFrame(metrs, columns=["AUROC", "AUPR", "ACC", "PR", "RE", "F1"])
        now = datetime.now()
        today = str(now).split('.')[0]
        myday = today.replace('-', '_').replace(' ', '_').replace(':', '_')
        df.to_csv(f"results/GNNBlockDTI_{Task}_{myday}.csv")

    