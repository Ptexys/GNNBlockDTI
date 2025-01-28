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

from Trainer import *
from models import *

def test(Task, args, k):  
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
    
    f_path = f"{path}/random_CV5/data_CV{n}.pkl"
    with open(f_path, 'rb') as f:
        data = pickle.load(f)
    train_data, valid_data, test_data = data
    # data
    dataset_test = mydataset(test_data, drug_graph, prot_seq, prot_data)
    test_iter = DataLoader(dataset_test, batch_size, shuffle=False, collate_fn=collate_fn)

    print("Data test: ", len(dataset_test))

    # model
    net_T = MultiscaleCNN(inputs=30, embed_dim=embed_dim, nums_conv=[32, 64, 96], ksize=[5,7,13], dropout=0.2)
    net_T1 = WGCN(input_size=30, hidden_size=hidden_size, dropout=0.2)
    net_D = GNNBlocks(input_size=64, hidden_size=96, n_layer=n_layer, n_gnn=n_gnn, dropout=dropout_d)
    model = GNNBlockDTI(net_D, net_T, net_T1, Drug_len=96*2, Target_len=192, Target_len1=hidden_size*2,
                            hid_size=hid_size, dropout=dropout_DT)
    Name_net = "GNNBlockDTI"
    Name = Name_net + '_' + Task + '_CV' + str(n)
    model_path= "models/" + Name + ".pth"
    model.load_state_dict(torch.load(model_path))

    # test
    loss = nn.CrossEntropyLoss(weight=None)
    test_loss, test_metric = test(model, test_iter, loss, device)
    print(f"test result of {Task} CV{k}: ")
    print("AUROC AUPR ACC PR RE F1")
    print(best_metric)
    return test_metric

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
    
    Task = "BIOSNAP"  #"BIOSNAP", "DrugBank", "Human", 
    metrs = test(Task, args, 0)