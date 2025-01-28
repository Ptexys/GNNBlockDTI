import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd
import dgl
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm, MaxPooling, AvgPooling, SumPooling


# Target
class WGCN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super(WGCN, self).__init__()
        self.norm = EdgeWeightNorm(norm='both')

        self.gcn1 = GraphConv(input_size, hidden_size, 
                                  norm='both', activation=nn.ReLU(), allow_zero_in_degree=True)
        self.gcn2 = GraphConv(hidden_size, hidden_size, 
                                  norm='both', activation=nn.ReLU(), allow_zero_in_degree=True)
        self.gcn3 = GraphConv(hidden_size, hidden_size*2, 
                                  norm='both', activation=nn.ReLU(), allow_zero_in_degree=True)
        self.LN1 = nn.LayerNorm(hidden_size)
        self.LN2 = nn.LayerNorm(hidden_size)
        self.LN3 = nn.LayerNorm(hidden_size*2)
        self.scale = torch.sqrt(torch.FloatTensor([0.5]))
        self.glu = nn.GLU(dim=-1)
        self.do = nn.Dropout(dropout)
        self.maxpool = MaxPooling()
        
    def forward(self, g, feats, edge_weight):
        norm_edge_weight = self.norm(g, edge_weight)
        h1 = self.gcn1(g, feats, edge_weight=norm_edge_weight)
        h1 = self.LN1(h1)
        h2 = self.gcn2(g, h1, edge_weight=norm_edge_weight)
        h2 = self.LN2(h2)
        h3 = self.gcn3(g, h2, edge_weight=norm_edge_weight)
        h3 = self.LN3(h3)
        
        return h3
    
class MultiscaleCNN(nn.Module):
    def __init__(self, inputs, embed_dim, nums_conv, ksize, dropout, **kwargs):
        super(MultiscaleCNN, self).__init__(**kwargs)
        self.embed = nn.Linear(inputs, embed_dim)
        self.cnn1 = nn.Conv1d(embed_dim, nums_conv[0], kernel_size=ksize[0], padding="same")
        self.cnn2 = nn.Conv1d(nums_conv[0], nums_conv[1], kernel_size=ksize[1], padding="same")
        self.cnn3 = nn.Conv1d(nums_conv[1], nums_conv[2], kernel_size=ksize[2], padding="same")
        self.fc1 = nn.Linear(nums_conv[0], nums_conv[0])
        self.fc2 = nn.Linear(nums_conv[1], nums_conv[1])
        self.fc3 = nn.Linear(nums_conv[2], nums_conv[2])
        self.relu = nn.ReLU()
    
    def forward(self, X):
        X = self.embed(X)
        X = X.permute(0,2,1)
        X1 = self.cnn1(X)
        X1 = self.relu(X1)
        X2 = self.cnn2(X1)
        X2 = self.relu(X2)
        X3 = self.cnn3(X2)
        X3 = self.relu(X3)
        
        X1 = self.fc1(X1.permute(0,2,1))
        X2 = self.fc2(X2.permute(0,2,1))
        X3 = self.fc3(X3.permute(0,2,1))
        X_co = torch.cat((X1, X2, X3), dim=-1)
    
        return X_co
    
# Drug
class GATGCN_Block(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nums_layer, n_heads=4):
        super(GATGCN_Block, self).__init__()
        assert nums_layer > 1
        netlist = [GATConv(input_size, hidden_size // n_heads, n_heads, 
                           activation=nn.ReLU(), allow_zero_in_degree=True)]
        for i in range(nums_layer-2):
            netlist.append(GATConv(hidden_size, hidden_size // n_heads, n_heads, 
                                   activation=nn.ReLU(), allow_zero_in_degree=True))
        self.net = nn.ModuleList(netlist)
        self.net_1 = GraphConv(hidden_size, output_size, 
                               norm='both', activation=nn.ReLU(), allow_zero_in_degree=True)
        
    def forward(self, g, feats):
        for layer in self.net:
            feats = layer(g, feats)
            feats = feats.reshape(feats.shape[0], -1)
        feats_1 = self.net_1(g, feats)
        
        return feats, feats_1

class Gated_NN(nn.Module):
    def __init__(self, hidden_size):
        super(Gated_NN, self).__init__()
        self.hidden_size = hidden_size
        w_r1, w_r2, b_r = self.get_params() 
        w_z1, w_z2, b_z = self.get_params() 
        w_h1, w_h2, b_h = self.get_params() 
        
        self.w_r1 = nn.Parameter(w_r1)
        self.w_r2 = nn.Parameter(w_r2)
        self.b_r = nn.Parameter(b_r)
        
        self.w_z1 = nn.Parameter(w_z1)
        self.w_z2 = nn.Parameter(w_z2)
        self.b_z = nn.Parameter(b_z)
        
        self.w_h1 = nn.Parameter(w_h1)
        self.w_h2 = nn.Parameter(w_h2)
        self.b_h = nn.Parameter(b_h)
        
    def forward(self, h0, h1):
        R = torch.sigmoid(h0 @ self.w_r1 + h1 @ self.w_r2 + self.b_r)
        Z = torch.sigmoid(h0 @ self.w_z1 + h1 @ self.w_z2 + self.b_z)
        h2 = torch.tanh(h1 @ self.w_h1 + ((R * h0) @ self.w_h2) + self.b_h)
        h = Z * h0 + (1-Z) * h2
        
        return h

    def get_params(self, ):
        return (self.normal((self.hidden_size, self.hidden_size)), 
                self.normal((self.hidden_size, self.hidden_size)), 
                torch.zeros(self.hidden_size))
    
    def normal(self, shape):
        return torch.normal(0, 1, shape) * 0.01
    
class GNNBlocks(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer, n_gnn, dropout):
        super(GNNBlocks, self).__init__()
        self.n_layer = n_layer
        self.GNNBlock0 = GCN_Block(input_size, hidden_size, hidden_size*2, nums_layer=3)
        # GNNBlocks
        GNNBlocks = []
        for _ in range(n_layer):
            GNNBlock_n =  GATGCN_Block(hidden_size*2, hidden_size*2, hidden_size*4, nums_layer=n_gnn) 
            GNNBlocks.append(GNNBlock_n)
        self.GNNBlocks = nn.ModuleList(GNNBlocks)
        # LayerNorm Block
        LNs = []
        for _ in range(n_layer):
            ln = nn.LayerNorm(hidden_size*2)
            LNs.append(ln)
        self.LNs = nn.ModuleList(LNs)
        
        self.gate = Gated_NN(hidden_size*2)
        self.scale = torch.sqrt(torch.FloatTensor([0.5]))
        self.glu = nn.GLU(dim=-1)
        self.do = nn.Dropout(dropout)
        self.maxpool = AvgPooling()
        
    def forward(self, g, feats):
        h, h_1 = self.GNNBlock0(g, feats)
        for i in range(self.n_layer):
            GNNBlock_n = self.GNNBlocks[i]
            LN = self.LNs[i]
            h1, h1_1 = GNNBlock_n(g, h_1)
            h2 = (h1 + self.do(self.glu(h1_1))) * self.scale.to(feats.device)
            h2 = LN(h2)
            h_1 = self.gate(h_1, h2)
        output = self.maxpool(g, h_1)

        return output

# DT prediction
class FeatureFusion(nn.Module):
    def __init__(self, size1, size2, hid_size):
        super(FeatureFusion, self).__init__()
        self.fc1 = nn.Linear(size1, hid_size)
        self.fc2 = nn.Linear(size2, hid_size)
        
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, x1, x2):
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x1 = F.normalize(x1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)
        xco = x1 + x2
        op = self.maxpool(xco.permute(0,2,1)).squeeze(-1)
        
        return op

class GNNBlockDTI(nn.Module):
    def __init__(self, net_D, net_T, net_T1, Drug_len, Target_len, Target_len1, hid_size, dropout=0.2):
        super(GNNBlockDTI, self).__init__()
        self.net_D = net_D
        self.net_T = net_T
        self.net_T1 = net_T1
        self.FF = FeatureFusion(Target_len, Target_len1, hid_size)
        self.pair_net = nn.Sequential(nn.Linear(Drug_len + hid_size, 1024),
                                      nn.ReLU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(1024, 1024),
                                      nn.ReLU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(1024, 512))
        self.fc = nn.Linear(512, 2)

    def forward(self, pairs):
        D_G, T, T_G = pairs
        Dfeats = D_G.ndata['feat']
        Tfeats = T_G.ndata['feat']
        Tweights = T_G.edata["weight"]

        D_x = self.net_D(D_G, Dfeats)
        T_x1 = self.net_T1(T_G, Tfeats, Tweights)
        T_x = self.net_T(T)
        T_x1 = T_x1.reshape(T_x.shape[0], -1, T_x1.shape[-1])
        T_xco = self.FF(T_x, T_x1)
        
        DT = torch.cat([D_x, T_xco], dim=-1)
        DT_x = self.pair_net(DT)
        output = self.fc(DT_x)

        return output.squeeze(-1)
    
