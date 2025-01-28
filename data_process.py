import pickle
import torch
import re
import math
import os
import numpy as np
import json
import pickle
from collections import OrderedDict
from tqdm import tqdm
import argparse
import dgl
from rdkit import Chem
import transformers
from transformers import BertForMaskedLM, BertTokenizer, pipeline
import esm
from esm import pretrained

def one_hot(char, dict): 
    h = torch.zeros(len(dict)+1)
    for i in range(len(dict)):
        if dict[i] == char:
            h[i] = 1
            break
        elif i == len(dict)-1:
            h[-1] = 1
    return h


def smi_2_graph(smi):
    mol = Chem.MolFromSmiles(smi)
    nums_atom = mol.GetNumAtoms()
    
    u, v = [], []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        u.append(begin.GetIdx())
        v.append(end.GetIdx())
        
    u, v = torch.tensor(u), torch.tensor(v)
    g = dgl.graph((u, v))
    
    feats = []
    for atom in mol.GetAtoms():
        feat_0 = one_hot(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca',
                                             'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn',
                                             'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au',
                                             'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb'])
        feat_1 = one_hot(atom.GetFormalCharge(), [1,2,3,4,5,6,7,8])
        feat_2 = one_hot(atom.GetDegree(), [1,2,3,4,5,6,7,8])
        feat_3 = torch.tensor([atom.GetIsAromatic()])
        feat_4 = torch.tensor([atom.IsInRing()])
        feat = torch.cat([feat_0,feat_1,feat_2,feat_3,feat_4])
        feats.append(feat)
        
    feats_1 = torch.cat([feats[idx].unsqueeze(0) for idx in range(len(feats))], 0)
    nums_n = min(max(max(u), max(v))+1, nums_atom)
    feats_2 = feats_1[:nums_n]
    bg = dgl.to_bidirected(g)
    bg.ndata['feat'] = feats_2
    
    return bg

def get_drug_graph(ligands):
    XD = {}
    for d in tqdm(ligands.keys(), ncols=0):
        try:
            XD[str(d)] = smi_2_graph(ligands[d])
        except:
            print(ligands[d])
            
    return XD

def get_protbert_embedding(data):
    embed = dict()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('prot_bert_bfd', do_lower_case=False )
    model = BertForMaskedLM.from_pretrained("prot_bert_bfd").to(device)
    
    for id, seq in tqdm(data.items(), ncols=0):
        seq1 = " ".join(list(re.sub(r"[UZOB]", "X", seq)))
        
        ids = tokenizer(seq1, return_tensors='pt')
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            
        output1 = output[0][0][1: -1]
        assert len(seq) == len(output1)
        
        embed[id] = output1.cpu()
        
    return embed

def padding(data, maxlen):
    data_new = dict()
    for id, vec in tqdm(data.items(), ncols=0):
        vec_new = torch.zeros((maxlen, vec.shape[1]), dtype=torch.float32)
        if vec.shape[0] < maxlen:
            vec_new[:vec.shape[0]] = vec[:]
        else:
            vec_new[:] = vec[:maxlen]
        data_new[id] = vec_new
        
    return data_new

def target_graph_construct(proteins):
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    target_distance = {}
    key_list=[]
    for key in proteins:
        key_list.append(key)
    device = torch.device("cuda:0")
    model.to(device)
    for k_i in tqdm(range(len(key_list))):
        key=key_list[k_i]
        data = []
        pro_id = key
        seq = proteins[key]
        if len(seq) <= 1000:
            data.append((pro_id, seq))
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            with torch.no_grad():
                results = model(batch_tokens.to(device), repr_layers=[33], return_contacts=True)
            contact_map = results["contacts"][0]
            target_distance[pro_id] = contact_map.cpu().numpy()
        else:
            contact_prob_map = np.zeros((len(seq), len(seq)))  # global contact map prediction
            interval = 500
            i = math.ceil(len(seq) / interval)
            # ======================
            # =                    =
            # =                    =
            # =          ======================
            # =          =*********=          =
            # =          =*********=          =
            # ======================          =
            #            =                    =
            #            =                    =
            #            ======================
            # where * is the overlapping area
            for s in range(i):
                start = s * interval  # sub seq predict start
                end = min((s + 2) * interval, len(seq))  # sub seq predict end
                sub_seq_len = end - start

                # prediction
                temp_seq = seq[start:end]
                temp_data = []
                temp_data.append((pro_id, temp_seq))
                batch_labels, batch_strs, batch_tokens = batch_converter(temp_data)
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                # insert into the global contact map
                row, col = np.where(contact_prob_map[start:end, start:end] != 0)
                row = row + start
                col = col + start
                contact_prob_map[start:end, start:end] = contact_prob_map[start:end, start:end] + results["contacts"][
                    0].numpy()
                contact_prob_map[row, col] = contact_prob_map[row, col] / 2.0
                if end == len(seq):
                    break
            target_distance[pro_id] = contact_prob_map

    return target_distance

def get_uv(adj):
    u, v = [], []
    weight = []
    m, n = len(adj), len(adj[0])
    for i in range(m):
        for j in range(n):
            if j == i-1 or j == i+1:
                u.append(i)
                v.append(j)
                weight.append(1.0)
                continue
            if adj[i][j] > 0.5:
                u.append(i)
                v.append(j)
                weight.append(adj[i][j])
                
    return u, v, weight

def get_target_graph(data, distance):  #{id:seq}; {id:map}
    Gs = dict()
    for id, feats in tqdm(data.items()):
        contact_map = distance[id]
        u, v, weight = get_uv(contact_map)
        g = dgl.graph((u,v))
        g.ndata["feat"] = feats
        g.edata["weight"] = torch.tensor(weight, dtype=torch.float32)
        Gs[id] = g
        
    return Gs

#--------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, default="BIOSNAP", help="dataset")
    args = parser.parse_args()
    
    path= os.path.join("./dataset", args.task)
    with open(os.path.join(path, "drug_smi_raw.pkl"), "rb") as f:
        drugs = pickle.load(f)
    with open(os.path.join(path, "prot_seq_raw.pkl"), "rb") as f:
        targets = pickle.load(f)
        
    drug_graph = get_drug_graph(drugs)
    target_embedding = get_protbert_embedding(targets)
    target_distance = target_graph_construct(targets)
    target_graph = get_target_graph(target_embedding, target_distance)
    
    with open(os.path.join(path, "drug_graph.pkl"), "wb") as f:
        pickle.dump(drug_graph, f)
    with open(os.path.join(path, "target_embedding.pkl"), "wb") as f:
        pickle.dump(target_embedding, f)
    with open(os.path.join(path, "target_graph.pkl"), "wb") as f:
        pickle.dump(target_graph, f)
    
    print(args.task, " Finished processing!")