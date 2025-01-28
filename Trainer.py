# 封装——训练（单机单卡/单机多卡版），测试
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import pickle
import os
import time
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc, f1_score

def estimate(y_true, Y_pred):
    Y_pred = F.softmax(Y_pred, 1) 
    Y_pred_label = np.argmax(Y_pred, axis=1)
    Y_pred_score = Y_pred[:, 1]
    
    auroc = round(roc_auc_score(y_true, Y_pred_score), ndigits = 5)
    tpr, fpr, _ = precision_recall_curve(y_true, Y_pred_score)
    aupr = round(auc(fpr, tpr), ndigits = 5)
    Precision = precision_score(y_true, Y_pred_label)
    Reacll = recall_score(y_true, Y_pred_label)
    Accuracy = accuracy_score(y_true, Y_pred_label)
    F1 = f1_score(y_true, Y_pred_label)

    return auroc, aupr, Accuracy, Precision, Reacll, F1 

def test(model, data_iter, loss, device):
    model.to(device)
    model.eval()
    test_L = []
    Y_preds = torch.empty(0)
    Ys = torch.empty(0)
    for _, (D,T,T1,Y) in enumerate(data_iter):
        D = D.to(device)
        T1 = T1.to(device)
        T = T.to(device)
        Y_pred = model((D,T,T1,))
        with torch.no_grad():
            L = loss(Y_pred.cpu().to(torch.float32), Y.to(torch.long))
            test_L.append(L)
            Y_preds = torch.cat([Y_preds, Y_pred.cpu()], dim=0)
            Ys = torch.cat([Ys, Y], dim=0)
            
    test_L = np.asarray(test_L)
    test_loss = np.average(test_L, axis=0)
    mess = estimate(Ys, Y_preds)

    return test_loss, mess

def train(model, epochs, train_iter, valid_iter, test_iter, loss, optimizer, compare, device, Name):  
    model.to(device)
    print('Running on', device)
    
    train_L = []
    test_L = []
    Metrics = []
    test_best = []
    best = 0 
    metrs_best = 10000 if compare=="min" else -10
    if not os.path.exists('models'):
        os.makedirs('models')
    print("--------------------------------------", Name.replace('_', '——'))
    for epoch in range(epochs):
        train_L_1epoch = []
        start = time.time()
        
        model.train()
        for step, (D,T,T1,Y) in enumerate(train_iter):
            D = D.to(device)
            T1 = T1.to(device)
            T = T.to(device)
            Y_pred = model((D,T,T1))
            L = loss(Y_pred.cpu().to(torch.float32), Y.to(torch.long))
            optimizer.zero_grad()
            L.backward()
            train_L_1epoch.append(float(L))
            optimizer.step()
        with torch.no_grad():
            l = np.average(train_L_1epoch)  
            train_L.append(l)
        valid_loss, valid_metric = test(model, valid_iter, loss, device)
        valid_L.append(valid_loss)
        Metrics.append(valid_metric)
        print(f'epoch: {epoch+1} train_loss: {train_L[-1]:.3f} valid_loss: {valid__L[-1]:.3f}')
        
        metrs = valid_metric[0]
        IsBetter = metrs < metrs_best if compare=="min" else metrs > metrs_best
        if IsBetter:
            metrs_best = metrs
            best = epoch+1
            torch.save(model.state_dict(), "models/" + Name + ".pth")
            print(f'The best model in epoch{epoch+1} has been saved!!! ')
            _, test_metric = test(model, test_iter, loss, device)
            test_best= test_metric
            print("test metric:", test_metric)

        time_epoch = time.time() - start
    torch.cuda.empty_cache()

    return train_L, test_L, Metrics, test_best