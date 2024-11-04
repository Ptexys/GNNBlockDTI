# 封装——训练（单机单卡/单机多卡版），测试
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import pickle
import os
import json
import time
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc, f1_score

# from mess import *  #评估函数

#==========================================================================

# 测试
# 评估函数 需要动态定义
def estimate(y_true, Y_pred):

    Y_pred = F.softmax(Y_pred, 1)  #torch.tensor(
    Y_pred_label = np.argmax(Y_pred, axis=1)
    Y_pred_score = Y_pred[:, 1]  # 取 正概率
    
    auroc = round(roc_auc_score(y_true, Y_pred_score), ndigits = 5)
    
    tpr, fpr, _ = precision_recall_curve(y_true, Y_pred_score)
    aupr = round(auc(fpr, tpr), ndigits = 5)
    
    Precision = precision_score(y_true, Y_pred_label)
    Reacll = recall_score(y_true, Y_pred_label)
    Accuracy = accuracy_score(y_true, Y_pred_label)
    F1 = f1_score(y_true, Y_pred_label)
    
     # 关键指标放在第一个
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

#==========================================================================

# 训练——单机单卡
# CV版，默认是1倍交叉验证，即 k=0； // 这个可以在保存名称中实现，不需要特地搞个变量；
def train(model, epochs, train_iter, test_iter, valid_iter, 
          loss, optimizer, compare, 
          device, Name):  
    '''
    ——test_iter : 有些训练不需要进行测试，如在预训练模型中；所以要先判断test_iter是否为空；
                  不需要训练时，test_iter=None;
    ——loss : 损失函数； eg: nn.MSELoss()
    ——optimizer : 优化器； eg. torch.optim.Adam(model.parameters(), lr=lr)
    ——messes : 评估函数列表； eg. [ACC, nn.MSELoss()也算]; 最关键地、起决定性地指标放在第一个
    ——compare : "min" 或 "max", "min"表示测试指标越小越好，反之；
    ——device : cpu 或 gpu；
    ——Name : 一般是 模型名称+数据集(+CV-K)；
    '''
    model.to(device)
    print('Running on', device)
    
    train_L = []  # 训练损失保存；
    test_L = []   # 测试损失保存；
    Metrics = []   # 关键指标保存
    
    test_best = []         # 最优的测试结果保存，初始化；
    best = 0               # 最优测试结果的轮次
    # 最优的测试指标保存，或大而优，或小而优；需要参数说明——compare
    metrs_best = 10000 if compare=="min" else -10
    train_L_best = 1000000 # 比如在预训练任务中，这时虽然不需要测试，但还是要保存训练模型；
    
    if not os.path.exists('models'):
        os.makedirs('models')
    print("--------------------------------------", Name.replace('_', '——'))
    

    for epoch in range(epochs):
        train_L_1epoch = []
        start = time.time()
        
        model.train()       # 训练模式
        for step, (D,T,T1,Y) in enumerate(train_iter):
            D = D.to(device)
            T1 = T1.to(device)
            T = T.to(device)
            Y_pred = model((D,T,T1))
            
            L = loss(Y_pred.cpu().to(torch.float32), Y.to(torch.long))
            
            # 反向传播
            optimizer.zero_grad()
            L.backward()
            train_L_1epoch.append(float(L))
            optimizer.step()
            # 训练损失保存
        with torch.no_grad():
            l = np.average(train_L_1epoch)  
            train_L.append(l)
            
        # 测试
        if test_iter is None:   # 比如在预训练任务中，这时虽然不需要测试，但还是要保存训练模型；返回的话，无所谓
            # 保存训练模型
            if train_L[-1] < train_L_best:
                torch.save(model.state_dict(), "models/" + Name + ".pth")
                print(f'The best model in epoch{epoch+1} has been saved!!!')
            continue # 断
            
        test_loss, test_metric = test(model, test_iter, loss, device)   # messes: 评估指标列表；
        test_L.append(test_loss)
        Metrics.append(test_metric)

        # 再说一遍！！
        # 测试指标保存，不一定是损失，还可以是ACC，AUPR等等；
        
        print(f'epoch: {epoch+1} train_loss: {train_L[-1]:.3f} test_loss: {test_L[-1]:.3f}')
        # 结果更好时，保存更新，并在验证集上进行测试
        metrs = test_metric[0]  # 关键指标放在第一个
        IsBetter = metrs < metrs_best if compare=="min" else metrs > metrs_best
        if IsBetter:
            test_best = test_metric
            metrs_best = metrs
            best = epoch+1
            torch.save(model.state_dict(), "models/" + Name + ".pth")
            print(f'The best model in epoch{epoch+1} has been saved!!! ', test_best)
            
            if valid_iter:
                _, valid_metric = test(model, valid_iter, loss, device)
                test_best = valid_metric
                print("valid metric:", valid_metric)

        time_epoch = time.time() - start
        print(f'best in epoch{best}!!!  {time_epoch:.3f}s')
        
    # 清除缓存
    torch.cuda.empty_cache()

    return train_L, test_L, Metrics, test_best
            
#==========================================================================

# 训练——单机多卡
def train_mmp(model, epochs, train_iter, test_iter, valid_iter, 
          loss, optimizer, compare, 
          device, Name):  
    '''
    ——test_iter : 有些训练不需要进行测试，如在预训练模型中；所以要先判断test_iter是否为空；
                  不需要训练时，test_iter=None;
    ——loss : 损失函数； eg: nn.MSELoss()
    ——optimizer : 优化器； eg. torch.optim.Adam(model.parameters(), lr=lr)
    ——messes : 评估函数列表； eg. [ACC, nn.MSELoss()也算]; 最关键地、起决定性地指标放在第一个
    ——compare : "min" 或 "max", "min"表示测试指标越小越好，反之；
    ——device : cpu 或 gpu；
    ——Name : 一般是 模型名称+数据集(+CV-K)；
    所有注释参照train()；
    '''
    model.to(device)
    model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=False)
    print('Running on', device)
    
    train_L = []  # 训练损失保存；
    test_L = []   # 测试损失保存；
    Metrics = []   # 关键指标保存
    
    test_best = []         # 最优的测试结果保存，初始化；
    best = 0               # 最优测试结果的轮次
    # 最优的测试指标保存，或大而优，或小而优；需要参数说明——compare
    metrs_best = 10000 if compare=="min" else -10
    train_L_best = 1000000 # 比如在预训练任务中，这时虽然不需要测试，但还是要保存训练模型；
    
    if dist.get_rank() == 0:
        if not os.path.exists('models'):
            os.makedirs('models')
        print("--------------------------------------", Name.replace('_', '——'))

    for epoch in range(epochs):
        train_iter.sampler.set_epoch(epoch)
        train_L_1epoch = []
        start = time.time()
        
        model.train()       # 训练模式
        for step, (D,T,T1,Y) in enumerate(train_iter):
            D = D.to(device)
            T1 = T1.to(device)
            T = T.to(device)
            Y_pred = model((D, T, T1))
            Y_pred = torch.sigmoid(Y_pred)
            L = loss(Y_pred.cpu().to(torch.float32), Y.to(torch.float32)) 
            # 反向传播
            optimizer.zero_grad()
            L.backward()
            train_L_1epoch.append(float(L))
            optimizer.step()
            # 训练损失保存
        with torch.no_grad():
            l = np.average(train_L_1epoch)  
            train_L.append(l)
            
        # 测试
        if test_iter is None:   # 比如在预训练任务中，这时虽然不需要测试，但还是要保存训练模型；返回的话，无所谓
            # 保存训练模型
            if train_L[-1] < train_L_best and dist.get_rank() == 0:
                torch.save(model.state_dict(), "models/" + Name + ".pth")
                print(f'The best model in epoch{epoch+1} has been saved!!!')
            continue # 断
            
        test_loss, test_metric = test(model, test_iter, loss, device)   # messes: 评估指标列表；
        test_L.append(test_loss)
        Metrics.append(test_metric)
        
        # 再说一遍！！
        # 测试指标保存，不一定是损失，还可以是ACC，AUPR等等；
        time_epoch = time.time() - start
        if dist.get_rank() == 0:
            print(f'epoch: {epoch+1} train_loss: {train_L[-1]:.3f} test_loss: {test_L[-1]:.3f} {time_epoch:.3f}s')
        # 结果更好时，保存更新
        metrs = test_metric[0]
        IsBetter = metrs < metrs_best if compare=="min" else metrs > metrs_best
        if IsBetter and dist.get_rank() == 0:
            test_best = test_metric
            metrs_best = metrs
            best = epoch+1
            torch.save(model.state_dict(), "models/" + Name + ".pth")
            print(f'The best model in epoch{epoch+1} has been saved!!!', test_best)
            
            if valid_iter:
                _, valid_metric = test(model, valid_iter, loss, device)
                print("valid metric:", valid_metric)

        if dist.get_rank() == 0:
            print(f'best in epoch{best}!!!')

    return train_L, test_L, Metrics, test_best
