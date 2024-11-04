import math
import torch
from torch import nn
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc

CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25 }

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHAR_ISO_SMI_LEN = 64
CHAR_PROT_LEN = 25


def get_cindex(Y, P):
    summ = 0
    pair = 0
    
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i != j:
                if(Y[i] > Y[j]):
                    pair +=1
                    summ +=  1* (P[i] > P[j]) + 0.5 * (P[i] == P[j])
        
            
    if pair != 0:
        return summ/pair
    else:
        return 0


def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        print('Using GPU!')
        return torch.device(f'cuda:{i}')
    else:
        print('Using CPU!')
        return torch.device('cpu')


# 梯度裁剪
def grad_clipping(net, theta): 
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

            
# 分类指标汇总：

def classify_metrics(y_true, Y_pred):
    auroc = round(roc_auc_score(y_true, Y_pred), ndigits = 5)
    aupr = round(average_precision_score(y_true, Y_pred), ndigits = 5)
    
    precision, recall, thresholds = precision_recall_curve(y_true, Y_pred)
    # 选取最佳阈值
    fscore = (2 * precision * recall) / (precision + recall)
    index = np.argmax(fscore)
    #thresholdOpt = round(thresholds[index], ndigits = 4)
    thresholdOpt = thresholds[index]
    
    fscoreOpt = round(fscore[index], ndigits = 5)
    recallOpt = round(recall[index], ndigits = 5)
    precisionOpt = round(precision[index], ndigits = 5)
    
    y_pred = Y_pred > thresholdOpt
    
    try:
        accuracyOpt = round(accuracy_score(y_true, y_pred), ndigits = 5)
    except:
        print(y_true)
        print(y_pred)
    
    return auroc, aupr, accuracyOpt, precisionOpt, recallOpt, fscoreOpt
    