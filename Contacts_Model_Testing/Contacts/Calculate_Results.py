import sys
sys.path.append('..')
from Final_Models import Final_Net
from Transformer_Net import Prot_Transformer
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import product
from tqdm import tqdm
from sklearn.metrics import average_precision_score




network = Final_Net(2, 36, 50, 5,5, 16).to('cuda')
network.load_state_dict(torch.load('../../Contact_Model_Training/Training_Checkpoints/Model_98.pt'))
network.eval()


def func(x):
    rslt = x.split('_')[0]
    if '-' in rslt:
        return rslt
    
    return rslt+'-D1'



a = set(pd.read_csv('CASP14_Results/L_FL.txt', delim_whitespace=True, skiprows=8).Domain)
b = set([func(i) for i in os.listdir('CASP14_Data/MSA_tensors/')])

proteins = list(a.intersection(b))
rnges = ['S', 'ML', 'L']

print(proteins)

df = pd.DataFrame()

for protein, rnge in tqdm(product(proteins, rnges)):
    
    try:
        features = torch.tensor(torch.load('CASP14_Data/MSA_tensors/' + protein + '_MSA.pt'))
        dist_mat = torch.load('CASP14_Data/distance_matrices/' + protein + '_distmat.pt')
        
    except:
        features = torch.tensor(torch.load('CASP14_Data/MSA_tensors/' + protein[:-3] + '_MSA.pt'))
        dist_mat = torch.load('CASP14_Data/distance_matrices/' + protein[:-3] + '_distmat.pt')
    
    
    features = torch.nn.functional.one_hot(features[None], num_classes=22).float().permute(0,2, 1,3)
    out = torch.bucketize(dist_mat, torch.tensor([2,8]))
    out = out[:]
    out -= 1
    out[out == -1] = 0
    out = 1 - out
    features = features[:,:, :300]


    
    with torch.no_grad():
        preds, _, _ = network(features.to('cuda'), mask=None)
    
    preds = preds.to('cpu')
    preds = torch.softmax(preds[0,:], dim=0).cpu()
    pred = preds[0] 
    
    pred = (pred + pred.T)/2
    
    if rnge == 'S':
        mask = torch.ones_like(out)

        torch.diagonal(mask,offset=0).fill_(0)
        for j in range(7, features.shape[1]):
            torch.diagonal(mask,offset=j).fill_(0)
            torch.diagonal(mask,offset=-j).fill_(0)
            
    if rnge == 'ML':
        mask = torch.ones_like(out)

        torch.diagonal(mask,offset=0).fill_(0)
        for j in range(12):
            torch.diagonal(mask,offset=j).fill_(0)
            torch.diagonal(mask,offset=-j).fill_(0)
            
    if rnge == 'L':
        mask = torch.ones_like(out)

        torch.diagonal(mask,offset=0).fill_(0)
        for j in range(24):
            torch.diagonal(mask,offset=j).fill_(0)
            torch.diagonal(mask,offset=-j).fill_(0)
    
    
    
    pred_cont = torch.zeros_like(pred)
    v, i = torch.topk((pred*mask).flatten(), (features.shape[1]*2 + 2)//5)
    pred_cont[torch.tensor(np.unravel_index(i.numpy(), pred.shape)).T] = 1

    L5_recall = ((((pred_cont) * (out))*mask).sum() / ((out)*mask).sum()).item()
    L5_prec =  ((((pred_cont) * (out))*mask).sum() / ((pred_cont)*mask).sum()).item()

    if (L5_recall + L5_prec) == 0:
        L5_f1 = None
    else:
        L5_f1 = 2*(L5_recall*L5_prec)/(L5_recall+L5_prec)
        
    pred_cont = (pred >= 0.5).long()

    FL_recall = ((((pred_cont) * (out))*mask).sum() / ((out)*mask).sum()).item()
    FL_prec =  ((((pred_cont) * (out))*mask).sum() / ((pred_cont)*mask).sum()).item()
    if (FL_recall + FL_prec) == 0:
        FL_f1 = None
    else:
        FL_f1 = 2*(FL_recall*FL_prec)/(FL_recall+FL_prec)
        
        
    AUC = average_precision_score((out*mask).flatten(), (pred*mask).flatten())
    
    
    df = df.append({'protein': protein,
                    'range' : rnge,
                    'L5': L5_f1,
                    'FL': FL_f1,
                    'AUC_PR': AUC
                   }, ignore_index=True)
    
df.to_csv('results.csv')
    
        
    

    
    
    
