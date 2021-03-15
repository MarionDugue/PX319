from os import path as p
import os
import torch
from torch import tensor
from torch.utils.data import Dataset
from tqdm import tqdm
import random


class in_mem_MSA_Dataset(Dataset):
    def __init__(self,  path='.', data='dist'):
        
        assert data == 'dist' or data == 'angle' or data == 'both'
        assert p.exists(p.join(path, 'Features'))
                
            
        self.path = path
        self.data = data
        self.index = ['_'.join(i.split('_')[:-1]) for i in os.listdir(p.join(path,'Features'))]
        
        self.features = [torch.load(p.join(self.path,'Features' ,self.index[idx] + '_MSA.pt')) for idx in tqdm(range(len(self.index)), desc = 'Loading Features')]
        
        self.dist = [ torch.load(p.join(self.path,'Dist_Labels' ,self.index[idx] + '_distmat.pt'))  for idx in tqdm(range(len(self.index)), desc = 'Loading Distances')]
        
        self.angles =[ torch.load(p.join(self.path,'Angle_Labels' ,self.index[idx] + '_torang.pt'))  for idx in tqdm(range(len(self.index)), desc = 'Loading Angles')]
       
    def __getitem__(self, idx):
        if self.data == 'dist':
            return self.features[idx], self.dist[idx]
            
        if self.data == 'angle':
            return self.features[idx], self.angles[idx]
            
        return self.features[idx], self.dist[idx], self.angles[idx]
    
    def __len__(self):
        return len(self.index)
    

class MSA_Dataset(Dataset):
    def __init__(self,  path='.', data='dist'):
        
        assert data == 'dist' or data == 'angle' or data == 'both'
        assert p.exists(p.join(path, 'Features'))
                
            
        self.path = path
        self.data = data
        self.index = ['_'.join(i.split('_')[:-1]) for i in os.listdir(p.join(path,'Features'))]
        
       
    def __getitem__(self, idx):
        if self.data == 'dist':
            return torch.load(p.join(self.path,'Features' ,self.index[idx] + '_MSA.pt')), torch.load(p.join(self.path,'Dist_Labels' ,self.index[idx] + '_distmat.pt')) 
            
        if self.data == 'angle':
            return torch.load(p.join(self.path,'Features' ,self.index[idx] + '_MSA.pt')), torch.load(p.join(self.path,'Angle_Labels' ,self.index[idx] + '_torang.pt'))
            
        return torch.load(p.join(self.path,'Features' ,self.index[idx] + '_MSA.pt')), torch.load(p.join(self.path,'Dist_Labels' ,self.index[idx] + '_distmat.pt')), torch.load(p.join(self.path,'Angle_Labels' ,self.index[idx] + '_torang.pt'))
    
    def __len__(self):
        return len(self.index)
        
        


def dist_custom_collate(batch, dist_bins=None, angle_bins=None, crop_size=64):
    
    
    feat = torch.zeros(len(batch), crop_size, 300, 22)
    feat_mask = torch.ones(len(batch), crop_size, 300)

    
    dist_labels = torch.zeros(len(batch), crop_size, crop_size)
    
    phi_labels = torch.zeros(len(batch), crop_size)
    psi_labels = torch.zeros(len(batch), crop_size)
    
    for i in range(len(batch)):
        
        seq_len = batch[i][0].shape[1]
        msa_len = batch[i][0].shape[0]
        
        sampled_msa = torch.randperm(msa_len)[:300]
        
        if msa_len > 300:
            msa_len = 300
        
        if seq_len <= crop_size:
            crop = 0
            
            
        else:    
            crop = random.randint(0, seq_len - crop_size)
            seq_len = crop_size
        
        
        
        one_hot = torch.nn.functional.one_hot(batch[i][0][sampled_msa], num_classes=22).permute(1,0,2)
        
        feat[i, :seq_len, :msa_len] = one_hot[crop:crop+seq_len]
        feat_mask[i, seq_len:, msa_len:] = 0
        
        dist_labels[i, :seq_len, :seq_len] = batch[i][1][crop:crop+seq_len, crop:crop+seq_len,0]
        dist_labels[i][dist_labels[i] == 0] = -100

        phi_labels[i, :seq_len] = batch[i][2][crop:crop+seq_len, 0, 0]
        phi_labels[i, :seq_len][batch[i][2][crop:crop+seq_len, 0, 1] == 0] = -100                 

        psi_labels[i, :seq_len] = batch[i][2][crop:crop+seq_len, 1, 0]
        psi_labels[i, :seq_len][batch[i][2][crop:crop+seq_len, 1, 1] == 0] = -100                 
        
    if dist_bins != None:
        dist_labels[dist_labels != -100] = torch.bucketize(dist_labels[dist_labels != -100], dist_bins).float()
        dist_labels[(dist_labels == (dist_bins.shape[0]))] = -100
        dist_labels[(dist_labels == 0)] = -100
        #dist_labels[dist_labels != -100] -= 1
        dist_labels = dist_labels.long()

    if angle_bins != None:
        phi_labels[phi_labels != -100] = torch.bucketize(phi_labels[phi_labels != -100], angle_bins).float() - 1        
        psi_labels[psi_labels != -100] = torch.bucketize(psi_labels[psi_labels != -100], angle_bins).float() - 1
        phi_labels = phi_labels.long()
        psi_labels = psi_labels.long()
        
    return feat,  feat_mask, dist_labels, phi_labels, psi_labels



