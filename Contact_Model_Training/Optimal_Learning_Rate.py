from Final_Models import Final_Net
import torch
from torch.utils.data import DataLoader
from  Dataloader import MSA_Dataset, in_mem_MSA_Dataset, dist_custom_collate
from tqdm import tqdm
import random
import time
import os
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiplicativeLR as MLR
import pandas as pd
from torch import nn
from torch.nn import functional as F



network = Final_Net(2,36, 50, 5, 5, 16, dropout=0.5).to('cuda')
network.train()
training_data = MSA_Dataset(path='Training_Data/', data='both')
validation_data = MSA_Dataset(path='Validation_Data/', data='both')
#dist_bins = torch.arange(65)/64 *20 + 2
#angle_bins = torch.arange(-18,19)/18*3.1416

dist_bins = torch.tensor([2,8])


def collate(batch):
    return dist_custom_collate(batch,dist_bins, 64)

training_loader = DataLoader(training_data, batch_size=4, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate)
accumulate = 1

torch.autograd.set_detect_anomaly(True)

optimizer = AdamW(network.parameters(), lr=1e-6)
scheduler = MLR(optimizer, lambda x: 1.1)

class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='mean')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy

        if self.reduction == 'mean':
            return torch.mean(loss)


def loss_func(pred, label):
    loss = torch.sum(((100/(pred+1e-6) - 100/(label+1e-6))* (label > 0).float())**2) / torch.sum(label > 0)
    return loss
    
loss_func = nn.CrossEntropyLoss(ignore_index=-100)
optimizer.zero_grad()


loss_hist = pd.DataFrame()


epochs = 100
desc = '""'
iterations = 1

t = tqdm(training_loader)

for i, batch in enumerate(t):


    feat, feat_mask, dist_labels, phi_labels, psi_labels = batch[0].to('cuda'), batch[1].to('cuda'), batch[2].to('cuda'), batch[3].to('cuda'), batch[4].to('cuda')
                
    dist, phi, psi = network(feat, iterations, mask=feat_mask, intermediates=False)

    loss = (1*loss_func(dist+1e-6, dist_labels)) / accumulate



    mean = torch.mean(torch.mean(torch.softmax(dist, dim=1), dim=0)).item()
    var = torch.mean(torch.var(torch.softmax(dist, dim=1), dim=0)).item()

    t.set_description(str(round((loss*accumulate).item(),2)) + ' ' + str(iterations) + ' ' + str(round(mean, 2))+ ' ' + str(round(var,4)))
    
    loss_hist = loss_hist.append({'loss':(loss*accumulate).item(), 'lr': scheduler.get_last_lr()[0]}, ignore_index=True)

    loss.backward()    

    if accumulate ==1 or (i % (accumulate -1)) == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        torch.save(loss_hist, 'lr_v_loss')


