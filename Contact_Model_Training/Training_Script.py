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
import pandas as pd
from torch import nn
from torch.nn import functional as F



network = Final_Net(2,36, 50, 5, 5, 16, dropout=0.5).to('cuda')
#state_dict = torch.load('Attempt2/Model_53.pt')
#state_dict = network.state_dict()
#pretrain_dict = {k:v for k, v in torch.load('Attempt4/Model_9.pt').items() if k.split('.')[0] == 'transformer'} 
#state_dict.update(pretrain_dict)
network.load_state_dict(state_dict)
training_data = MSA_Dataset(path='Training_Data/', data='both')
validation_data = MSA_Dataset(path='Validation_Data/', data='both')
#dist_bins = torch.arange(0,17)/16*20+2
dist_bins = torch.tensor([2,8])


def collate(batch):
    return dist_custom_collate(batch, dist_bins, 128)

training_loader = DataLoader(training_data, batch_size=2, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate)
validation_loader = DataLoader(validation_data, batch_size=2, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate)
torch.autograd.set_detect_anomaly(True)
accumulate = 4


optimizer = AdamW(network.parameters(), lr=2e-5)


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
        
loss_func = FocalLoss(2,ignore_index=-100)

loss_hist = pd.DataFrame()

if os.path.exists('Attempt1/loss'):
    loss_hist = torch.load('Attempt1/loss')


epochs = 100
desc = '""'
for epoch in range(53, epochs):

    t = tqdm(training_loader, desc=desc)
    
    k = 0

    for i, batch in enumerate(t):
        

        optimizer.zero_grad()
        
        feat, feat_mask, dist_labels = batch[0].to('cuda'), batch[1].to('cuda'), batch[2].to('cuda')

        dist, _, _ = network(feat, mask=feat_mask)


        loss = loss_func(dist+1e-6, dist_labels)/accumulate

        var = torch.mean(torch.var(torch.softmax(dist[-1], dim=1), dim=0)).item()

        t.set_description(str(round(loss.item() * accumulate,2)) + ' ' + str(round(var, 4)))

        loss.backward()    
        if (accumulate ==1) or (i % (accumulate -1)) == 0:
            torch.nn.utils.clip_grad_value_(network.res_net.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            k += 1

            if k % 20 == 0:
                with torch.no_grad():
    
                    network.eval()
                    d = {}
                
                    val_loss = 0
                    val_accuracy = 0
                
                    train_loss = 0
                    train_accuracy = 0
                    for j, val_batch in enumerate(validation_loader):
                        feat, feat_mask, dist_labels  = val_batch[0].to('cuda'), val_batch[1].to('cuda'), val_batch[2].to('cuda') 
                        dist, _, _ = network(feat, mask=feat_mask)
                        val_loss += loss_func(dist, dist_labels) 
                    
                        val_accuracy += torch.sum((torch.max(dist, dim=1)[1] == dist_labels).float() * (dist_labels != -100).float()) / torch.sum((dist_labels != -100).float())
                        if j > 3:
                            break

                

                    for j, val_batch in enumerate(training_loader):
                        feat, feat_mask, dist_labels = val_batch[0].to('cuda'), val_batch[1].to('cuda'), val_batch[2].to('cuda')
                        dist, _, _ = network(feat, mask=feat_mask)
                        train_loss += loss_func(dist, dist_labels)
                    

                        train_accuracy += torch.sum((torch.max(dist, dim=1)[1] == dist_labels).float() * (dist_labels != -100).float()) / torch.sum((dist_labels != -100).float())
                        if j > 3:
                            break

                    d['training_accuracy'] = train_accuracy.item()/5
                    d['validation_accuracy'] = val_accuracy.item()/5
                    d['training_loss'] = train_loss.item()/5
                    d['validation_loss'] = val_loss.item()/5
                    
                    loss_hist = loss_hist.append(d, ignore_index=True)

                    del val_loss, train_loss
                    torch.cuda.empty_cache()
                    network.train()
                    torch.save(loss_hist, 'Attempt1/loss')
            
                torch.save(network.state_dict(), 'Attempt1/Model_'+ str(epoch+1) +'.pt')


