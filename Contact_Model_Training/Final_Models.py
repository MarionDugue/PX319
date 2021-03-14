from Conv_Net.two_dim_resnet import make_two_dim_resnet
from Transformer_Net import Prot_Transformer
import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint_sequential as cp_seq, checkpoint as cp


class Final_Net(nn.Module):
    '''Putting the whole model together.'''
    
    
    
    def __init__(self, ndistbins, nangbins, embed_size, res_heads, seq_heads, resnet_layers=64,  embed_expand=4, dropout=0.0):
        super(Final_Net, self).__init__()        
        
        self.resnet_layers = resnet_layers
        self.transformer = Prot_Transformer(embed_size, res_heads, seq_heads, embed_expand, dropout)
        
        self.res_net = make_two_dim_resnet(res_heads*10 + 2*embed_size, ndistbins, 20*res_heads + 4*embed_size, resnet_layers, batch_norm=True, atrou_rates=[1,2], dropout=dropout)
        
        
        self.angle_net = nn.Sequential(nn.Linear(embed_size,embed_size*embed_expand),
                                       nn.ReLU(),
                                       nn.Linear(embed_size*embed_expand,embed_size*embed_expand),
                                       nn.ReLU()
                                     )
        
        self.phi_net  = nn.Sequential(nn.Linear(embed_size*embed_expand,embed_size*embed_expand),
                                      nn.ReLU(),
                                      nn.Linear(embed_size*embed_expand,nangbins)
                                     )
        
        self.psi_net  = nn.Sequential(nn.Linear(embed_size*embed_expand,embed_size*embed_expand),
                              nn.ReLU(),
                              nn.Linear(embed_size*embed_expand,nangbins)
                             )
        
        

 
    def forward(self, x, mask=None):
        '''Intermediates allows for recording intermediate guesses for the distance bin prediction'''
        
            
        
        x, res_attn = self.transformer(x, mask)
        

        conv_inp = torch.cat(torch.broadcast_tensors(x[:,:, None], x[:, :, :, None]), dim=1)
        conv_inp = torch.cat((conv_inp, res_attn, res_attn.permute(0,1,3,2)),dim=1)
        
        #conv_inp = torch.cat((res_attn, res_attn.permute(0,1,3,2)), dim=1)

        dist = self.res_net(conv_inp)
        angle =  None #self.angle_net(x.permute(0,2,1))
        phi = None #self.phi_net(angle)       
        psi = None #self.psi_net(angle)
        
        return dist, phi, psi
        
        
       
