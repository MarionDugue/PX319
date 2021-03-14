import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from Attention_Mechanisms import Seq_Attention, Res_Attention
from Conv_Net.two_dim_resnet import make_two_dim_resnet
from torch.utils.checkpoint import checkpoint as cp
import os

    
class Seq_Transformer_Block(nn.Module):
    '''sequence wise transfomer block '''
    def __init__(self, embed_size, heads, embed_expand=4, dropout=0.0):
        super(Seq_Transformer_Block, self).__init__()
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size) 
                
        self.attention = Seq_Attention(embed_size, heads)
        
        self.FF = nn.Sequential(
            nn.Linear(embed_size, embed_expand*embed_size),
            nn.ReLU(),
            nn.Linear(embed_size*embed_expand, embed_size)
        )
        self.dropout = nn.Dropout(p=dropout)

               
    def forward(self, x, mask= None, dummy=None):
        
        attn = self.attention(x, mask, dummy=None)
        x = self.dropout(self.norm1(x)+ attn)
        x = self.dropout(self.norm2(self.FF(x) + x)) 
        
        return x
    
    
class Res_Transformer_Block(nn.Module):
    '''Residue wise transfomer block '''

    def __init__(self, embed_size, heads, embed_expand=4, dropout=0.0):
        super(Res_Transformer_Block, self).__init__()
        

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.attention = Res_Attention(embed_size, heads)
        
        self.FF = nn.Sequential(
            nn.Linear(embed_size, embed_expand*embed_size),
            nn.ReLU(),
            nn.Linear(embed_size*embed_expand, embed_size)
        )
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, mask= None, dummy=None):
        
        _, attn = self.attention(x, mask, dummy=None)
        x = self.dropout(_ + self.norm1(x))
        x = self.dropout(self.norm2(self.FF(x) + x))
        
        return x, attn
        
        

    
class Prot_Transformer(nn.Module):
    
    '''The iterative transformer, each iteation I stuck with 1 transformer block for simplicity and time.
    An initial embedding layer is used
    '''
    
    def __init__(self, embed_size, res_heads, seq_heads, embed_expand=4, dropout=0.0):
        super(Prot_Transformer, self).__init__()
        
        self.embedding = nn.Sequential(nn.Linear(22, embed_size*embed_expand),
                                      nn.ELU(),
                                      nn.Linear(embed_expand*embed_size, embed_size),
                                      nn.ELU())

        self.Res_transformerlist = nn.ModuleList([Res_Transformer_Block(embed_size, res_heads, embed_expand, dropout) for i in range(5)])
        self.Seq_transformerlist = nn.ModuleList([Seq_Transformer_Block(embed_size, seq_heads, embed_expand, dropout) for i in range(5)])
        
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        
    
    def forward(self, x, mask=None ):
        

        
        x = self.embedding(x)
        recorded = []
        
        for  i in range(len(self.Res_transformerlist)):
            
            x = cp(self.Seq_transformerlist[i], x, mask, self.dummy_tensor)            
            
            x, res_attn = cp(self.Res_transformerlist[i], x, mask, self.dummy_tensor)
        
            recorded.append(res_attn)            
            
            

        res_attn = torch.cat(recorded, dim=1)

        return x[:,:,0].permute(0,2,1) , res_attn
            
  
