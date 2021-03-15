import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import os

class Seq_Attention(nn.Module):
    
    '''Sequence-wise multihead attention, returns Attn(Q,K,V) across sequence dimension.
       Idea is the network can learn which sequences to focus to get a better representation
       of our residue that converts to proximity when sending through the conv net.
    '''
    
    def __init__(self, embed_size,  seq_heads):
        
        super(Seq_Attention, self).__init__()

        self.embed_size = embed_size
        self.seq_heads = seq_heads
        
        self.seq_head_dim = embed_size // seq_heads

        assert embed_size == seq_heads*self.seq_head_dim, "embed size isnt divisible by heads"
        
        
        self.qseq = nn.Linear(embed_size, embed_size, bias=False)
        self.vseq = nn.Linear(embed_size, embed_size, bias=False)
        self.kseq = nn.Linear(embed_size, embed_size, bias=False)
        
    def forward(self, x, mask=None, dummy=None):
    
        #x shape (batch, res, seq, embed) -> (batch, head,  res, seq, head_dim)
    
        q = self.qseq(x)
        k = self.kseq(x)
        v = self.vseq(x)
        
        q = q.reshape(*q.shape[:-1], self.seq_heads, self.seq_head_dim).permute(0,3,1,2,4).contiguous()
        k = k.reshape(*k.shape[:-1], self.seq_heads, self.seq_head_dim).permute(0,3,1,2,4).contiguous()
        v = v.reshape(*v.shape[:-1], self.seq_heads, self.seq_head_dim).permute(0,3,1,2,4).contiguous()
        

        seq_attn = torch.einsum('abcdi, abcei -> abcde', q,k) / np.sqrt(self.seq_head_dim)    
        if mask != None:
            seq_attn = torch.einsum('abcde, ace -> abcde', seq_attn, mask)
            
        seq_attn = torch.softmax(seq_attn, dim=-1)

            
        out = torch.einsum('abcde, abcef -> abcdf', seq_attn, v)
        out = torch.cat(out.unbind(dim=1),dim=-1)
        
        return out
        
        
class Res_Attention(nn.Module):
    
    '''Res-wise multihead attention, returns Attn(Q,K,V) across reside dimension
       Calculates the Attention between residues which is used as input for the conv net.
       Used to update residue representations for learning proximity and possibly physical 
       notions that can affect proximity.
    '''

    
    def __init__(self, embed_size,  res_heads):
        
        super(Res_Attention, self).__init__()

        self.embed_size = embed_size
        self.res_heads = res_heads
        
        self.res_head_dim = embed_size // res_heads

        assert embed_size == res_heads*self.res_head_dim, "embed size isnt divisible by heads"
        
        
        self.qres = nn.Linear(embed_size, embed_size, bias=False)
        self.vres = nn.Linear(embed_size, embed_size, bias=False)
        self.kres = nn.Linear(embed_size, embed_size, bias=False)
        
    def forward(self, x, mask=None, dummy=None):
    
        #x shape (batch,  res, seq, embed) -> (batch, head, res, seq, head_dim)
    
        q = self.qres(x)
        k = self.kres(x)
        v = self.vres(x)
        
        q = q.reshape(*q.shape[:-1], self.res_heads, self.res_head_dim).permute(0,3,1,2,4).contiguous()
        k = k.reshape(*k.shape[:-1], self.res_heads, self.res_head_dim).permute(0,3,1,2,4).contiguous()
        v = v.reshape(*v.shape[:-1], self.res_heads, self.res_head_dim).permute(0,3,1,2,4).contiguous()
        

        res_attn = torch.einsum('abcdi, abedi -> abcde', q,k) / np.sqrt(self.res_head_dim)
        
        res_attn = torch.mean(res_attn, dim=-2)
        
        
        if mask != None:
            res_attn = torch.einsum('abce, ae -> abce', res_attn, mask[:,:, 0])
            
        attn = res_attn
        res_attn = torch.softmax(res_attn, dim=-1)
            
        out = torch.einsum('abce, abedf -> abcdf', res_attn, v)
        out = torch.cat(out.unbind(dim=1),dim=-1)
        
        return out, attn
        
