
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from . import SelfAttention, AdditiveAttention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim, 
                 n_head = 1, 
                 penalization = False, 
                 dropout = 0): 
        super().__init__()
        
        # relevant quantities
        self.emb_dim = emb_dim
        self.out_dim = n_head * emb_dim
        self.penalization = penalization
        self.n_head = n_head
        
        # layers
        self.attn_list = nn.ModuleList([SelfAttention(emb_dim, dropout) for i in range(n_head)])
        
    def compute_penalty(self, weights) :
        weights_t = weights.transpose(1, 2)
        def_pos   = [torch.mm(weights[i], weights_t[i]) for i in range(weights.size(0))] # size (batch_size, n_heads, n_heads)
        ide       = Variable(torch.eye(self.n_head)).to(weights.device)
        penal     = torch.sum(torch.cat([torch.norm(mmt - ide).view(1) for mmt in def_pos]))
        return penal
    
    def forward(self, embeddings, 
                lengths = None,
                compute_penalty = False):
        # compute self-attention
        outputs = [attn(embeddings, lengths) for attn in self.attn_list]
        applied = torch.cat([out[0] for out in outputs], dim = 1) # size (batch_size, n_heads, embedding_dim)
        weights = torch.cat([out[1] for out in outputs], dim = 1) # size (batch_size, n_heads, input_length)

        # compute penalty
        if self.penalization and compute_penalty and self.n_head > 1 :
            penal = self.compute_penalty(weights)
            return (applied, weights, penal)
        elif compute_penalty : 
            return (applied, weights, None)
        else :
            return (applied, weights)


# -- OLD --
class MultiHeadAttention(nn.Module):
    '''Module performing additive attention over a sequence of vectors stored in
       a memory block, conditionned by some vector. At instanciation it takes as imput :
       
                - query_dim : the dimension of the conditionning vector
                - targets_dim : the dimension of vectors stored in memory
                
      Other ideas on Multi head attention on 
      https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py
      https://github.com/tlatkowski/multihead-siamese-nets/blob/master/layers/attention.py
    '''
    def __init__(self, 
                 device, 
                 n_heads, 
                 query_dim, 
                 targets_dim, 
                 n_layers = 2
                ): 
        super(MultiHeadAttention, self).__init__()
        
        # relevant quantities
        self.device = device
        self.n_level = 1
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # parameters
        self.attn_modules_list = nn.ModuleList([AdditiveAttention(query_dim, targets_dim, n_layers) for i in range(n_heads)])

    def forward(self, query = None, targets = None):
        '''takes as parameters : 
                a query tensor conditionning the attention,     size = (1, n_heads, query_dim)
                a tensor containing attention targets           size = (targets_length, n_heads, targets_dim)
           returns : 
                the resulting tensor of the attention process,  size = (1, n_heads, targets_dim)
                the attention weights,                          size = (n_heads, 1, targets_length)
        '''
        print("multihead attention")
        targets_length = targets.size(0)
        targets_dim    = targets.size(2)
        attn_applied   = Variable(torch.zeros(1, self.n_heads, targets_dim)).to(self.device)
        attn_weights   = torch.zeros(self.n_heads, 1, targets_length).to(self.device)
        for i, attn in enumerate(self.attn_modules_list) :
            que = query[:, i, :] if query is not None else None
            print(que.size())
            tar = targets[:, i, :].unsqueeze(1)
            print(tar.size())
            attn_appl, attn_wghts = attn(que, tar)
            print(attn_appl.size())
            print(attn_wghts.size())
            attn_applied[:, i, :] = attn_appl.squeeze(1)
            attn_weights[i, :, :] = attn_wghts.squeeze(0)
        return attn_applied, attn_weights
