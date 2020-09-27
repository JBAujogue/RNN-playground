
import torch
import torch.nn as nn
import torch.nn.functional as F

from libDL4NLP.misc import Highway, HighwayQ


class SelfAttention(nn.Module):
    def __init__(self, emb_dim, 
                 dropout = 0): 
        super().__init__()

        # relevant quantities
        self.emb_dim = emb_dim
        self.out_dim = emb_dim

        # layers
        self.dropout = nn.Dropout(p = dropout)
        self.key     = nn.Sequential(Highway(emb_dim), nn.Linear(emb_dim, 1, bias = False))
        self.value   = Highway(emb_dim, dropout = dropout)
        self.act     = F.softmax
    
    def computeMask(self, lengths, max_length) :
        # see http://juditacs.github.io/2018/12/27/masked-attention.html
        return torch.arange(max_length).to(lengths.device)[None, :] < lengths[:, None]

    def forward(self, embeddings,
               lengths = None) :             # size (batch_size, max_length, emb_dim)  
        # compute weights
        weights = self.key(embeddings)       # size (batch_size, max_length, 1) 
        weights = weights.squeeze(2)         # size (batch_size, max_length)
        # mask attention over padded tokens
        if lengths is not None :
            mask = self.computeMask(lengths, weights.size(1))
            weights[~mask] = float('-inf')   # size (batch_size, max_length)
        # compute weighted sum
        weights = self.act(weights, dim = 1) # size (batch_size, max_length)
        weights = weights.unsqueeze(1)       # size (batch_size, 1, max_length)
        values  = self.value(embeddings)     # size (batch_size, max_length, emb_dim)
        applied = torch.bmm(weights, values) # size (batch_size, 1, emb_dim)
        return applied, weights



class Attention(nn.Module):
    def __init__(self, emb_dim, query_dim, 
                 dropout = 0, 
                 method = 'concat'):
        '''method must be chosen among [dot, concat, general]'''
        super().__init__()
        
        # relevant quantities
        self.method  = method
        self.emb_dim = emb_dim
        self.out_dim = emb_dim
        
        # layers
        self.dropout = nn.Dropout(p = dropout)
        if method == 'concat' : 
            self.key1 = HighwayQ(emb_dim, query_dim, dropout)
            self.key2 = nn.Linear(emb_dim, 1, bias = False)
        elif method == 'general' :
            self.key = (HighwayQ(query_dim, 0, dropout) if query_dim == emb_dim else nn.Linear(query_dim, emb_dim, bias = False))
        self.value   = HighwayQ(emb_dim, 0, dropout)
        self.act     = F.softmax

    def computeMask(self, lengths, emb_length) :
        # see http://juditacs.github.io/2018/12/27/masked-attention.html
        return torch.arange(emb_length).to(lengths.device)[None, :] < lengths[:, None]
        
    def forward(self, embeddings, query = None, 
                lengths = None):
        '''embeddings       of size (batch_size, emb_length, emb_dim)
           query (optional) of size (batch_size, 1, emb_dim)
        '''
        # compute attention weights
        # dot method
        if self.method == 'dot' :
            query   = query.transpose(1, 2)        # size (batch_size, query_dim, 1)
            weights = torch.bmm(embeddings, query) # size (batch_size, emb_length, 1)
        # concat method
        elif self.method == 'concat' :
            query   = query.expand(embeddings.size(0), 
                                   embeddings.size(1), 
                                   query.size(2)) if query is not None else None
            weights = self.key1(embeddings, query) # size (batch_size, emb_length, 1)
            weights = self.key2(weights)           # size (batch_size, emb_length, 1)
        # general method
        elif self.method == 'general' :
            query   = self.key(query)              # size (batch_size, 1, query_dim)
            query   = query.transpose(1, 2)        # size (batch_size, query_dim, 1)
            weights = torch.bmm(embeddings, query) # size (batch_size, emb_length, 1)
        # mask attention over padded tokens
        weights = weights.squeeze(2)               # size (batch_size, emb_length)
        if lengths is not None :
            mask = self.computeMask(lengths, weights.size(1))
            weights[~mask] = float('-inf')     # size (batch_size, emb_length)
        # compute weighted sum
        weights = self.act(weights, dim = 1)       # size (batch_size, emb_length)
        weights = weights.unsqueeze(1)             # size (batch_size, 1, emb_length)
        values  = self.value(embeddings)           # size (batch_size, emb_length, emb_dim)
        applied = torch.bmm(weights, values)       # size (batch_size, 1, emb_dim)
        return (applied, weights)


# -- OLD --
class AdditiveAttention(nn.Module):
    def __init__(self, 
                 query_dim, 
                 targets_dim, 
                 n_layers = 1
                ): 
        super(AdditiveAttention, self).__init__()
        # relevant quantities
        self.n_level = 1
        self.query_dim = query_dim
        self.targets_dim = targets_dim
        self.output_dim = targets_dim
        self.n_layers = n_layers
        # parameters
        self.attn_layer = nn.Linear(query_dim + targets_dim, targets_dim) if n_layers >= 1 else None
        self.attn_layer2 = nn.Linear(targets_dim, targets_dim) if n_layers >= 2 else None
        self.attn_v = nn.Linear(targets_dim, 1, bias = False) if n_layers >= 1 else None
        self.act = F.softmax
        
    def forward(self, query = None, targets = None):
        '''takes as parameters : 
                a query tensor conditionning the attention,     size = (1, batch_size, query_dim)
                a tensor containing attention targets           size = (targets_length, batch_size, targets_dim)
           returns : 
                the resulting tensor of the attention process,  size = (1, batch_size, targets_dim)
                the attention weights,                          size = (1, targets_length)
        '''
        if targets is not None :
            # concat method 
            if self.n_layers >= 1 :
                poids = torch.cat((query.expand(targets.size(0), -1, -1), targets), 2) if query is not None else targets
                poids = self.attn_layer(poids).tanh()                 # size (targets_length, batch_size, targets_dim)
                if self.n_layers >= 2 :
                    poids = self.attn_layer2(poids).tanh()            # size (targets_length, batch_size, targets_dim)
                attn_weights = self.attn_v(poids)                     # size (targets_length, batch_size, 1)
                attn_weights = torch.transpose(attn_weights, 0,1)     # size (batch_size, targets_length, 1)
                targets = torch.transpose(targets, 0,1)               # size (batch_size, targets_length, targets_dim)
            # dot method
            else :
                targets = torch.transpose(targets, 0,1)               # size (batch_size, targets_length, targets_dim)
                query = torch.transpose(query, 0, 1)                  # size (batch_size, 1, query_dim)
                query = torch.transpose(query, 1, 2)                  # size (batch_size, query_dim, 1)
                attn_weights = torch.bmm(targets, query)              # size (batch_size, targets_length, 1)
                
            attn_weights = self.act(attn_weights, dim = 1)        # size (batch_size, targets_length, 1)
            attn_weights = torch.transpose(attn_weights, 1,2)     # size (batch_size, 1, targets_length)
            attn_applied = torch.bmm(attn_weights, targets)       # size (batch_size, 1, targets_dim)
            attn_applied = torch.transpose(attn_applied, 0,1)     # size (1, batch_size, targets_dim)

        else :
            attn_applied = query
            attn_weights = None
        return attn_applied, attn_weights
