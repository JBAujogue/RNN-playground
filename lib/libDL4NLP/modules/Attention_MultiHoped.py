
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from . import AdditiveAttention


class MultiHopedAttention(nn.Module):
    '''Module performing additive attention over a sequence of vectors stored in
       a memory block, conditionned by some vector. At instanciation it takes as imput :
       
                - query_dim : the dimension of the conditionning vector
                - targets_dim : the dimension of vectors stored in memory
    '''
    def __init__(self, 
                 device,
                 targets_dim,
                 base_query_dim = 0,
                 hops = 1,
                 share = True,
                 transf = False,
                 dropout = 0
                ):
        super(MultiHopedAttention, self).__init__()

        # dimensions
        self.targets_dim = targets_dim
        self.output_dim = targets_dim
        self.hops_query_dim = self.output_dim if hops > 1 else 0
        self.query_dim = base_query_dim + self.hops_query_dim
        
        # structural coefficients
        self.device = device
        self.n_level = 1
        self.hops = hops
        self.share = share
        self.transf = transf
        self.dropout_p = dropout
        if dropout > 0 : self.dropout = nn.Dropout(p = dropout)
        
        # parameters
        self.attn = AdditiveAttention(self.query_dim, self.targets_dim) 
        self.transf = nn.GRU(self.targets_dim, self.targets_dim) if transf else None
        
        
    def initQuery(self): 
        if self.hops_query_dim > 0 :
            return Variable(torch.zeros(1, 1, self.hops_query_dim)).to(self.device)
        return None
    
    
    def update(self, hops_query, decision_vector):
        if self.transf is not None : _ , update = self.transf(decision_vector, hops_query)
        else                       :     update = hops_query + decision_vector
        return update
    
    
    def forward(self, words_memory, base_query = None):
        attn_weights_list = []
        hops_query = self.initQuery() if (self.hops > 1 and self.share) else None
        
        for hop in range(self.hops) :
            if base_query is not None and hops_query is not None : query = torch.cat((base_query, hops_query), 2) # size (1, self.n_heads, self.query_dim)
            elif base_query is not None                          : query = base_query
            elif hops_query is not None                          : query = hops_query
            else                                                 : query = None
            
            decision_vector, attn_weights = self.attn(query, words_memory)
            attn_weights_list.append(attn_weights)
            hops_query = self.update(hops_query, decision_vector) if (self.hops > 1 and hops_query is not None) else decision_vector
  
        # output decision vector
        return hops_query, attn_weights_list
