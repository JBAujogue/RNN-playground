
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, dropout = 0): 
        super(SelfAttention, self).__init__()
        
        # relevant quantities
        self.embedding_dim = embedding_dim
        self.output_dim = embedding_dim
        
        # parameters
        self.dropout = nn.Dropout(p = dropout)
        self.attn_layer = nn.Linear(embedding_dim, embedding_dim)
        self.attn_v = nn.Linear(embedding_dim, 1, bias = False)
        self.act = F.softmax
        
    def forward(self, embeddings):
        weights = self.attn_layer(embeddings).tanh()       # size (batch_size, input_length, embedding_dim)
        weights = self.act(self.attn_v(weights), dim = 1)  # size (batch_size, input_length, 1)
        weights = torch.transpose(weights, 1, 2)           # size (batch_size, 1, input_length)
        attn_applied = torch.bmm(weights, embeddings)      # size (batch_size, 1, embedding_dim)
        attn_applied = self.dropout(attn_applied)
        return attn_applied, weights



class Attention(nn.Module):
    def __init__(self, embedding_dim, query_dim, 
                 dropout = 0, 
                 method = 'concat' 
                ): 
        super(Attention, self).__init__()
        
        # relevant quantities
        self.method        = method
        self.embedding_dim = embedding_dim
        self.output_dim    = embedding_dim
        
        # parameters
        self.dropout    = nn.Dropout(p = dropout)
        self.attn_layer = nn.Linear(embedding_dim + query_dim, embedding_dim)
        self.attn_v     = nn.Linear(embedding_dim, 1, bias = False)
        self.act        = F.softmax
        
    def forward(self, embeddings, query):
        '''embeddings       of size (batch_size, input_length, embedding_dim)
           query (optional) of size (batch_size, 1, embedding_dim)
        '''
        # query is optional for this method
        if self.method == 'concat' :
            weights = torch.cat((query, embeddings), 2) if query is not None else embeddings
            weights = self.attn_layer(weights).tanh()          # size (batch_size, input_length, embedding_dim)
            weights = self.act(self.attn_v(weights), dim = 1)  # size (batch_size, input_length, 1)
            weights = torch.transpose(weights, 1, 2)           # size (batch_size, 1, input_length)
            
        # query is necessary for this method
        elif self.method == 'dot' :
            query   = torch.transpose(query, 1, 2)             # size (batch_size, query_dim, 1)
            weights = torch.bmm(embeddings, query)             # size (batch_size, input_length, 1)
            weights = self.act(weights, dim = 1)               # size (batch_size, input_length, 1)
            weights = torch.transpose(weights, 1, 2)           # size (batch_size, 1, input_length)
        applied = self.dropout(torch.bmm(weights, embeddings)) # size (batch_size, 1, embedding_dim)
        return applied, weights


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
