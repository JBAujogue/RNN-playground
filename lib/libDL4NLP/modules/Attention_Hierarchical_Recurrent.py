
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from . import (RecurrentEncoder, 
               Attention, 
               AdditiveAttention, 
               MultiHeadAttention)


class HAN(nn.Module):
    '''Ce module d'attention est :
    
    - hiérarchique avec bi-GRU entre les deux niveaux d'attention
    - globalement multi-hopé, où il est possible d'effectuer plusieurs passes pour accumuler de l'information
    '''
    def __init__(self, emb_dim, hid_dim, query_dim,
                 n_layer = 1,
                 hops = 1,
                 share = True,
                 transf = False,
                 dropout = 0):
        super(HAN, self).__init__()
        
        # dimensions
        self.emb_dim = emb_dim
        self.query_dim = query_dim
        self.hid_dim = hid_dim
        self.out_dim = self.query_dim if (self.query_dim > 0 and \
                                         (transf or (hops > 1 and query_dim != hid_dim))) \
                                      else hid_dim
        self.hops = hops
        self.share = share
        
        # modules
        self.dropout = nn.Dropout(p = dropout)
        # first attention module
        if share : self.attn1 = nn.ModuleList([Attention(emb_dim, query_dim, dropout)] * hops)
        else     : self.attn1 = nn.ModuleList([Attention(emb_dim, query_dim, dropout) for _ in range(hops)])
        # intermediate encoder module
        self.bigru = RecurrentEncoder(emb_dim, hid_dim, n_layer, dropout, bidirectional = True)
        # second attention module
        if share : self.attn2 = nn.ModuleList([Attention(self.bigru.out_dim, query_dim, dropout)] * hops)
        else     : self.attn2 = nn.ModuleList([Attention(self.bigru.out_dim, query_dim, dropout) for _ in range(hops)])
        # accumulation step
        self.transf = nn.Linear(self.bigru.out_dim, self.out_dim, bias = False) if (transf or (self.hops > 1 and query_dim != self.bigru.out_dim)) else None
        
        
    def singlePass(self, packed_embeddings, query, attn1, attn2, lengths): 
        # first attention
        query1 = query.expand(packed_embeddings.size(0), 
                              packed_embeddings.size(1), 
                              query.size(2)) if query is not None else None
        output, weights1 = attn1(packed_embeddings, query1, lengths) # size (dialogue_length, 1, emb_dim)
        # intermediate biGRU
        output, _ = self.bigru(output.transpose(0, 1))               # size (1, dialogue_length, hid_dim)
        output = self.dropout(output)
        # second attention
        query2 = query.expand(output.size(0), 
                              output.size(1), 
                              query.size(2)) if query is not None else None
        output, weights2 = attn2(output, query2)                     # size (1, dialogue_length, hid_dim)
        # output decision vector
        if self.transf is not None : output = self.transf(output)    # size (1, 1, out_dim)
        if query is not None       : output = output + query
        return (output, weights1, weights2)
        
        
    def forward(self, packed_embeddings, query = None, lengths = None):
        weights1_list = []
        weights2_list = []
        # perform attention loops
        if packed_embeddings is not None :
            for hop in range(self.hops) :
                # perform attention pass
                query, weights1, weights2 = self.singlePass(
                    packed_embeddings, 
                    query, 
                    self.attn1[hop], 
                    self.attn2[hop], 
                    lengths)
                weights1_list.append(weights1)
                weights2_list.append(weights2)
        # output decision vector
        return (query, weights1_list, weights2_list)



# -- OLD --
class RecurrentHierarchicalAttention(nn.Module):
    '''Ce module d'attention est :
    
    - hiérarchique avec bi-GRU entre chaque niveau d'attention
    - multi-tête sur chaque niveau d'attention
    - globalement multi-hopé, où il est possible d'effectuer plusieurs passes pour accumuler de l'information
    '''

    def __init__(self, 
                 device,
                 word_hid_dim, 
                 sentence_hid_dim,
                 query_dim = 0, 
                 n_heads = 1,
                 n_layer = 1,
                 hops = 1,
                 share = True,
                 transf = False,
                 dropout = 0
                ):
        super(RecurrentHierarchicalAttention, self).__init__()
        
        # dimensions
        self.query_dim = query_dim
        self.word_hid_dim = word_hid_dim
        self.sentence_input_dim = self.word_hid_dim
        self.sentence_hid_dim = sentence_hid_dim
        self.context_vector_dim = sentence_hid_dim * 2
        self.out_dim = self.query_dim if (transf or self.hops > 0) else self.context_vector_dim
        
        # structural coefficients
        self.device = device
        self.n_level = 2
        self.n_heads = n_heads
        self.n_layer = n_layer
        self.hops = hops
        self.share = share
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p = dropout)
        
        # first attention module
        attn1_list = []
        if share :
            attn1 = MultiHeadAdditiveAttention(n_heads, self.query_dim, self.word_hid_dim) if n_heads > 1 else \
                    AdditiveAttention(self.query_dim, self.word_hid_dim) 
            for hop in range(hops) : attn1_list.append(attn1)
            self.attn1 = nn.ModuleList(attn1_list)
        else :
            for hop in range(hops):
                attn1 = MultiHeadAdditiveAttention(n_heads, self.query_dim, self.word_hid_dim) if n_heads > 1 else \
                        AdditiveAttention(self.query_dim, self.word_hid_dim) 
                attn1_list.append(attn1)
            self.attn1 = nn.ModuleList(attn1_list)
        
        # intermediate encoder module
        self.bigru = nn.GRU(self.sentence_input_dim, 
                            self.sentence_hid_dim, 
                            n_layer,
                            dropout=(0 if n_layer == 1 else dropout), 
                            bidirectional=True)
        
        # second attention module
        attn2_list = []
        if share :
            attn2 = MultiHeadAdditiveAttention(n_heads, self.query_dim, self.context_vector_dim) if n_heads > 1 else \
                    AdditiveAttention(self.query_dim, self.context_vector_dim) 
            for hop in range(hops) : attn2_list.append(attn2)
            self.attn2 = nn.ModuleList(attn2_list)
        else :
            for hop in range(hops):
                attn2 = MultiHeadAdditiveAttention(n_heads, self.query_dim, self.context_vector_dim) if n_heads > 1 else \
                        AdditiveAttention(self.query_dim, self.context_vector_dim) 
                attn2_list.append(attn2)
            self.attn2 = nn.ModuleList(attn2_list)
        
        # accumulation step
        self.transf = nn.Linear(self.context_vector_dim, self.out_dim, bias = False) \
                      if (transf or self.hops > 0) else None


    def initQuery(self): 
        if self.query_dim > 0 :
            return Variable(torch.zeros(1, self.n_heads, self.query_dim)).to(self.device)
        return None
        
                
    def initHidden(self): 
        return Variable(torch.zeros(2 * self.n_layer, self.n_heads, self.sentence_hid_dim)).to(self.device)
        
        
    def singlePass(self, words_memory, query, attn1, attn2): 
        L = len(words_memory)
        attn1_weights = {}
        bigru_inputs = Variable(torch.zeros(L, self.n_heads, self.sentence_input_dim)).to(self.device)
        # first attention layer
        for i in range(L) :
            targets = words_memory[i]                              # size (N_i, 1, 2*word_hid_dim)
            targets = targets.repeat(1, self.n_heads, 1)           # size (N_i, n_heads, 2*word_hid_dim)
            attn1_output, attn1_wghts = attn1(query, targets)
            attn1_weights[i] = attn1_wghts
            bigru_inputs[i] = attn1_output.squeeze(0)              # size (n_heads, 2*word_hid_dim)
        # intermediate biGRU
        bigru_hidden = self.initHidden()
        attn2_inputs, bigru_hidden = self.bigru(bigru_inputs, bigru_hidden)  # size (L, n_heads, 2*word_hid_dim)
        # second attention layer
        attn2_inputs = self.dropout(attn2_inputs)
        decision_vector, attn2_weights = attn2(query = query, targets = attn2_inputs)
        attn2_weights = attn2_weights.view(-1)
        # output decision vector
        return decision_vector, attn1_weights, attn2_weights
    
    
    def update(self, query, decision_vector):
        update = query + self.transf(decision_vector) if self.transf is not None else query + decision_vector
        return update
        
        
    def forward(self, words_memory, query = None):
        '''takes as parameters : 
                a tensor containing words_memory vectors        dim = (words_memory_length, word_hid_dim)
                a tensor containing past queries                dim = (words_memory_length, query_dim)
           returns : 
                the resulting decision vector                   dim = (1, 1, query_dim)
                the weights of first attention layer (dict)     
                the weights of second attention layer (dict)
        '''
        attn1_weights_list = []
        attn2_weights_list = []
        if len(words_memory) > 0 :
            if query is not None : query = query.repeat(1, self.n_heads, 1)
            elif self.hops > 1   : query = self.initQuery()
            for hop in range(self.hops) :
                decision_vector, attn1_weights, attn2_weights = self.singlePass(words_memory, 
                                                                                query, 
                                                                                self.attn1[hop], 
                                                                                self.attn2[hop])
                attn1_weights_list.append(attn1_weights)
                attn2_weights_list.append(attn2_weights)
                query = self.update(query, decision_vector)  # size (L, self.n_heads, self.out_dim)

        # output decision vector
        return query, attn1_weights_list, attn2_weights_list
