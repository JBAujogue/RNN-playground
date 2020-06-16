
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from . import Attention, AdditiveAttention 


class PAFAttnDecoder(nn.Module):
    '''Transforms a vector into a sequence of words'''
    def __init__(self, word2vec, attn_dim, hid_dim,
                 n_layer = 1,
                 attn_method = 'concat',
                 dropout = 0.1,
                 bound   = 25,
                 temperature = 1,
                 top = None
                ):
        super().__init__()
        
        # relevant quantities
        self.attn_dim = attn_dim
        self.hid_dim  = hid_dim
        self.n_layer  = n_layer
        self.method   = attn_method
        self.bound    = bound
        self.temp     = temperature
        self.top      = top
        
        # modules
        self.word2vec = word2vec
        
        self.gru = nn.GRU(
            word2vec.out_dim + attn_dim,
            hid_dim, 
            n_layer, 
            dropout = dropout,
            batch_first = True)
        
        self.attn = Attention(
            emb_dim   = attn_dim, 
            query_dim = hid_dim, 
            method    = attn_method,
            dropout   = dropout)
        
        self.out = nn.Linear(hid_dim + attn_dim, word2vec.lang.n_words)
        
        self.dropout = nn.Dropout(dropout)
    
    
    def initWordTensor(self, index_list, device = None) :
        word = torch.LongTensor(index_list).view(-1, 1)       # size (batch_size, 1)
        word = Variable(word)                                 # size (batch_size, 1)
        if device is not None : word = word.to(device)        # size (batch_size, 1)
        return word
        
    
    def initContext(self, batch_size) :
        return torch.zeros((batch_size, 1, self.attn_dim), dtype = torch.float)
        
    
    def generateWord(self, hidden, word, context, embeddings, lengths = None):
        '''word is a LongTensor with size (batch_size, 1)'''
        # merge previous word with previous attention
        embedding = self.word2vec.embedding(word)             # size (batch_size, 1, emb_dim)
        embedding = self.dropout(embedding)                   # size (batch_size, 1, emb_dim)
        context   = self.dropout(context)                     # size (batch_size, 1, attn_dim)
        embedding = torch.cat((embedding, context), dim = 2)  # size (batch_size, 1, emb_dim + attn_dim)
        # update hidden state
        _, hidden = self.gru(embedding, hidden)               # size (n_layers, batch_size, hid_dim)
        # compute next attention
        query = self.dropout(hidden[-1].unsqueeze(1))         # size (batch_size, 1, hid_dim)
        context, attn = self.attn(embeddings, query, lengths) # size (batch_size, 1, attn_dim)
        # merge hidden with attention
        merge = torch.cat([hidden[-1], context.squeeze(1)], dim = 1) 
        merge = self.dropout(merge)                           # size (batch_size, hid_dim + attn_dim)
        # generate next word
        vect = self.out(merge)                                # size (batch_size, lang_size)
        return (vect, hidden, context, attn)
    

    def computeProba(self, vect, device = None):
        '''Converts a word repartition vector with size (batch_size, lang_size)
           into a probability vector with size (batch_size, lang_size)'''
        # apply temperature
        vect /= self.temp
        # top-p proba refactoring
        if type(self.top) == float and 0 < self.top < 1 : 
            # TODO
            proba = F.softmax(vect, dim = 1)
        #top-k proba refactoring
        elif type(self.top) == int and self.top > 0 : 
            vals, inds = vect.topk(self.top, dim = 1)        # size (batch_size, self.top)
            proba_vals = F.softmax(vals, dim = 1)            # size (batch_size, self.top)
            proba = torch.FloatTensor(vect.size()).zero_()
            if device is not None : proba = proba.to(device)
            proba.scatter_(1, inds, proba_vals)              # size (batch_size, lang_size)
        # vanilla softmax
        else :
            proba = F.softmax(vect, dim = 1)
        return proba
    
    
    def sampleWord(self, proba) :
        '''Selects word indices out of a probability distribution over vocab'''
        # choose best
        word = proba.topk(1, dim = 1)[1] 
        # random sampling
        # TODO
        return word
    
    
    def forward(self, hidden, embeddings, lengths = None, device = None) :
        answer  = []
        weights = None
        EOS_token = self.word2vec.lang.getIndex('EOS')
        SOS_token = self.word2vec.lang.getIndex('SOS')
        word      = self.initWordTensor([SOS_token], device = device)
        context   = self.initContext(1)
        hidden    = hidden[-self.n_layer:]             # size (n_layer, 1, hid_dim)
        # word generation
        for t in range(self.bound) :
            # compute next word
            vect, hidden, context, attn = self.generateWord(hidden, word, context, embeddings, lengths)
            proba = self.computeProba(vect, device)
            word  = self.sampleWord(proba)
            # cumulate word and attention weight
            if word.item() != EOS_token :
                answer.append(word.item())
                if t == 0 : weights = attn
                else      : weights = torch.cat((weights, attn), dim = 1) # size(1, out_length, in_length)
            # stopping criterion
            else : break
        return (answer, weights)
