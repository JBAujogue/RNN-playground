
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from . import Attention, AdditiveAttention 


class SmoothAttnDecoder(nn.Module):
    '''
    Converts a vector into a sequence of words,
    where next token prediction is based on previous token probability distribution
    rather than previous token selection
    '''
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
        lang_size, emb_dim = word2vec.embedding.weight.size()
        self.lang_size = lang_size
        self.emb_dim   = emb_dim
        self.attn_dim  = attn_dim
        self.hid_dim   = hid_dim
        self.n_layer   = n_layer
        self.method    = attn_method
        self.bound     = bound
        self.temp      = temperature
        self.top       = top
        
        # modules
        # dense embedding with shared weights
        self.embedding = nn.Linear(lang_size, emb_dim, bias = False)
        self.embedding.weight.data = word2vec.embedding.weight.data.t()
        for param in self.embedding.parameters() : param.requires_grad = False
            
        self.word2vec = word2vec
        
        self.gru = nn.GRU(
            emb_dim,
            hid_dim,
            n_layer,
            dropout = dropout,
            batch_first = True)
        
        self.attn = Attention(
            emb_dim   = attn_dim, 
            query_dim = hid_dim, 
            method    = attn_method,
            dropout   = dropout)
        
        self.out = nn.Linear(attn_dim + hid_dim, lang_size)

        self.act = F.log_softmax
        self.dropout = nn.Dropout(dropout)

        
    def initWordTensor(self, index_list, device = None) :
        '''converts a list of N word indices with vocab of size L
           into a dense FloatTensor of size (N, L)'''
        word = torch.zeros((len(index_list), self.lang_size), dtype = torch.float)
        for i, index in enumerate(index_list) : word[i, index] = 1.
        word = Variable(word)                                 # size (batch_size, lang_size)
        if device is not None : word = word.to(device)        # size (batch_size, lang_size)
        return word
        
        
    def generateWord(self, hidden, word, embeddings, lengths = None):
        '''word is a FloatTensor with size (batch_size, lang_size)'''
        # update hidden state
        embedding = self.embedding(word.unsqueeze(1))         # size (batch_size, 1, emb_dim)
        embedding = self.dropout(embedding)                   # size (batch_size, 1, emb_dim)
        _, hidden  = self.gru(embedding, hidden)              # size (n_layer, batch_size, hid_dim)
        # compute attention
        query = self.dropout(hidden[-1].unsqueeze(1))         # size (batch_size, 1, hid_dim)
        attn, weights = self.attn(embeddings, query, lengths) # size (batch_size, 1, attn_dim)
        # merge hidden with attention
        merge = torch.cat([hidden[-1], attn.squeeze(1)], dim = 1) 
        merge = self.dropout(merge)                           # size (batch_size, hid_dim + attn_dim)
        # generate next word
        vect = self.out(merge)                                # size (batch_size, lang_size)
        return (vect, hidden, weights)
    
    
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
        proba     = self.initWordTensor([SOS_token], device = device) 
        hidden    = hidden[-self.n_layer:]                 # size (n_layer, 1, hid_dim)
        # word generation
        for t in range(self.bound) :
            # compute next word
            vect, hidden, attn = self.generateWord(hidden, proba, embeddings, lengths)
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
