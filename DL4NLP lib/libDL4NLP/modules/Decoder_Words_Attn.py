
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from . import Attention, AdditiveAttention 


class AttnDecoder(nn.Module):
    '''Transforms a vector into a sequence of words'''
    def __init__(self, word2vec, attn_dim, hid_dim,
                 n_layer = 1,
                 dropout = 0.1,
                 bound   = 25,
                 temperature = 1,
                 top = None
                ):
        super().__init__()
        
        # relevant quantities
        self.hid_dim = hid_dim
        self.n_layer = n_layer
        self.bound   = bound
        self.temp    = temperature
        self.top     = top
        
        # modules
        self.word2vec = word2vec
        
        self.gru = nn.GRU(
            word2vec.out_dim,
            hid_dim, 
            n_layer, 
            dropout = dropout,
            batch_first = True)
        
        self.attn = Attention(
            emb_dim   = attn_dim, 
            query_dim = hid_dim, 
            dropout   = dropout)
        
        self.out  = nn.Linear(
            attn_dim + hid_dim, 
            word2vec.lang.n_words)
        
        self.dropout = nn.Dropout(dropout)
    
    
    def initWordTensor(self, index_list, device = None) :
        word = torch.LongTensor(index_list).view(-1, 1)     # size (batch_size, 1)
        word = Variable(word)                               # size (batch_size, 1)
        if device is not None : word = word.to(device)      # size (batch_size, 1)
        return word
        
        
    def generateWord(self, hidden, embeddings, word):
        '''word is a LongTensor with size (batch_size, 1)'''
        # update hidden state
        embedding = self.word2vec.embedding(word)       # size (batch_size, 1, embedding_dim)
        embedding = self.dropout(embedding)             # size (batch_size, 1, embedding_dim)
        _, hidden = self.gru(embedding, hidden)         # size (n_layers, batch_size, embedding_dim)
        # compute attention
        query = hidden[-1].unsqueeze(1)                 # size (batch_size, 1, embedding_dim)
        query = query.expand(query.size(0), 
                             embeddings.size(1), 
                             query.size(2))             # size (batch_size, sequence_length, embedding_dim)
        attn, weights = self.attn(embeddings, query)    # size (batch_size, 1, embedding_dim)
        # merge hidden with attention
        merge = torch.cat([hidden[-1], attn.squeeze(1)], dim = 1) 
        merge = self.dropout(merge)                     # size (batch_size, embedding_dim + hidden_dim)
        # generate next word
        vect = self.out(merge)                          # size (batch_size, lang_size)
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
    
    
    def forward(self, hidden, embeddings, device = None) :
        answer  = []
        weights = None
        EOS_token = self.word2vec.lang.getIndex('EOS')
        SOS_token = self.word2vec.lang.getIndex('SOS')
        word      = self.initWordTensor([SOS_token], device = device)
        hidden    = hidden[-self.n_layer:]             # size (n_layer, 1, hidden_dim)
        # word generation
        for t in range(self.bound) :
            # compute next word
            vect, hidden, attn = self.generateWord(hidden, embeddings, word)
            proba = self.computeProba(vect, device)
            word  = self.sampleWord(proba)
            # cumulate word and attention weight
            if word.item() != EOS_token :
                answer.append(word.item())
                if t == 0 : weights = attn
                else      : weights = torch.cat((weights, attn), dim = 1) # size(1, output_length, input_length)
            # stopping criterion
            else : break
        return (answer, weights)


# -- OLD --
class AttnWordsDecoder(nn.Module):
    '''Transforms a vector into a sequence of words'''
    def __init__(self, 
                 device, 
                 embedding, 
                 hidden_dim, 
                 tracking_dim,
                 n_layers = 0, 
                 dropout = 0.1,
                 tf_ratio = 1,
                 bound = 25
                ):
        super(AttnWordsDecoder, self).__init__()
        # relevant quantities
        self.device = device
        self.hidden_dim = hidden_dim
        self.tracking_dim = tracking_dim
        self.n_layers = n_layers
        self.tf_ratio = tf_ratio
        self.bound = bound
        # modules
        self.embedding = embedding
        for p in embedding.parameters() :
            lang_size     = p.data.size()[0]
            embedding_dim = p.data.size()[1]
        self.gru = nn.GRU(embedding_dim + tracking_dim, hidden_dim)
        self.attn = AdditiveAttention(hidden_dim, hidden_dim, n_layers) 
        self.concat = nn.Linear(2 * hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, lang_size)
        self.dropout = nn.Dropout(dropout)
        
        
    def generateWord(self, query_words, query_vector, hidden, current_word_index):
        # update hidden state
        embedded = self.embedding(current_word_index).view(1, 1, -1)
        if query_vector is not None : embedded = torch.cat((query_vector, embedded), dim = 2)
        embedded = self.dropout(embedded)
        _, hidden = self.gru(embedded, hidden)
        # generate next word
        attn, attn_weights = self.attn(hidden, query_words)
        vector = self.concat(torch.cat((hidden, attn), dim = 2)).tanh()
        vector = self.out(vector).squeeze(0)
        log_proba = F.log_softmax(vector, dim = 1)
        return log_proba, hidden
    
    
    def forward(self, query_words, query_vector, decision_vector, target_answer = None) :
        log_probas = []
        answer = []
        di = 0
        ta = target_answer if random.random() < self.tf_ratio else None
        current_word_index = Variable(torch.LongTensor([[0]])).to(self.device) # SOS_token
        hidden = self.dropout(decision_vector)
        for di in range(self.bound) :
            log_proba, hidden = self.generateWord(query_words, query_vector, hidden, current_word_index)
            topv, topi = log_proba.data.topk(1)
            log_probas.append(log_proba)
            ni = topi[0][0] # index of current generated word
            if ni == 1 : # EOS_token
                break
            elif ta is not None : # Teacher forcing
                answer.append(ni)
                if di < ta.size(0) :
                    current_word_index = ta[di].to(self.device)
                else :
                    break
            else :
                answer.append(ni)
                current_word_index = Variable(torch.LongTensor([[ni]])).to(self.device)
        return answer, log_probas
