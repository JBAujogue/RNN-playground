
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Decoder(nn.Module):
    '''Transforms a vector into a sequence of words'''
    def __init__(self, word2vec, hid_dim, 
                 n_layer = 1,
                 dropout = 0.1,
                 bound   = 25
                ):
        super().__init__()
        
        # relevant quantities
        self.hid_dim = hid_dim
        self.n_layer = n_layer
        self.bound   = bound
        # modules
        self.word2vec = word2vec
        
        self.gru = nn.GRU(
            word2vec.out_dim,
            hid_dim, 
            n_layer, 
            dropout = dropout, 
            batch_first = True)
        
        self.out = nn.Linear(
            hid_dim, 
            word2vec.lang.n_words)
        
        self.dropout = nn.Dropout(dropout)
    
    
    def initWordTensor(self, index_list, device = None) :
        word = torch.LongTensor(index_list).view(-1, 1)     # size (batch_size, 1)
        word = Variable(word)                               # size (batch_size, 1)
        if device is not None : word = word.to(device)      # size (batch_size, 1)
        return word
        
        
    def generateWord(self, hidden, word):
        '''word is a LongTensor with size (batch_size, 1)'''
        embedding = self.word2vec.embedding(word)       # size (batch_size, 1, emb_dim)
        embedding = self.dropout(embedding)             # size (batch_size, 1, emb_dim)
        _, hidden = self.gru(embedding, hidden)         # size (n_layer, batch_size, emb_dim)
        vect      = self.out(hidden[-1])                # size (batch_size, lang_size)
        return vect, hidden
    
    
    def forward(self, hidden, device = None) :
        answer = []
        EOS_token = self.word2vec.lang.getIndex('EOS')
        SOS_token = self.word2vec.lang.getIndex('SOS')
        word      = self.initWordTensor([SOS_token], device = device)
        hidden    = hidden[-self.n_layer:]              # size (n_layer, 1, hid_dim)
        # word generation
        for t in range(self.bound) :
            # compute next word proba
            vect, hidden = self.generateWord(hidden, word)
            # compute next word index
            word_index = vect.topk(1, dim = 1)[1].item()
            # stopping criterion
            if word_index == EOS_token : break
            else : 
                answer.append(word_index)
                word = vect.topk(1, dim = 1)[1].view(1, 1)
        return answer



# -- OLD --
class WordsDecoder(nn.Module):
    '''Transforms a vector into a sequence of words'''
    def __init__(self, 
                 device, 
                 embedding, 
                 hidden_dim, 
                 tracking_dim, 
                 dropout = 0.1,
                 tf_ratio = 1,
                 EOS_token = 1,
                 bound = 25
                ):
        super(WordsDecoder, self).__init__()
        # relevant quantities
        self.device = device
        self.hidden_dim = hidden_dim
        self.tracking_dim = tracking_dim
        self.tf_ratio = tf_ratio
        self.EOS_token = EOS_token
        self.bound = bound
        # modules
        self.embedding = embedding
        for p in embedding.parameters() :
            lang_size     = p.data.size(0)
            embedding_dim = p.data.size(1)
        self.gru = nn.GRU(embedding_dim + tracking_dim, tracking_dim)
        self.out = nn.Linear(tracking_dim, lang_size)
        self.dropout = nn.Dropout(dropout)
        
        
    def generateWord(self, query_vector, hidden, current_word_index):
        # update hidden state
        embedded = self.embedding(current_word_index).view(1, 1, -1)
        if query_vector is not None : embedded = torch.cat((query_vector, embedded), dim = 2)
        embedded = self.dropout(embedded)
        _, hidden = self.gru(embedded, hidden)
        # generate next word
        vector = self.out(hidden).squeeze(0)
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
            log_proba, hidden = self.generateWord(decision_vector, hidden, current_word_index) 
            topv, topi = log_proba.data.topk(1)
            log_probas.append(log_proba)
            ni = topi[0][0] # index of current generated word
            if ni == self.EOS_token : # EOS_token
                break
            elif ta is not None : # Teacher forcing
                answer.append(ni)
                if di < ta.size(0) : current_word_index = ta[di].to(self.device)
                else               : break
            else :
                answer.append(ni)
                current_word_index = Variable(torch.LongTensor([[ni]])).to(self.device)
        return answer, log_probas
