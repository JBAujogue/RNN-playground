
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SmoothWordsDecoder(nn.Module):
    '''Transforms a vector into a sequence of words'''
    def __init__(self, 
                 device,
                 embedding, 
                 hidden_dim, 
                 tracking_dim, 
                 dropout = 0.1,
                 tf_ratio = 1,
                 bound = 25
                ):
        super(SmoothWordsDecoder, self).__init__()
        # relevant quantities
        self.device = device
        self.hidden_dim = hidden_dim
        self.tracking_dim = tracking_dim
        self.tf_ratio = tf_ratio
        self.bound = bound
        for p in embedding.parameters() :
            lang_size     = p.data.size(0)
            embedding_dim = p.data.size(1)
        # modules
        self.enbedding = nn.Linear((lang_size, embedding_dim), bias = False)
        # TODO : put embedding weights into the self.embedding layer
        self.gru = nn.GRU(embedding_dim + tracking_dim, tracking_dim)
        self.out = nn.Linear(tracking_dim, lang_size)
        self.dropout = nn.Dropout(dropout)
        
        
    def generateWord(self, query_vector, hidden, embedded):
        # update hidden state
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
        current_word_index = Variable(torch.LongTensor(1, 1, self.lang_size)).to(self.device)
        current_word_index.zero_()
        current_word_index = Variable(torch.LongTensor([[0]])).to(self.device) # SOS_token
        current_embedded_word = self.embedding(current_word_index).view(1, 1, -1)
        hidden = self.dropout(decision_vector)
        for di in range(self.bound) :
            log_proba, hidden = self.generateWord(decision_vector, hidden, current_word_index) 
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
                current_embedded_word = self.embedding(current_word_index).view(1, 1, -1)
        return answer, log_probas
