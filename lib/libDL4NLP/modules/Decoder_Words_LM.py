
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class LMWordsDecoder(nn.Module):
    '''Transforms a vector into a sequence of words'''
    def __init__(self, 
                 device, 
                 language_model, 
                 hidden_dim, 
                 tracking_dim, 
                 dropout = 0.1,
                 tf_ratio = 1,
                 bound = 25
                ):
        super(LMWordsDecoder, self).__init__()
        # relevant quantities
        self.device = device
        self.hidden_dim = hidden_dim
        self.tracking_dim = tracking_dim
        self.tf_ratio = tf_ratio
        self.lm_ratio = 0.25
        self.bound = bound
        self.pos = 0
        self.max_pos = 3
        # modules
        self.language_model = language_model.to(self.device)
        for param in self.language_model.parameters() : param.requires_grad = False 
        self.embedding = self.language_model.embedding
        for p in self.embedding.parameters() :
            lang_size     = p.data.size(0)
            embedding_dim = p.data.size(1)
        self.gru = nn.GRU(embedding_dim + hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, lang_size)
        self.dropout = nn.Dropout(dropout)
        
        
    def generateWord(self, query_vector, hidden, hidden_lm, current_word_index, current_word_index_lm):
        # update hidden state
        embedded = self.embedding(current_word_index).view(1, 1, -1)
        if query_vector is not None : embedded = torch.cat((query_vector, embedded), dim = 2)
        embedded = self.dropout(embedded)
        _, hidden = self.gru(embedded, hidden)
        # generate next word
        vector = self.out(hidden).squeeze(0)
        log_proba = F.log_softmax(vector, dim = 1)
        # Language Model contribution
        log_proba_lm, hidden_lm = self.language_model.generateWord(current_word_index_lm, hidden_lm)
        return log_proba + (self.pos/self.max_pos) * self.lm_ratio * log_proba_lm, hidden, hidden_lm
    
    
    def forward(self, query_words, query_vector, decision_vector, target_answer) :
        log_probas = []
        answer = []
        di = 0
        ta = target_answer if random.random() < self.tf_ratio else None
        current_word_index    = Variable(torch.LongTensor([[0]])).to(self.device) # SOS_token
        current_word_index_lm = Variable(torch.LongTensor([[0]])).to(self.device) # SOS_token
        hidden    = self.dropout(decision_vector)
        hidden_lm = None
        for di in range(self.bound) :
            self.pos = min(di, self.max_pos)
            log_proba, hidden, hidden_lm = self.generateWord(query_vector, 
                                                             hidden, 
                                                             hidden_lm, 
                                                             current_word_index, 
                                                             current_word_index_lm)
            topv, topi = log_proba.data.topk(1)
            log_probas.append(log_proba)
            ni = topi[0][0] # index of current generated word
            if ni == 1 : # EOS_token
                break
            elif ta is not None : # Teacher forcing
                answer.append(ni)
                if di < ta.size(0) :
                    current_word_index    = ta[di].view(-1, 1).to(self.device)
                    current_word_index_lm = target_answer[di].view(-1, 1).to(self.device) if di < target_answer.size(0) else \
                                            target_answer[-1].view(-1, 1).to(self.device)
                else :
                    break
            else :
                answer.append(ni)
                current_word_index    = Variable(torch.LongTensor([[ni]])).to(self.device)
                if target_answer is not None and di < target_answer.size(0): 
                    current_word_index_lm = target_answer[di].view(-1, 1).to(self.device)
                else :
                    current_word_index_lm = Variable(torch.LongTensor([[ni]])).to(self.device)
        return answer, log_probas
