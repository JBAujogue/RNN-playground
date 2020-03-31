
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from . import Attention, AdditiveAttention 


class AttnDecoder(nn.Module):
    '''Transforms a vector into a sequence of words'''
    def __init__(self, word2vec, attention_dim, hidden_dim,
                 n_layers = 1,
                 dropout = 0.1,
                 bound = 25
                ):
        super(AttnDecoder, self).__init__()
        # relevant quantities
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bound = bound
        # modules
        self.word2vec = word2vec
        self.gru = nn.GRU(word2vec.output_dim, 
                          hidden_dim, 
                          n_layers, 
                          dropout = dropout, 
                          batch_first = True)
        self.attn = Attention(attention_dim, hidden_dim, dropout = dropout)
        self.out = nn.Linear(attention_dim + hidden_dim, word2vec.lang.n_words)
        self.act = F.log_softmax
        self.dropout = nn.Dropout(dropout)
        
    def generateWord(self, hidden, embeddings, word_index):
        # update hidden state
        embedding = self.word2vec.embedding(word_index) # size (batch_size, 1, embedding_dim)
        embedding = self.dropout(embedding)
        _, hidden = self.gru(embedding, hidden)         # size (n_layers, batch_size, embedding_dim)
        # merge with attention
        query = hidden[-1].unsqueeze(1)                 # size (batch_size, 1, embedding_dim)
        attn, weights = self.attn(embeddings, query)    # size (batch_size, 1, embedding_dim)
        merge = torch.cat([hidden[-1], attn.squeeze(1)], dim = 1) 
        merge = self.dropout(merge)                     # size (batch_size, embedding_dim + hidden_dim)
        # generate next word
        log_prob = self.out(merge)                      # size (batch_size, lang_size)
        log_prob = self.act(log_prob, dim = 1)          # size (batch_size, lang_size)
        return log_prob, hidden, weights
    
    def forward(self, hidden, embeddings, device = None) :
        answer  = []
        EOS_token  = self.word2vec.lang.getIndex('EOS')
        word = self.word2vec.lang.getIndex('SOS')
        word = Variable(torch.LongTensor([[word]])) # size (1)
        hidden = hidden[-self.n_layers:]
        for t in range(self.bound) :
            # compute next word
            if device is not None : word = word.to(device) # size (1)
            log_prob, hidden, atn = self.generateWord(hidden, embeddings, word)
            word = log_prob.topk(1, dim = 1)[1].view(1, 1)
            # add to output
            if word.item() == EOS_token : break
            else : answer.append(word.item())
            # cumulate attention weights
            if t == 0 : weights = atn
            else      : weights = torch.cat((weights, atn), dim = 1) # size(1, output_length, input_length)
        return answer, weights


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
