
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SmoothAttnDecoder(nn.Module):
    '''Transforms a vector into a sequence of words'''
    def __init__(self, word2vec, attention_dim, hidden_dim,
                 n_layers = 1,
                 dropout = 0.1,
                 bound = 25
                ):
        super(SmoothAttnDecoder, self).__init__()
        T = word2vec.embedding.weight.cpu().detach().numpy()
        # relevant quantities
        self.lang_size   = T.shape[0]
        self.embedd_dim  = T.shape[1]
        self.hidden_dim  = hidden_dim
        self.n_layers    = n_layers
        self.bound       = bound
        self.temperature = 10
        # embedding module
        self.embedding = nn.Linear(self.lang_size, self.embedd_dim, bias = False)
        self.embedding.weight = nn.Parameter(torch.FloatTensor(T.transpose()))
        for param in self.embedding.parameters() : param.requires_grad = False
        # other modules
        self.word2vec = word2vec
        self.gru = nn.GRU(self.embedd_dim, 
                          hidden_dim, 
                          n_layers, 
                          dropout = dropout, 
                          batch_first = True)
        self.attn = Attention(attention_dim, hidden_dim, dropout = dropout)
        self.out = nn.Linear(attention_dim + hidden_dim, self.lang_size)
        self.act = F.log_softmax
        self.dropout = nn.Dropout(dropout)

    def initWordTensor(self, index_list, device = None) :
        word = torch.zeros((len(index_list), self.lang_size), dtype = torch.float)
        for i, index in enumerate(index_list) : word[i, index] = 1.
        word = Variable(word)                               # size (batch_size, lang_size)
        if device is not None : word = word.to(device)      # size (batch_size, lang_size)
        return word
        
    def generateWord(self, hidden, embeddings, word):
        '''word is a FloatTensor with size (batch_size, lang_size)'''
        # update hidden state
        embedding = self.embedding(word.unsqueeze(1))       # size (batch_size, 1, embedding_dim)
        embedding = self.dropout(embedding)                 # size (batch_size, 1, embedding_dim)
        _, hidden  = self.gru(embedding, hidden)            # size (n_layers, batch_size, embedding_dim)
        # merge with attention
        query = hidden[-1].unsqueeze(1)                     # size (batch_size, 1, embedding_dim)
        query = query.expand(query.size(0), 
                             embeddings.size(1), 
                             query.size(2))                 # size (batch_size, sequence_length, embedding_dim)
        attn, weights = self.attn(embeddings, query)        # size (batch_size, 1, embedding_dim)
        merge = torch.cat([hidden[-1], attn.squeeze(1)], dim = 1) 
        merge = self.dropout(merge)                         # size (batch_size, embedding_dim + hidden_dim)
        # generate next word
        vect = self.out(merge) * self.temperature           # size (batch_size, lang_size)
        return vect, hidden, weights
    
    def forward(self, hidden, embeddings, device = None) :
        answer  = []
        EOS_token = self.word2vec.lang.getIndex('EOS')
        SOS_token = self.word2vec.lang.getIndex('SOS')
        word      = self.initWordTensor([SOS_token], device = device) 
        hidden    = hidden[-self.n_layers:]                 # size (n_layers, 1, hidden_dim)
        # word generation
        for t in range(self.bound) :
            # compute next word proba
            vect, hidden, attn = self.generateWord(hidden, embeddings, word)
            # compute next word index
            word_index = vect.topk(1, dim = 1)[1].item()
            # stopping criterion
            if word_index == EOS_token : break
            else : 
                answer.append(word_index)
                word = F.softmax(vect, dim = 1)             # size (1, lang_size)
            # cumulate attention weights
            if t == 0 : weights = attn
            else      : weights = torch.cat((weights, attn), dim = 1) # size(1, output_length, input_length)
        return answer, weights
