
import math
import time
import unicodedata
import re
import random
import copy

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker #, FuncFormatter
#%matplotlib inline

import numpy as np
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from libDL4NLP.utils import Lang


#-------------------------------------------------------------------#
#                       Word Embedding model                        #
#-------------------------------------------------------------------#


class Word2Vec(nn.Module) :
    def __init__(self, lang, T = 100):
        super(Word2Vec, self).__init__()
        self.lang = lang
        if type(T) == int :
            self.embedding = nn.Embedding(lang.n_words, T)  
        else :
            self.embedding = nn.Embedding(T.shape[0], T.shape[1])
            self.embedding.weight = nn.Parameter(torch.FloatTensor(T))
            
        self.output_dim = self.lookupTable().shape[1]
        self.sims = None
        
    def lookupTable(self) :
        return self.embedding.weight.cpu().detach().numpy()
        
    def computeSimilarities(self) :
        T = normalize(self.lookupTable(), norm = 'l2', axis = 1)
        self.sims = np.matmul(T, T.transpose())
        return

    def most_similar(self, word, bound = 10) :
        if word not in self.lang.word2index : return
        if self.sims is None : self.computeSimilarities()
        index = self.lang.word2index[word]
        coefs = self.sims[index]
        indices = coefs.argsort()[-bound -1 :-1]
        output = [(self.lang.index2word[i], coefs[i]) for i in reversed(indices)]
        return output
    
    def wv(self, word) :
        return self.lookupTable()[self.lang.getIndex(word)]
    
    def addWord(self, word, vector = None) :
        if word not in self.lang.word2index :
            self.lang.addWord(word)
            T = self.lookupTable()
            v = np.random.rand(1, T.shape[1]) if vector is None else vector
            updated_T = np.concatenate((T, v), axis = 0)
            self.embedding = nn.Embedding(updated_T.shape[0], updated_T.shape[1])
            self.embedding.weight = nn.Parameter(torch.FloatTensor(updated_T))
        return
    
    def freeze(self) :
        for param in self.embedding.parameters() : param.requires_grad = False
        return self
    
    def unfreeze(self) :
        for param in self.embedding.parameters() : param.requires_grad = True
        return self
    
    def forward(self, words, device = None) :
        '''Transforms a list of n words into a torch.FloatTensor of size (1, n, emb_dim)'''
        indices  = [self.lang.getIndex(w) for w in words]
        indices  = [[i for i in indices if i is not None]]
        variable = Variable(torch.LongTensor(indices)) # size (1, n)
        if device is not None : variable = variable.to(device)
        tensor   = self.embedding(variable)            # size (1, n, embedding_dim)
        return tensor



class Word2VecConnector(nn.Module) :
    '''A Pytorch module wrapping a FastText word2vec model'''
    def __init__(self, word2vec) :
        super(Word2VecConnector, self).__init__()
        self.word2vec = word2vec
        self.twin = Word2Vec(lang = Lang([list(word2vec.wv.index2word)], base_tokens = []), T = word2vec.wv.vectors)
        self.twin.addWord('PADDING_WORD')
        self.twin.addWord('UNK')
        self.twin = self.twin.freeze()
        
        self.lang       = self.twin.lang
        self.embedding  = self.twin.embedding
        self.output_dim = self.twin.output_dim
        
    def lookupTable(self) :
        return self.word2vec.wv.vectors
        
    def forward(self, words, device = None) :
        '''Transforms a sequence of n words into a Torch FloatTensor of size (1, n, emb_dim)'''
        try :
            embeddings = Variable(torch.Tensor(self.word2vec[words])).unsqueeze(0)
            if device is not None : embeddings = embeddings.to(device)
        except :
            embeddings = self.twin(words, device)
        return embeddings


#-------------------------------------------------------------------#
#                         training shell                            #
#-------------------------------------------------------------------#



class Word2VecShell(nn.Module):
    '''Word2Vec model :
        - sg = 0 yields CBOW training procedure
        - sg = 1 yields Skip-Gram training procedure
    '''
    def __init__(self, word2vec, device, sg = 0, context_size = 5, hidden_dim = 150, 
                 criterion = nn.NLLLoss(size_average = False), optimizer = optim.SGD):
        super(Word2VecShell, self).__init__()
        self.device = device
        
        # core of Word2Vec
        self.word2vec = word2vec
        
        # training layers
        self.input_n_words  = (2 * context_size if sg == 0 else 1)
        self.output_n_words = (1 if sg == 0 else 2 * context_size)
        self.linear_1  = nn.Linear(self.input_n_words * word2vec.embedding.weight.size(1), self.output_n_words * hidden_dim)
        self.linear_2  = nn.Linear(hidden_dim, word2vec.lang.n_words)
        
        # training tools
        self.sg = sg
        self.criterion = criterion
        self.optimizer = optimizer
        
        # load to device
        self.to(device)
        
    def forward(self, batch):
        '''Transforms a batch of Ngrams of size (batch_size, input_n_words)
           Into log probabilities of size (batch_size, lang.n_words, output_n_words)
           '''
        batch = batch.to(self.device)                 # size = (batch_size, self.input_n_words)
        embed = self.word2vec.embedding(batch)        # size = (batch_size, self.input_n_words, embedding_dim)
        embed = embed.view((batch.size(0), -1))       # size = (batch_size, self.input_n_words * embedding_dim)
        out = self.linear_1(embed)                    # size = (batch_size, self.output_n_words * hidden_dim) 
        out = out.view((batch.size(0),self.output_n_words, -1))
        out = F.relu(out)                             # size = (batch_size, self.output_n_words, hidden_dim)                                         
        out = self.linear_2(out)                      # size = (batch_size, self.output_n_words, lang.n_words)
        out = torch.transpose(out, 1, 2)              # size = (batch_size, lang.n_words, self.output_n_words)
        log_probs = F.log_softmax(out, dim = 1)       # size = (batch_size, lang.n_words, self.output_n_words)
        return log_probs
    
    def generatePackedNgrams(self, corpus, context_size = 5, batch_size = 32, seed = 42) :
        # generate Ngrams
        data = []
        for text in corpus :
            text = [w for w in text if w in self.word2vec.lang.word2index]
            text = ['SOS' for i in range(context_size)] + text + ['EOS' for i in range(context_size)]
            for i in range(context_size, len(text) - context_size):
                context = text[i-context_size : i] + text[i+1 : i+context_size+1]
                word = text[i]
                data.append([word, context])
        # pack Ngrams into mini_batches
        random.seed(seed)
        random.shuffle(data)
        packed_data = []
        for i in range(0, len(data), batch_size):
            pack0 = [el[0] for el in data[i:i + batch_size]]
            pack0 = [[self.word2vec.lang.getIndex(w)] for w in pack0]
            pack0 = Variable(torch.LongTensor(pack0)) # size = (batch_size, 1)
            pack1 = [el[1] for el in data[i:i + batch_size]]
            pack1 = [[self.word2vec.lang.getIndex(w) for w in context] for context in pack1]
            pack1 = Variable(torch.LongTensor(pack1)) # size = (batch_size, 2*context_size)   
            if   self.sg == 1 : packed_data.append([pack0, pack1])
            elif self.sg == 0 : packed_data.append([pack1, pack0])
            else :
                print('A problem occured')
                pass
        return packed_data
    
    def train(self, ngrams, iters = None, epochs = None, lr = 0.025, random_state = 42,
              print_every = 10, compute_accuracy = False):
        """Performs training over a given dataset and along a specified amount of loop
        s"""
        def asMinutes(s):
            m = math.floor(s / 60)
            s -= m * 60
            return '%dm %ds' % (m, s)

        def timeSince(since, percent):
            now = time.time()
            s = now - since
            rs = s/percent - s
            return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

        def computeAccuracy(log_probs, targets) :
            accuracy = 0
            acc = sum([log_probs[i, :, j].data.topk(1)[1].item() == targets[i, j].item() 
                       for i in range(targets.size(0)) 
                       for j in range(targets.size(1))])
            return (acc * 100) / (targets.size(0) * targets.size(1))

        def printScores(start, iter, iters, tot_loss, tot_loss_words, print_every, compute_accuracy) :
            avg_loss = tot_loss / print_every
            avg_loss_words = tot_loss_words / print_every
            if compute_accuracy : print(timeSince(start, iter / iters) + ' ({} {}%) loss : {:.3f}  accuracy : {:.1f} %'.format(iter, int(iter / iters * 100), avg_loss, avg_loss_words))
            else                : print(timeSince(start, iter / iters) + ' ({} {}%) loss : {:.3f}                     '.format(iter, int(iter / iters * 100), avg_loss))
            return 0, 0

        def trainLoop(couple, optimizer, compute_accuracy = False):
            """Performs a training loop, with forward pass and backward pass for gradient optimisation."""
            optimizer.zero_grad()
            self.zero_grad()
            log_probs = self(couple[0])           # size = (batch_size, agent.output_n_words, agent.lang.n_words)
            targets   = couple[1].to(self.device) # size = (batch_size, agent.output_n_words)
            loss      = self.criterion(log_probs, targets)
            loss.backward()
            optimizer.step() 
            accuracy = computeAccuracy(log_probs, targets) if compute_accuracy else 0
            return float(loss.item() / (targets.size(0) * targets.size(1))), accuracy
        
        # --- main ---
        np.random.seed(random_state)
        start = time.time()
        optimizer = self.optimizer([param for param in self.parameters() if param.requires_grad == True], lr = lr)
        tot_loss = 0  
        tot_loss_words = 0
        if epochs is None :
            for iter in range(1, iters + 1):
                couple = random.choice(ngrams)
                loss, loss_words = trainLoop(couple, optimizer, compute_accuracy)
                tot_loss += loss
                tot_loss_words += loss_words      
                if iter % print_every == 0 : 
                    tot_loss, tot_loss_words = printScores(start, iter, iters, tot_loss, tot_loss_words, print_every, compute_accuracy)
        else :
            iter = 0
            iters = len(ngrams) * epochs
            for epoch in range(1, epochs + 1):
                print('epoch ' + str(epoch))
                np.random.shuffle(ngrams)
                for couple in ngrams :
                    loss, loss_words = trainLoop(couple, optimizer, compute_accuracy)
                    tot_loss += loss
                    tot_loss_words += loss_words 
                    iter += 1
                    if iter % print_every == 0 : 
                        tot_loss, tot_loss_words = printScores(start, iter, iters, tot_loss, tot_loss_words, print_every, compute_accuracy)
        return
