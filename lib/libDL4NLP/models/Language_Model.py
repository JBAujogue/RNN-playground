
import math
import time
import unicodedata
import re
import random
import copy
import itertools

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker #, FuncFormatter
#%matplotlib inline

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from libDL4NLP.modules import RecurrentEncoder


#-------------------------------------------------------------------#
#                        Language model                             #
#-------------------------------------------------------------------#



class LanguageModel(nn.Module) :
    def __init__(self, device, tokenizer, word2vec, 
                 hidden_dim = 100, 
                 n_layer = 1, 
                 dropout = 0, 
                 class_weights = None, 
                 optimizer = optim.SGD
                 ):
        super().__init__()
        
        # layers
        self.tokenizer = tokenizer
        self.word2vec  = word2vec
        self.context   = RecurrentEncoder(
            emb_dim = self.word2vec.out_dim, 
            hid_dim = hidden_dim, 
            n_layer = n_layer, 
            dropout = dropout, 
            bidirectional = False)
        self.out       = nn.Linear(self.context.out_dim, self.word2vec.lang.n_words)
        self.act       = F.softmax
        
        # optimizer
        self.criterion = nn.NLLLoss(size_average = False, weight = class_weights)
        self.optimizer = optimizer
        
        # load to device
        self.device = device
        self.to(device)
        
    def nbParametres(self) :
        return sum([p.data.nelement() for p in self.parameters() if p.requires_grad == True])
    
    def forward(self, 
                sentence = '.', 
                hidden = None, 
                limit = 10, 
                color_code = '\033[94m'):
        # init variables
        words  = self.tokenizer(sentence)
        result = words + [color_code]
        hidden, count, stop = None, 0, False
        while not stop :
            # compute probs
            embeddings = self.word2vec(words, self.device)
            _, hidden  = self.context(embeddings, lengths = None, hidden = hidden) # WARNING : dim = (n_layers, batch_size, hidden_dim)
            probs      = self.act(self.out(hidden[-1]), dim = 1).view(-1)
            # get predicted word
            topv, topi = probs.data.topk(1)
            words = [self.word2vec.lang.index2word[topi.item()]]
            result += words
            # stopping criterion
            count += 1
            if count == limit or words == [limit] or count == 50 : stop = True
        print(' '.join(result + ['\033[0m']))
        return 
    
    def generatePackedSentences(self, sentences, batch_size = 32, lengths = [5, 10, 15]) :
        sentences = [s[i: i+j] \
                     for s in sentences \
                     for j in lengths \
                     for i in range(len(s)-j)]
        sentences.sort(key = lambda s: len(s), reverse = True)
        packed_data = []
        for i in range(0, len(sentences), batch_size) :
            pack0 = sentences[i:i + batch_size]
            pack0 = [[self.word2vec.lang.getIndex(w) for w in s] for s in pack0]
            pack0 = [[w for w in words if w is not None] for words in pack0]
            pack0.sort(key = len, reverse = True)
            pack1 = Variable(torch.LongTensor([s[-1] for s in pack0]))
            pack0 = [s[:-1] for s in pack0]
            lengths = torch.tensor([len(p) for p in pack0]) # size = (batch_size) 
            pack0 = list(itertools.zip_longest(*pack0, fillvalue = self.word2vec.lang.getIndex('PADDING_WORD')))
            pack0 = Variable(torch.LongTensor(pack0).transpose(0, 1))   # size = (batch_size, max_length) 
            packed_data.append([[pack0, lengths], pack1])
        return packed_data
    
    def fit(self, batches, iters = None, epochs = None, lr = 0.025, random_state = 42,
              print_every = 10, compute_accuracy = True):
        """Performs training over a given dataset and along a specified amount of loops"""
        def asMinutes(s):
            m = math.floor(s / 60)
            s -= m * 60
            return '%dm %ds' % (m, s)

        def timeSince(since, percent):
            now = time.time()
            s = now - since
            rs = s/percent - s
            return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
        
        def computeLogProbs(batch) :
            embeddings = self.word2vec.embedding(batch[0].to(self.device))
            _, hidden  = self.context(embeddings, lengths = batch[1].to(self.device)) # WARNING : dim = (n_layers, batch_size, hidden_dim)
            log_probs  = F.log_softmax(self.out(hidden[-1]), dim = 1)   # dim = (batch_size, lang_size)
            return log_probs

        def computeAccuracy(log_probs, targets) :
            return sum([targets[i].item() == log_probs[i].data.topk(1)[1].item() for i in range(targets.size(0))]) * 100 / targets.size(0)

        def printScores(start, iter, iters, tot_loss, tot_loss_words, print_every, compute_accuracy) :
            avg_loss = tot_loss / print_every
            avg_loss_words = tot_loss_words / print_every
            if compute_accuracy : print(timeSince(start, iter / iters) + ' ({} {}%) loss : {:.3f}  accuracy : {:.1f} %'.format(iter, int(iter / iters * 100), avg_loss, avg_loss_words))
            else                : print(timeSince(start, iter / iters) + ' ({} {}%) loss : {:.3f}                     '.format(iter, int(iter / iters * 100), avg_loss))
            return 0, 0

        def trainLoop(batch, optimizer, compute_accuracy = True):
            """Performs a training loop, with forward pass, backward pass and weight update."""
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            self.zero_grad()
            log_probs = computeLogProbs(batch[0])
            targets   = batch[1].to(self.device).view(-1)
            loss      = self.criterion(log_probs, targets)
            loss.backward()
            optimizer.step() 
            accuracy = computeAccuracy(log_probs, targets) if compute_accuracy else 0
            return float(loss.item() / targets.size(0)), accuracy
        
        # --- main ---
        self.train()
        random.seed(random_state)
        start = time.time()
        optimizer = self.optimizer([param for param in self.parameters() if param.requires_grad == True], lr = lr)
        tot_loss = 0  
        tot_acc  = 0
        if epochs is None :
            for iter in range(1, iters + 1):
                batch = random.choice(batches)
                loss, acc = trainLoop(batch, optimizer, compute_accuracy)
                tot_loss += loss
                tot_acc += acc      
                if iter % print_every == 0 : 
                    tot_loss, tot_acc = printScores(start, iter, iters, tot_loss, tot_acc, print_every, compute_accuracy)
        else :
            iter = 0
            iters = len(batches) * epochs
            for epoch in range(1, epochs + 1):
                print('epoch ' + str(epoch))
                np.random.shuffle(batches)
                for batch in batches :
                    loss, acc = trainLoop(batch, optimizer, compute_accuracy)
                    tot_loss += loss
                    tot_acc += acc 
                    iter += 1
                    if iter % print_every == 0 : 
                        tot_loss, tot_acc = printScores(start, iter, iters, tot_loss, tot_acc, print_every, compute_accuracy)
        return
