
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

from libDL4NLP.modules import RecurrentEncoder, SelfAttention, MultiHeadSelfAttention


#-------------------------------------------------------------------#
#                       Sentence Classifier                         #
#-------------------------------------------------------------------#


class SentenceClassifier(nn.Module) :
    def __init__(self, device, tokenizer, word2vec, 
                 hidden_dim = 100, 
                 n_layers = 1, 
                 n_attn_heads = 1, 
                 attn_penalization = None,
                 n_class = 2, 
                 dropout = 0, 
                 class_weights = None, 
                 optimizer = optim.SGD
                 ):
        super(SentenceClassifier, self).__init__()
        
        # embedding
        self.bin_mode  = (n_class == 'binary')
        self.tokenize  = tokenizer
        self.word2vec  = word2vec
        self.context   = RecurrentEncoder(self.word2vec.output_dim, hidden_dim, n_layers, dropout, bidirectional = True)
        self.attention = MultiHeadSelfAttention(self.context.output_dim, 
                                                n_head = n_attn_heads, penalization = attn_penalization, dropout = dropout)
        self.out       = nn.Linear(self.attention.output_dim, (1 if self.bin_mode else n_class))
        self.act       = F.sigmoid if self.bin_mode else F.softmax
        
        # optimizer
        if self.bin_mode : self.criterion = nn.BCEWithLogitsLoss(size_average = False)
        else             : self.criterion = nn.NLLLoss(size_average = False, weight = class_weights)
        self.optimizer = optimizer
        
        # load to device
        self.device = device
        self.to(device)
        
    def nbParametres(self) :
        return sum([p.data.nelement() for p in self.parameters() if p.requires_grad == True])
    
    # main method
    def forward(self, sentence, attention_method = None) :
        '''classifies a sentence as string'''
        words      = self.tokenize(sentence)
        embeddings = self.word2vec(words, self.device)
        hiddens, _ = self.context(embeddings) 
        out, attn  = self.attention(hiddens)
        attn       = np.array(attn[0].data.cpu().numpy()) # size (n_heads, input_length)
        labels     = ['head '+str(i+1) for i in range(attn.shape[0])] if attn.shape[0] > 1 else ['']
        # show attention
        if attention_method is not None : attention_method(attn, labels, words)
        # compute prediction
        if self.bin_mode : pred = self.act(self.out(out).view(-1)).data.topk(1)[0].item()
        else             : pred = self.act(self.out(out.squeeze(1)), dim = 1).data.topk(1)[1].item()
        return pred
    
    # load data
    def generatePackedSentences(self, sentences, batch_size = 32) :
        sentences.sort(key = lambda s: len(self.tokenize(s[0])), reverse = True)
        packed_data = []
        for i in range(0, len(sentences), batch_size) :
            pack0 = [self.tokenize(s[0]) for s in sentences[i:i + batch_size]]
            pack0 = [[self.word2vec.lang.getIndex(w) for w in words] for words in pack0]
            pack0 = [[w for w in words if w is not None] for words in pack0]
            pack0.sort(key = len, reverse = True)
            lengths = torch.tensor([len(p) for p in pack0])               # size (batch_size) 
            pack0 = list(itertools.zip_longest(*pack0, fillvalue = self.word2vec.lang.getIndex('PADDING_WORD')))
            pack0 = Variable(torch.LongTensor(pack0).transpose(0, 1))     # size (batch_size, max_length)
            pack1 = [[el[1]] for el in sentences[i:i + batch_size]]
            if self.bin_mode : pack1 = Variable(torch.FloatTensor(pack1)) # size (batch_size) 
            else             : pack1 = Variable(torch.LongTensor(pack1))  # size (batch_size) 
            packed_data.append([[pack0, lengths], pack1])
        return packed_data
    
    # compute model perf
    def compute_accuracy(self, sentences, batch_size = 32) :
        batches = self.generatePackedSentences(sentences, batch_size)
        score = 0
        for batch, target in batches :
            embeddings  = self.word2vec.embedding(batch[0].to(self.device))
            hiddens, _  = self.context(embeddings, lengths = batch[1].to(self.device))
            attended, _ = self.attention(hiddens)
            if self.bin_mode : 
                vects  = self.out(attended).view(-1)
                target = target.to(self.device).view(-1)
                score += sum(torch.abs(target - self.act(vects)) < 0.5).item()
            else : 
                log_probs = F.log_softmax(self.out(attended.squeeze(1)))
                target    = target.to(self.device).view(-1)
                score    += sum([target[i].item() == log_probs[i].data.topk(1)[1].item() for i in range(target.size(0))])
        return score * 100 / len(sentences)
    
    # fit model
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
            embeddings  = self.word2vec.embedding(batch[0].to(self.device))
            hiddens, _  = self.context(embeddings, lengths = batch[1].to(self.device))
            attended, atn, penal = self.attention(hiddens, penal = True, device = self.device)
            if self.bin_mode : return self.out(attended).view(-1), penal
            else             : return F.log_softmax(self.out(attended.squeeze(1))), penal

        def computeAccuracy(log_probs, targets) :
            if self.bin_mode : return sum(torch.abs(targets - self.act(log_probs)) < 0.5).item() * 100 / targets.size(0)
            else             : return sum([targets[i].item() == log_probs[i].data.topk(1)[1].item() for i in range(targets.size(0))]) * 100 / targets.size(0)
            
        def printScores(start, iter, iters, tot_loss, tot_loss_words, print_every, compute_accuracy) :
            avg_loss = tot_loss / print_every
            avg_loss_words = tot_loss_words / print_every
            if compute_accuracy : print(timeSince(start, iter / iters) + ' ({} {}%) loss : {:.3f}  accuracy : {:.1f} %'.format(iter, int(iter / iters * 100), avg_loss, avg_loss_words))
            else                : print(timeSince(start, iter / iters) + ' ({} {}%) loss : {:.3f}                     '.format(iter, int(iter / iters * 100), avg_loss))
            return 0, 0

        def trainLoop(batch, optimizer, compute_accuracy = True):
            """Performs a training loop, with forward pass, backward pass and weight update."""
            optimizer.zero_grad()
            self.zero_grad()
            log_probs, penal = computeLogProbs(batch[0])
            targets = batch[1].to(self.device).view(-1)
            loss    = self.criterion(log_probs, targets)
            if penal is not None and penal.item() > 10 : loss = loss + penal
            loss.backward()
            optimizer.step() 
            accuracy = computeAccuracy(log_probs, targets) if compute_accuracy else 0
            return float(loss.item() / targets.size(0)), accuracy
        
        # --- main ---
        self.train()
        np.random.seed(random_state)
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
