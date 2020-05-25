
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
                 n_layer = 1, 
                 n_attn_heads = 1, 
                 attn_penalization = False,
                 n_class = 2, 
                 dropout = 0, 
                 class_weights = None, 
                 optimizer = optim.SGD
                 ):
        super().__init__()
        
        # relevant quantities
        self.bin_mode  = (n_class == 'binary')
        self.opc       = (n_attn_heads == 'opc' and n_class != 'binary')
        
        # layers
        self.tokenize  = tokenizer
        self.word2vec  = word2vec
        self.context   = RecurrentEncoder(
            emb_dim = self.word2vec.out_dim,
            hid_dim = hidden_dim, 
            n_layer = n_layer, 
            dropout = dropout, 
            bidirectional = True)
        self.attention = MultiHeadSelfAttention(
            emb_dim      = self.context.out_dim, 
            n_head       = (n_class if self.opc else n_attn_heads),
            penalization = attn_penalization, 
            dropout      = dropout)
        self.out = nn.Linear(
            (self.context.out_dim if self.opc else self.attention.out_dim), 
            (1 if self.opc or self.bin_mode else n_class))
        self.act = F.sigmoid if self.bin_mode else F.softmax
        
        # optimizer
        if self.bin_mode : self.criterion = nn.BCEWithLogitsLoss(size_average = False)
        else             : self.criterion = nn.NLLLoss(size_average = False, weight = (torch.Tensor(class_weights) if class_weights is not None else None))
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
        # compute prediction
        if self.opc : 
            out    = self.out(out).squeeze(2) # size (1, n_class)
            out    = self.act(out, dim = 1)   # size (1, n_class)
            labels = ['class '+str(i+1) for i in range(attn.shape[0])] if attn.shape[0] > 1 else ['']
            pred   = out.data.topk(1)[1].item()
        elif self.bin_mode : 
            out    = out.view(1, -1) # size (1, n_heads * embedding_dim)
            out    = self.out(out)   # size (1, 1)
            labels = ['head '+str(i+1) for i in range(attn.shape[0])] if attn.shape[0] > 1 else ['']
            pred   = out.item()
        else :
            out    = out.view(1, -1)         # size (1, n_heads * embedding_dim)
            out    = self.act(self.out(out)) # size (1, n_class)
            labels = ['head '+str(i+1) for i in range(attn.shape[0])] if attn.shape[0] > 1 else ['']
            pred   = out.data.topk(1)[1].item()
        # show attention
        if attention_method is not None : attention_method(attn, labels, words)
        return pred
    
    # load data
    def generatePackedSentences(self, sentences, batch_size = 32) :
        def sentence2indices(words) :
            inds  = [self.word2vec.lang.getIndex(w) for w in words]
            inds  = [i for i in inds if i is not None]
            return inds
        
        # --- main ---
        sentences.sort(key = lambda s: len(s[0]), reverse = True)
        packed_data = []
        for i in range(0, len(sentences), batch_size) :
            pack = [[sentence2indices(s[0]), s[1]] for s in sentences[i:i + batch_size]]
            pack.sort(key = lambda s : len(s[0]), reverse = True)
            pack0 = [p[0] for p in pack]
            lengths = torch.tensor([len(p) for p in pack0])               # size (batch_size) 
            pack0 = list(itertools.zip_longest(*pack0, fillvalue = self.word2vec.lang.getIndex('PADDING_WORD')))
            pack0 = Variable(torch.LongTensor(pack0).transpose(0, 1))     # size (batch_size, max_length)
            pack1 = [p[1] for p in pack]
            if self.bin_mode : pack1 = Variable(torch.FloatTensor(pack1)) # size (batch_size) 
            else             : pack1 = Variable(torch.LongTensor(pack1))  # size (batch_size) 
            packed_data.append([[pack0, lengths], pack1])
        return packed_data
    
    # compute model perf
    def compute_accuracy(self, sentences, batch_size = 32) :
        def compute_batch_accuracy(batch, target) :
            torch.cuda.empty_cache()
            batch_size  = batch[0].size(0)
            embeddings  = self.word2vec.embedding(batch[0].to(self.device))
            hiddens, _  = self.context(embeddings, lengths = batch[1].to(self.device))
            out, attn   = self.attention(hiddens) # size (batch_size, n_heads, embedding_dim)
            # compute score
            if self.opc : 
                vect   = self.out(out).squeeze(2)        # size (batch_size, n_class)
                log_ps = F.log_softmax(vect, dim = 1)    # size (batch_size, n_class)
                target = target.to(self.device).view(-1) # size (batch_size)
                score  = sum([target[i].item() == log_ps[i].data.topk(1)[1].item() for i in range(target.size(0))])
            elif self.bin_mode : 
                vect   = out.view(batch_size, -1)        # size (batch_size, n_heads * embedding_dim)
                vect   = self.out(vect).view(-1)         # size (batch_size)
                target = target.to(self.device).view(-1)
                score  = sum(torch.abs(target - self.act(vect)) < 0.5).item()
            else :
                vect   = out.view(batch_size, -1)        # size (batch_size, n_heads * embedding_dim)
                vect   = self.out(vect)                  # size (batch_size, n_class)
                log_ps = F.log_softmax(vect, dim = 1)    # size (batch_size, n_class)
                target = target.to(self.device).view(-1) # size (batch_size)
                score  = sum([target[i].item() == log_ps[i].data.topk(1)[1].item() for i in range(target.size(0))])
            return score
            
        # --- main ---
        batches = self.generatePackedSentences(sentences, batch_size)
        score = 0
        for batch, target in batches : score += compute_batch_accuracy(batch, target)
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
            batch_size = batch[0].size(0)
            embeddings  = self.word2vec.embedding(batch[0].to(self.device))
            hiddens, _  = self.context(embeddings, lengths = batch[1].to(self.device))
            out, attn, penal = self.attention(hiddens, penal = True, device = self.device) # size (batch_size, n_heads, embedding_dim)
            # compute log probs
            if self.opc : 
                vect   = self.out(out).squeeze(2) # size (batch_size, n_class)
                log_ps = F.log_softmax(vect, dim = 1)  # size (batch_size, n_class)
            elif self.bin_mode : 
                vect   = out.view(batch_size, -1)      # size (batch_size, n_heads * embedding_dim)
                log_ps = self.out(vect).view(-1)       # size (batch_size)
            else :
                vect   = out.view(batch_size, -1)      # size (batch_size, n_heads * embedding_dim)
                vect   = self.out(vect)                # size (batch_size, n_class)
                log_ps = F.log_softmax(vect, dim = 1)  # size (batch_size, n_class)
            return log_ps, penal

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
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            self.zero_grad()
            log_probs, penal = computeLogProbs(batch[0])
            targets = batch[1].to(self.device).view(-1)
            loss    = self.criterion(log_probs, targets)
            if penal is not None and penal.item() > 1 : loss = loss + penal * 0.01
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
