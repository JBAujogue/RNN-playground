
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

from libDL4NLP.modules import RecurrentEncoder, Attention, MultiHeadAttention, HAN


#-------------------------------------------------------------------#
#                          Text Classifier                          #
#-------------------------------------------------------------------#


class TextClassifier(nn.Module) :
    def __init__(self, device, tokenizer, word2vec, 
                 hidden1_dim = 100,
                 hidden2_dim = 100,
                 n1_layers = 1, 
                 n2_layers = 1,
                 hops = 1, 
                 share = True,
                 transf = False,
                 n_class = 2, 
                 dropout = 0, 
                 class_weights = None, 
                 optimizer = optim.SGD
                ):
        super(TextClassifier, self).__init__()

        # embedding
        self.bin_mode  = (n_class == 'binary')
        self.tokenizer = tokenizer
        self.word2vec  = word2vec
        self.context   = RecurrentEncoder(self.word2vec.output_dim, 
                                          hidden1_dim, 
                                          n1_layers, 
                                          dropout, 
                                          bidirectional = True)
        self.attention = HAN(embedding_dim = self.context.output_dim,
                             hidden_dim = hidden2_dim,
                             query_dim = 0, # self.context.output_dim,
                             n_layers = n2_layers,
                             hops = hops,
                             share = share,
                             transf = transf,
                             dropout = dropout)
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
    
    def showAttention(self, words, attn) :
        for i in range(attn.size(1)) :
            fig, ax  = plt.subplots()
            im       = heatmap(np.array(attn[:, i, :].data.cpu().numpy()),  [' '], words, ax=ax, cmap="YlGn", cbarlabel="harvest [t/year]")
            texts    = annotate_heatmap(im, valfmt="{x:.2f}")
            fig.tight_layout()
            plt.show()
        return
        
    def forward(self, text, show_attention = False) :
        '''classifies a sentence as string'''
        sentences        = self.tokenizer(text)
        embeddings       = [self.word2vec(words, self.device).squeeze(0) for words in sentences] # list of tensors of size (1, n_words, embedding_dim)
        embeddings       = nn.utils.rnn.pad_sequence(embeddings, batch_first = True, padding_value = 0)  # size (n_sentences, n_words, embedding_dim)
        hiddens, _       = self.context(embeddings, enforce_sorted = False) # size (n_sentences, n_words, embedding_dim)
        attended, w1, w2 = self.attention(hiddens)  # size (1, 1, embedding_dim)
        if self.bin_mode : prediction = self.act(self.out(attended).view(-1)).data.topk(1)[0].item()
        else             : prediction = self.act(self.out(attended.squeeze(1)), dim = 1).data.topk(1)[1].item()
        if show_attention : self.showAttention(words, atn)
        return prediction
    
    def generatePaddedTexts(self, texts) :
        padded_data = []
        for text, label in texts :
            pack0 = self.tokenizer(text)
            pack0 = [[self.word2vec.lang.getIndex(w) for w in words] for words in pack0]
            pack0 = [[w for w in words if w is not None] for words in pack0]
            lengths = torch.tensor([len(p) for p in pack0])               # size = (text_length) 
            pack0 = list(itertools.zip_longest(*pack0, fillvalue = self.word2vec.lang.getIndex('PADDING_WORD')))
            pack0 = Variable(torch.LongTensor(pack0).transpose(0, 1))     # size = (text_length, max_length)
            pack1 = [label]
            if self.bin_mode : pack1 = Variable(torch.FloatTensor(pack1)) # size = (1) 
            else             : pack1 = Variable(torch.LongTensor(pack1))  # size = (1) 
            padded_data.append([[pack0, lengths], pack1])
        return padded_data
    
    def compute_accuracy(self, texts) :
        #batches = self.generatePaddedTexts(texts)
        score = 0
        for text, label in texts :
            predict = self(text)
            if self.bin_mode : score += (abs(label - predict) < 0.5)
            else             : score += (label == predict)
        return score * 100 / len(texts)
    
    
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
            embeddings       = self.word2vec.embedding(batch[0].to(self.device))
            hiddens, _       = self.context(embeddings, lengths = batch[1].to(self.device), enforce_sorted = False)
            attended, w1, w2 = self.attention(hiddens)
            if self.bin_mode : return self.out(attended).view(-1)
            else             : return F.log_softmax(self.out(attended.squeeze(1)))

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
            log_probs = computeLogProbs(batch[0])
            targets = batch[1].to(self.device).view(-1)
            loss    = self.criterion(log_probs, targets)
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
