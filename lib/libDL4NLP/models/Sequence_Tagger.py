
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
from libDL4NLP.misc    import Highway


#-------------------------------------------------------------------#
#                        Sequence Tagger                            #
#-------------------------------------------------------------------#



class SequenceTagger(nn.Module) :
    def __init__(self, device, tokenizer, word2vec, 
                 hidden_dim = 100, 
                 n_layer = 1, 
                 n_class = 2,
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
            bidirectional = True)
        self.out       = nn.Sequential(
            Highway(self.context.out_dim, dropout), 
            nn.Linear(self.context.out_dim, n_class))
        self.act       = F.softmax
        self.n_class   = n_class
        
        # optimizer
        self.ignore_index_in  = self.word2vec.lang.getIndex('PADDING_WORD')
        self.ignore_index_out = n_class
        self.criterion = nn.NLLLoss(size_average = False, 
                                    ignore_index = self.ignore_index_out, 
                                    weight = class_weights)
        self.optimizer = optimizer
        
        # load to device
        self.device = device
        self.to(device)
        
    def nbParametres(self) :
        return sum([p.data.nelement() for p in self.parameters() if p.requires_grad == True])
    
    def predict_proba(self, words):
        embeddings = self.word2vec.twin(words, self.device) # dim = (1, input_length, hidden_dim)
        hiddens, _ = self.context(embeddings)               # dim = (1, input_length, hidden_dim)
        probs      = self.act(self.out(hiddens), dim = 2)   # dim = (1, input_length, n_class)
        return probs

    # main method
    def forward(self, sentence = '.', color = '\033[94m'):
        def addColor(w1, w2, color) : return color + w2 + '\033[0m' if w1 != w2 else w2
        words  = self.tokenizer(sentence)
        probs  = self.predict_proba(words).squeeze(0) # dim = (input_length,  n_class)
        inds   = [probs[i].data.topk(1)[1].item() for i in range(probs.size(0))]
        return [(w, i) for w, i in zip(words, inds)]

    # load data
    def generatePackedSentences(self, sentences, 
                                batch_size = 32, 
                                mask_ratio = 0,
                                seed = 42) :
        def maskInput(index, b) :
            if   b and random.random() > 0.25 : return self.word2vec.lang.getIndex('UNK')
            elif b and random.random() > 0.10 : return random.choice(list(self.word2vec.twin.lang.word2index.values()))
            else                              : return index
            
        def sentence2indices(words) :
            # convert to indices
            inds = [self.word2vec.lang.getIndex(w) for w in words]
            inds = [i for i in inds if i is not None]
            # apply mask
            mask = [m for m, i in enumerate(inds) if i != self.word2vec.lang.getIndex('UNK')]
            mask = random.sample(mask, k = int(mask_ratio*(len(mask) +1)))
            inds = [maskInput(i, m in mask) for m, i in enumerate(inds)]
            return inds
        
        random.seed(seed)
        sentences.sort(key = lambda s: len(s[0]), reverse = True)
        packed_data = []
        for i in range(0, len(sentences), batch_size) :
            pack = [[sentence2indices(s[0]), s[1]] for s in sentences[i:i + batch_size]]
            pack.sort(key = lambda p : len(p[0]), reverse = True)
            pack0 = [p[0] for p in pack] 
            pack0 = list(itertools.zip_longest(*pack0, fillvalue = self.ignore_index_in))
            pack0 = Variable(torch.LongTensor(pack0).transpose(0, 1)) # size (batch_size, max_length)
            lengths = torch.tensor([len(p[0]) for p in pack])         # size (batch_size)
            pack1 = [p[1] for p in pack]                              # size (batch_size, max_length)
            pack1 = list(itertools.zip_longest(*pack1, fillvalue = self.ignore_index_out))
            pack1 = Variable(torch.LongTensor(pack1).transpose(0, 1)) # size (batch_size, max_length) 
            packed_data.append([[pack0, lengths], pack1])
        return packed_data

    # compute model perf
    def compute_accuracy(self, sentences, batch_size = 32) :
        def computeLogProbs(batch) :
            torch.cuda.empty_cache()
            embeddings = self.word2vec.embedding(batch[0].to(self.device))
            hiddens,_  = self.context(embeddings, lengths = batch[1].to(self.device)) # dim = (batch_size, input_length, hidden_dim)
            log_probs  = F.log_softmax(self.out(hiddens), dim = 2)                    # dim = (batch_size, input_length, lang_size)
            return log_probs

        def computeSuccess(log_probs, targets) :
            total   = np.sum(targets.data.cpu().numpy() != self.ignore_index_out)
            success = sum([self.ignore_index_out != targets[i, j].item() == log_probs[i, :, j].data.topk(1)[1].item() \
                           for i in range(targets.size(0)) \
                           for j in range(targets.size(1)) ])
            return success, total
        
        # --- main ----
        self.eval()
        batches = self.generatePackedSentences(sentences, batch_size)
        score, total = 0, 0
        for batch, targets in batches :
            log_probs = computeLogProbs(batch).transpose(1, 2) # dim = (batch_size, lang_size, input_length)
            targets = targets.to(self.device)                  # dim = (batch_size, input_length)
            s, t = computeSuccess(log_probs, targets)
            score += s
            total += t
        return score * 100 / total
    
    # fit model
    def fit(self, batches, 
            iters = None, epochs = None, lr = 0.025, random_state = 42, 
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

        def printScores(start, iter, iters, tot_loss, tot_loss_words, print_every, compute_accuracy) :
            avg_loss = tot_loss / print_every
            avg_loss_words = tot_loss_words / print_every
            if compute_accuracy : print(timeSince(start, iter / iters) + ' ({} {}%) loss : {:.3f}  accuracy : {:.1f} %'.format(iter, int(iter / iters * 100), avg_loss, avg_loss_words))
            else                : print(timeSince(start, iter / iters) + ' ({} {}%) loss : {:.3f}                     '.format(iter, int(iter / iters * 100), avg_loss))
            return 0, 0
        
        def computeLogProbs(batch) :
            embeddings = self.word2vec.embedding(batch[0].to(self.device))
            hiddens,_  = self.context(embeddings, lengths = batch[1].to(self.device)) # dim = (batch_size, input_length, hidden_dim)
            log_probs  = F.log_softmax(self.out(hiddens), dim = 2)                    # dim = (batch_size, input_length, lang_size)
            return log_probs

        def computeAccuracy(log_probs, targets) :
            total   = np.sum(targets.data.cpu().numpy() != self.ignore_index_out)
            success = sum([self.ignore_index_out != targets[i, j].item() == log_probs[i, :, j].data.topk(1)[1].item() \
                           for i in range(targets.size(0)) \
                           for j in range(targets.size(1)) ])
            return  success * 100 / total

        def trainLoop(batch, optimizer, compute_accuracy = True):
            """Performs a training loop, with forward pass, backward pass and weight update."""
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            self.zero_grad()
            log_probs  = computeLogProbs(batch[0]).transpose(1, 2) # dim = (batch_size, lang_size, input_length)
            targets    = batch[1].to(self.device)                  # dim = (batch_size, input_length)
            loss       = self.criterion(log_probs, targets)
            loss.backward()
            optimizer.step()
            if compute_accuracy :
                accuracy = computeAccuracy(log_probs, targets)
            else : 
                accuracy = 0
            error = float(loss.item() / np.sum(targets.data.cpu().numpy() != self.ignore_index_out))
            return error, accuracy
        
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
