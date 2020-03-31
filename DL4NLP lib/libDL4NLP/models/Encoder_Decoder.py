
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

from libDL4NLP.modules import (RecurrentEncoder,
                               Decoder,
                               AttnDecoder)



#-------------------------------------------------------------------#
#                          Encoder-Decoder                          #
#-------------------------------------------------------------------#


class EncoderDecoder(nn.Module) :
    def __init__(self, device, tokenizer, word2vec_in, word2vec_out, 
                 hidden_dim_in = 50,
                 hidden_dim_out = 50,
                 n_layers_in = 1,
                 n_layers_out = 1,
                 bound = 25,
                 dropout = 0,
                 decoder_warm_start = True,
                 decoder_attention = True,
                 optimizer = optim.SGD
                 ):
        super(EncoderDecoder, self).__init__()
        #relevant quantities
        self.decoder_warm_start = decoder_warm_start
        
        # modules
        self.tokenizer    = tokenizer
        self.word2vec_in  = word2vec_in
        self.word2vec_out = word2vec_out
        self.context      = RecurrentEncoder(word2vec_in.output_dim, hidden_dim_in, n_layers_in, dropout, bidirectional = True)
        if decoder_attention : self.decoder = AttnDecoder(word2vec_out, hidden_dim_in, hidden_dim_out, n_layers_out, dropout, bound)
        else :                 self.decoder = Decoder(word2vec_out, hidden_dim_out, n_layers_out, dropout, bound)
        
        # optimizer
        self.ignore_index_in  = self.word2vec_in.lang.getIndex('PADDING_WORD')
        self.ignore_index_out = self.word2vec_out.lang.getIndex('PADDING_WORD')
        self.criterion = nn.NLLLoss(size_average = False, ignore_index = self.ignore_index_out)
        self.optimizer = optimizer
        
        # load to device
        self.device = device
        self.to(device)
        
    def nbParametres(self) :
        return sum([p.data.nelement() for p in self.parameters() if p.requires_grad == True])
    
    # main method
    def forward(self, sentence, attention_method = None):
        # encode sentence
        words = self.tokenizer(sentence)
        words = [w for w in words if self.word2vec_in.lang.getIndex(w) is not None]
        indices = [self.word2vec_in.lang.getIndex(w) for w in words]
        embeddings = Variable(torch.LongTensor([indices])).to(self.device)
        embeddings = self.word2vec_in.embedding(embeddings)
        #embeddings = self.word2vec_in(words, self.device)
        embeddings, hidden  = self.context(embeddings)
        # prepare for decoding
        if self.decoder_warm_start :
            if self.context.bidirectional :
                hidden = hidden.view(self.context.n_layers, 2, -1, self.context.hidden_dim)
                hidden = torch.sum(hidden, dim = 1) # size (n_layers, batch_size, hidden_dim)
            hidden = hidden[-self.decoder.n_layers:]
        else : hidden = None    
        ## compute answer
        indices, attn = self.decoder(hidden, embeddings, self.device)
        attn = np.array(attn[0].data.cpu().numpy()) # size (input_length, output_length)
        words_out = [self.word2vec_out.lang.index2word[i] for i in indices]
        answer = ' '.join(words_out)
        if attention_method is not None : attention_method(attn, words_out, words)
        return answer

    # load data
    def generatePackedSentences(self, sentences, batch_size = 32) : 
        sentences.sort(key = lambda s: len(s[1]), reverse = True)
        packed_data = []
        for i in range(0, len(sentences), batch_size) :
            # prepare input and target pack
            pack = sentences[i:i + batch_size]
            pack.sort(key = lambda s: len(self.tokenizer(s[0])), reverse = True)
            pack0 = [[self.word2vec_in.lang.getIndex(w) for w in self.tokenizer(qa[0])] for qa in pack]
            pack0 = [[w for w in words if w is not None] for words in pack0]
            pack1 = [[self.word2vec_out.lang.getIndex(w) for w in self.tokenizer(qa[1]) + ['EOS']] for qa in pack]
            pack1 = [[w for w in words if w is not None] for words in pack1]
            lengths0 = torch.tensor([len(p) for p in pack0])           # size (batch_size) 
            lengths1 = torch.tensor([len(p) for p in pack1])           # size (batch_size) 
            # padd packs
            pack0 = list(itertools.zip_longest(*pack0, fillvalue = self.ignore_index_in))
            pack0 = Variable(torch.LongTensor(pack0).transpose(0, 1)) # size (batch_size, max_length0) 
            pack1 = list(itertools.zip_longest(*pack1, fillvalue = self.ignore_index_out))
            pack1 = Variable(torch.LongTensor(pack1))       # WARNING : size (max_length1, batch_size) 
            packed_data.append([pack0, lengths0, pack1, lengths1])
        return packed_data
    
    # compute model perf
    def compute_accuracy(self, sentences, batch_size = 32) :
        batches = self.generatePackedSentences(sentences, batch_size)
        score = 0
        for batch in batches :
            input, input_l, target, target_l = batch
            target = target.to(self.device)
            # encode sentences
            embeddings = self.word2vec_in.embedding(input.to(self.device))
            embeddings, hidden = self.context(embeddings, lengths = input_l.to(self.device)) # size (n_layers * num_directions, batch_size, hidden_dim)
            # prepare for decoding
            if self.decoder_warm_start :
                if self.context.bidirectional :
                    hidden = hidden.view(self.context.n_layers, 2, -1, self.context.hidden_dim)
                    hidden = torch.sum(hidden, dim = 1) # size (n_layers, batch_size, hidden_dim)
                hidden = hidden[-self.decoder.n_layers:]
            else : hidden = None  
            # compute answers
            answers = torch.zeros(target.size(), dtype = torch.long)
            word_index = self.word2vec_out.lang.getIndex('SOS')
            word_index = Variable(torch.LongTensor([word_index])) # size (1)
            word_index = word_index.expand(target.size(1))        # size (batch_size)
            for t in range(target.size(0)) :
                # compute word probs
                log_prob, hidden, atn = self.decoder.generateWord(hidden, embeddings, word_index.unsqueeze(1).to(self.device))
                word_index = log_prob.topk(1, dim = 1)[1].view(-1) # size (batch_size)
                answers[t] = word_index
            # update score
            score += sum([sum(answers[:l, i].data.cpu() == target[:l, i].data.cpu()) == l 
                          for i, l in enumerate(target_l.data.cpu().tolist())]).item()
        return score * 100 / len(sentences)
    
    # fit model
    def fit(self, batches, iters = None, epochs = None, tf_ratio = 0, lr = 0.025, random_state = 42,
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
        
        def computeSuccess(log_probs, targets) :
            success = sum([self.ignore_index_out != targets[i].item() == log_probs[i].topk(1)[1].item() \
                           for i in range(targets.size(0))])
            return success
        
        def computeLogProbs(batch, tf_ratio = 0, compute_accuracy = True) :
            loss = 0
            success = 0
            forcing = (random.random() < tf_ratio)
            input, input_l, target, target_l = batch
            target = target.to(self.device)
            # encode sentences
            embeddings = self.word2vec_in.embedding(input.to(self.device))
            embeddings, hidden  = self.context(embeddings, lengths = input_l.to(self.device)) # size (n_layers * num_directions, batch_size, hidden_dim)
            # prepare for decoding
            if self.decoder_warm_start :
                if self.context.bidirectional :
                    hidden = hidden.view(self.context.n_layers, 2, -1, self.context.hidden_dim)
                    hidden = torch.sum(hidden, dim = 1) # size (n_layers, batch_size, hidden_dim)
                hidden = hidden[-self.decoder.n_layers:]
            else : hidden = None  
            # compute answers
            word_index = self.word2vec_out.lang.getIndex('SOS')
            word_index = Variable(torch.LongTensor([word_index])) # size (1)
            word_index = word_index.expand(target.size(1))        # size (batch_size)
            for t in range(target.size(0)) :
                # compute word probs
                log_prob, hidden, atn = self.decoder.generateWord(hidden, embeddings, word_index.unsqueeze(1).to(self.device))
                # compute loss
                loss += self.criterion(log_prob, target[t])
                if compute_accuracy : success += computeSuccess(log_prob, target[t])
                # apply teacher forcing
                if forcing : word_index = target[t]                             # size (batch_size) 
                else       : word_index = log_prob.topk(1, dim = 1)[1].view(-1) # size (batch_size)
            return loss, success       

        def printScores(start, iter, iters, tot_loss, tot_loss_words, print_every, compute_accuracy) :
            avg_loss = tot_loss / print_every
            avg_loss_words = tot_loss_words / print_every
            if compute_accuracy : print(timeSince(start, iter / iters) + ' ({} {}%) loss : {:.3f}  accuracy : {:.1f} %'.format(iter, int(iter / iters * 100), avg_loss, avg_loss_words))
            else                : print(timeSince(start, iter / iters) + ' ({} {}%) loss : {:.3f}                     '.format(iter, int(iter / iters * 100), avg_loss))
            return 0, 0

        def trainLoop(batch, optimizer, tf_ratio = 0, compute_accuracy = True):
            """Performs a training loop, with forward pass, backward pass and weight update."""
            optimizer.zero_grad()
            self.zero_grad()
            total = torch.sum(batch[-1]).item()
            loss, success = computeLogProbs(batch, tf_ratio, compute_accuracy)
            loss.backward()
            optimizer.step()
            return float(loss.item() / total), float(success * 100 / total)
        
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
                loss, acc = trainLoop(batch, optimizer, tf_ratio, compute_accuracy)
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
                    loss, acc = trainLoop(batch, optimizer, tf_ratio, compute_accuracy)
                    tot_loss += loss
                    tot_acc += acc 
                    iter += 1
                    if iter % print_every == 0 : 
                        tot_loss, tot_acc = printScores(start, iter, iters, tot_loss, tot_acc, print_every, compute_accuracy)
        return
