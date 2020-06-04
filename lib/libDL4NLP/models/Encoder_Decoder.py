
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
                 hid_dim_in = 50,
                 hid_dim_out = 50,
                 n_layer_in = 1,
                 n_layer_out = 1,
                 bound = 25,
                 dropout = 0,
                 decoder_warm_start = True,
                 decoder_type = None,
                 optimizer = optim.SGD
                 ):
        
        super().__init__()
        #relevant quantities
        self.decoder_warm_start = decoder_warm_start
        self.decoder_type = decoder_type
        
        # modules
        self.tokenizer    = tokenizer
        self.word2vec_in  = word2vec_in
        self.word2vec_out = word2vec_out
        
        self.context      = RecurrentEncoder(
            emb_dim = word2vec_in.out_dim,
            hid_dim = hid_dim_in,
            n_layer = n_layer_in,
            dropout = dropout, 
            bidirectional = True)

        if self.decoder_type == 'smooth' : 
            self.decoder = SmoothAttnDecoder(
                word2vec = word2vec_out, 
                attn_dim = hid_dim_in, 
                hid_dim  = hid_dim_out, 
                n_layer  = n_layer_out, 
                dropout  = dropout, 
                bound    = bound,
                top      = 5)
            
        elif self.decoder_type == 'attention' : 
            self.decoder = AttnDecoder(
                word2vec = word2vec_out, 
                attn_dim = hid_dim_in, 
                hid_dim  = hid_dim_out, 
                n_layer  = n_layer_out, 
                dropout  = dropout, 
                bound    = bound)
            
        else : 
            self.decoder = Decoder(
                word2vec = word2vec_out,  
                hid_dim  = hid_dim_out, 
                n_layer  = n_layer_out, 
                dropout  = dropout, 
                bound    = bound)
        
        # optimizer
        self.ignore_index_in  = self.word2vec_in.lang.getIndex('PADDING_WORD')
        self.ignore_index_out = self.word2vec_out.lang.getIndex('PADDING_WORD')
        self.criterion = nn.NLLLoss(size_average = False, ignore_index = self.ignore_index_out)
        self.optimizer = optimizer
        
        # load to device
        self.device = device
        self.to(device)
        

    # count parameters
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
                hidden = hidden.view(self.context.n_layer, 2, -1, self.context.hid_dim)
                hidden = torch.sum(hidden, dim = 1) # size (n_layer, batch_size, hid_dim)
            hidden = hidden[-self.decoder.n_layer:]
        else : hidden = None    
        # compute answer
        if self.decoder_type in ['smooth', 'attention'] : 
            indices, attn = self.decoder(hidden, embeddings, self.device)
            words_out = [self.word2vec_out.lang.index2word[i] for i in indices]
            # display attention
            if attention_method is not None : 
                attn = np.array(attn[0].data.cpu().numpy()) # size (input_length, output_length)
                attention_method(attn, words_out, words)
        else :
            indices   = self.decoder(hidden, self.device)
            words_out = [self.word2vec_out.lang.index2word[i] for i in indices]
        # convert answer to string
        answer = ' '.join(words_out)
        return answer

    
    # load data
    def generatePackedSentences(self, sentences, batch_size = 32) : 
        '''forms minibatches of sentences, where input sentences must be pre-tokenized'''
        sentences.sort(key = lambda s: len(s[1]), reverse = True)
        packed_data = []
        for i in range(0, len(sentences), batch_size) :
            # prepare input and target pack
            pack = sentences[i:i + batch_size]
            pack.sort(key = lambda s: len(s[0]), reverse = True)
            pack0 = [[self.word2vec_in.lang.getIndex(w) for w in qa[0]] for qa in pack]
            pack0 = [[w for w in words if w is not None] for words in pack0]
            pack1 = [[self.word2vec_out.lang.getIndex(w) for w in qa[1] + ['EOS']] for qa in pack]
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
        def compute_batch_accuracy(batch) :
            torch.cuda.empty_cache()
            input, input_l, target, target_l = batch
            target = target.to(self.device)
            # encode sentences
            embeddings = self.word2vec_in.embedding(input.to(self.device))
            embeddings, hidden = self.context(embeddings, lengths = input_l.to(self.device)) # size (n_layer * num_directions, batch_size, hid_dim)
            # prepare for decoding
            if self.decoder_warm_start :
                if self.context.bidirectional :
                    hidden = hidden.view(self.context.n_layer, 2, -1, self.context.hid_dim)
                    hidden = torch.sum(hidden, dim = 1) # size (n_layer, batch_size, hid_dim)
                hidden = hidden[-self.decoder.n_layer:]
            else : hidden = None  
            # compute answers
            answers   = torch.zeros(target.size(), dtype = torch.long)
            SOS_token = self.word2vec_out.lang.getIndex('SOS')
            word      = self.decoder.initWordTensor([SOS_token]*target.size(1), device = self.device) 
            # word generation
            for t in range(target.size(0)) :
                # feeds word proba at previous step as input for next word prediction
                if self.decoder_type == 'smooth' :
                    vect, hidden, attn = self.decoder.generateWord(hidden, embeddings, word)
                    proba = self.decoder.computeProba(vect) # size (batch_size, lang_size)
                    best  = self.decoder.sampleWord(proba)   # size (batch_size, 1)
                    word  = proba
                # feeds most probable word at previous step as input for next word prediction
                elif self.decoder_type == 'attention' :
                    vect, hidden, attn = self.decoder.generateWord(hidden, embeddings, word)
                    proba = self.decoder.computeProba(vect) # size (batch_size, lang_size)
                    best  = self.decoder.sampleWord(proba)  # size (batch_size, 1)
                    word  = best
                # feeds most probable word at previous step as input for next word prediction
                else :
                    vect, hidden = self.decoder.generateWord(hidden, word)
                    word = vect.topk(1, dim = 1)[1]   # size (batch_size, 1)
                    best = word                       # size (batch_size, 1)
                answers[t] = best.view(-1)
            # compute score
            score = sum([sum(answers[:l, i].data.cpu() == target[:l, i].data.cpu()) == l 
                          for i, l in enumerate(target_l.data.cpu().tolist())]).item()
            return score
        
        # -- main --
        batches = self.generatePackedSentences(sentences, batch_size)
        score = 0
        for batch in batches : score += compute_batch_accuracy(batch)
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
        
        def computeSuccess(log_prob, targets) :
            success = sum([self.ignore_index_out != targets[i].item() == log_prob[i].topk(1)[1].item() \
                           for i in range(targets.size(0))])
            return success
        
        def computeLogProbs(batch, tf_ratio = 0, compute_accuracy = True) :
            torch.cuda.empty_cache()
            loss = 0
            success = 0
            forcing = (random.random() < tf_ratio)
            input, input_l, target, target_l = batch
            target = target.to(self.device)
            # encode sentences
            embeddings = self.word2vec_in.embedding(input.to(self.device))
            embeddings, hidden  = self.context(embeddings, lengths = input_l.to(self.device)) # size (n_layer * num_directions, batch_size, hid_dim)
            # init decoder state
            if self.decoder_warm_start :
                if self.context.bidirectional :
                    hidden = hidden.view(self.context.n_layer, 2, -1, self.context.hid_dim)
                    hidden = torch.sum(hidden, dim = 1) # size (n_layer, batch_size, hid_dim)
                hidden = hidden[-self.decoder.n_layer:]
            else : hidden = None  
            # decode answers
            SOS_token = self.word2vec_out.lang.getIndex('SOS')
            word      = self.decoder.initWordTensor([SOS_token]*target.size(1), device = self.device) 
            for t in range(target.size(0)) :
                # feeds word proba at previous step as input for next word prediction
                if self.decoder_type == 'smooth' :
                    vect, hidden, attn = self.decoder.generateWord(hidden, embeddings, word)
                    if forcing : word  = self.decoder.initWordTensor(target[t].data.tolist(), device = self.device) # size (batch_size, 1) 
                    else       : word  = self.decoder.computeProba(vect) # size (batch_size, lang_size)
                # feeds most probable word at previous step as input for next word prediction
                elif self.decoder_type == 'attention' :
                    vect, hidden, attn = self.decoder.generateWord(hidden, embeddings, word)
                    if forcing : word  = target[t].view(-1, 1)    # size (batch_size, 1) 
                    else       : word  = vect.topk(1, dim = 1)[1] # size (batch_size, 1)
                # feeds most probable word at previous step as input for next word prediction
                else :
                    vect, hidden      = self.decoder.generateWord(hidden, word)
                    if forcing : word = target[t].view(-1, 1)    # size (batch_size, 1) 
                    else       : word = vect.topk(1, dim = 1)[1] # size (batch_size, 1)
                # cumulate loss
                log_prob = F.log_softmax(vect, dim = 1)
                loss += self.criterion(log_prob, target[t])
                if compute_accuracy : success += computeSuccess(log_prob, target[t])
            return loss, success
        
        def printScores(start, iter, iters, tot_loss, tot_loss_words, print_every, compute_accuracy) :
            avg_loss = tot_loss / print_every
            avg_loss_words = tot_loss_words / print_every
            if compute_accuracy : print(timeSince(start, iter / iters) + ' ({} {}%) loss : {:.3f}  accuracy : {:.1f} %'.format(iter, int(iter / iters * 100), avg_loss, avg_loss_words))
            else                : print(timeSince(start, iter / iters) + ' ({} {}%) loss : {:.3f}                     '.format(iter, int(iter / iters * 100), avg_loss))
            return 0, 0
        
        def trainLoop(batch, optimizer, tf_ratio = 0, compute_accuracy = True):
            """Performs a training loop, with forward pass, backward pass and weight update"""
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
