
import math
import time
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable


class NoiseFilter(nn.Module):

    def __init__(self, chatbot, pretrained = True, layers = [50], dropout = 0.15):
        super(NoiseFilter, self).__init__()
        
        # modules        
        self.device = chatbot.device
        self.chatbot = chatbot
        if pretrained : 
            for param in chatbot.parameters() : param.requires_grad = False
        self.decoder = nn.ModuleList([nn.Linear(chatbot.encoder.output_dim, layers[0])] + 
                                     [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1) if len(layers) > 1] +
                                     [nn.Linear(layers[-1], 2)])
        self.dropout = nn.Dropout(dropout)
        
        
    # ---------------------- Technical methods -----------------------------
    def nbParametres(self) :
        count = 0
        for p in self.parameters():
            if p.requires_grad == True : count += p.data.nelement()
        return count
        
        
    # ------------ 2nd working mode : test mode ------------
    def forward(self, input):

        sentence = self.chatbot.variableFromSentence(input)
        if sentence is None :
            return 0, None, None
        else :
            sentence = sentence.to(self.device)
            last_words, hidden = self.chatbot.encoder(sentence)
            hidden = self.dropout(hidden.view(1,1,-1))
            for layer in self.decoder : hidden = self.dropout(F.relu(layer(hidden)))
                
            log_probas  = F.log_softmax(hidden.view(1, -1), dim = 1)
            topv, topi = log_probas.data.topk(1)
            predict = topi[0][0]
            return predict, log_probas
        
        

        
class NoiseFilterWrapper(nn.Module) :
    def __init__(self, noise_filter, chatbot) :
        
        super(NoiseFilterWrapper, self).__init__()
        self.noise_filter = noise_filter
        self.chatbot = chatbot
        self.basic_answer = "Je n'ai pas compris, merci de reformuler la question"
        
    def forward(self, sentence) :
        #print(self.noise_filter(sentence)[1].data)
        if self.noise_filter(sentence)[0] == 1 : return self.chatbot(sentence)
        else                                   : return self.basic_answer, None, None
        
        

class NoiseFilterTrainer(object):
    def __init__(self, 
                 device,
                 criterion = nn.NLLLoss(), 
                 optimizer = optim.SGD, 
                 clipping = 10, 
                 print_every=100):
        
        # relevant quantities
        self.device = device
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.clip = clipping
        self.print_every = print_every# timer
        
        
    def asMinutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)


    def timeSince(self, since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))
        
        
    def distance(self, probas, target_var) :
        """ Compute cumulated error between predicted output and ground answer."""
        loss = self.criterion(probas, target_var)
        loss_diff = int(ni != target_var.item())
        return loss, loss_diff
        
        
    def trainLoop(self, agent, sentence, target, optimizer):
        """Performs a training loop, with forward pass and backward pass for gradient optimisation."""
        optimizer.zero_grad()
        target_var = Variable(torch.LongTensor([target])).to(self.device)
        answer, log_probas = agent(sentence) 
        loss = self.criterion(log_probas, target_var)
        loss_diff_mots = int(answer != target_var.item())
        
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(agent.parameters(), self.clip)
        optimizer.step()
        return loss.data[0] , loss_diff_mots
        
        
    def train(self, agent, sentences, n_iters = 10000, learning_rate=0.01):
        """Performs training over a given dataset and along a specified amount of loops."""
        start = time.time()
        optimizer = self.optimizer([param for param in agent.parameters() if param.requires_grad == True], lr=learning_rate)
        print_loss_total = 0  
        print_loss_diff_mots_total = 0
        for iter in range(1, n_iters + 1):
            training_sentence = random.choice(sentences)
            sentence = training_sentence[0]
            target   = training_sentence[1]

            loss, loss_diff_mots = self.trainLoop(agent, sentence, target, optimizer)
            # quantité d'erreurs sur la réponse i
            print_loss_total += loss
            print_loss_diff_mots_total += loss_diff_mots       
            if iter % self.print_every == 0:
                print_loss_avg = print_loss_total / self.print_every
                print_loss_diff_mots_avg = print_loss_diff_mots_total / self.print_every
                print_loss_total = 0
                print_loss_diff_mots_total = 0
                print('%s (%d %d%%) %.4f %.2f' % (self.timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg, print_loss_diff_mots_avg))
