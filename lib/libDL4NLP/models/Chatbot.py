
import math
import time
import unicodedata
import re
import random

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker #, FuncFormatter
#%matplotlib inline

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from .Language_Model import LanguageModel


from libDL4NLP.modules import  (RecurrentWordsEncoder, 
                            
                                AdditiveAttention,
                                MultiHeadAttention,
                                MultiHopedAttention,
                                RecurrentHierarchicalAttention, 

                                WordsDecoder,
                                AttnWordsDecoder,
                                LMWordsDecoder)



#-------------------------------------------------------------------#
#                         Chatbot model                             #
#-------------------------------------------------------------------#


class Chatbot(nn.Module):
    """Conversationnal agent with bi-GRU Encoder, taking as parameters at training time :
    
            -a complete dialogue of the form (with each content as string)
    
                    [['question 1', 'answer 1'],
                     ['question 2', 'answer 2'],
                             ..........
                     ['current question', 'current answer']]
     
            -the current answer for teacher forcing, or None
    
    and at test time :
    
            -the current question as string
    
    Returns :
     
            -word indices of the generated answer, according to output language of the model
            -attention weights of first attention layer, or None is no attention
            -attention weights of second attention layer, or None is no attention
    """
    def __init__(self, device, lang, encoder, attention, decoder, autoencoder = None):
        super(Chatbot, self).__init__()
        
        # relevant quantities
        self.lang = lang 
        self.device = device
        self.n_level = attention.n_level if attention is not None else 1
        self.memory_dim = encoder.output_dim
        self.memory_length = 0
        # modules        
        self.encoder = encoder
        self.attention = attention
        self.decoder = decoder
        self.autoencoder = autoencoder
        
        
        
    # ---------------------- Technical methods -----------------------------
    def loadSubModule(self, encoder = None, attention = None, decoder = None) :
        if encoder is not None   : self.encoder = encoder
        if attention is not None : self.attention = attention
        if decoder is not None   : self.decoder = decoder
        return
    
    def freezeSubModule(self, encoder = False, attention = False, decoder = False) :
        for param in self.encoder.parameters()  : param.requires_grad = not encoder
        for param in self.attention.parameters(): param.requires_grad = not attention
        for param in self.decoder.parameters()  : param.requires_grad = not decoder
        return
    
    def nbParametres(self) :
        count = 0
        for p in self.parameters():
            if p.requires_grad == True : count += p.data.nelement()
        return count
    
    def flatten(self, description):
        '''Baisse le nombre de niveaux de 1 dans la description'''
        flatten = []
        for line in description :
            if type(line) == list : flatten += line  
            else                  : flatten.append(line)
        return [[int(word) for word in sentence.data.view(-1)] for sentence in flatten]

    
    
    # ------------------------ Text processing methods ---------------------------------
    def variableFromSentence(self, sentence):
        def normalizeString(sentence) :
            '''Remove rare symbols from a string'''
            def unicodeToAscii(s):
                """Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427"""
                return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
            sentence = unicodeToAscii(sentence.lower().strip())
            sentence = re.sub(r"[^a-zA-Z0-9?&\%\-\_]+", r" ", sentence) 
            return sentence
        sentence = normalizeString(sentence).split(' ') # a raw string transformed into a list of clean words
        indexes = []
        #unknowns = 0
        for word in sentence:
            if word not in self.lang.word2index.keys() :
                if 'UNK' in self.lang.word2index.keys() : indexes.append(self.lang.word2index['UNK'])
            else :
                indexes.append(self.lang.word2index[word])
        #indexes.append(self.lang.word2index['EOS']) 
        indexes.append(1) # EOS_token
        result = Variable(torch.LongTensor([[i] for i in indexes]))
        return result
    
    
    
    # ------------------------ Visualisation methods ---------------------------------
    def flattenDialogue(self, dialogue):
        flatten = []
        for paire in dialogue : flatten += paire
        return [[int(word) for word in sentence.data.view(-1)] for sentence in flatten]
    
    def flattenWeights(self, weights) :
        '''Baisse le nombre de niveaux de 1 dans les poids d'attention'''
        flatten = []
        for weight_layer in weights : flatten.append(torch.cat(tuple(weight_layer.values()), dim = 2))
        return flatten
    
    def formatWeights(self, dialogue, attn1_weights, attn2_weights) :
        if self.n_level == 2 : attn1_weights = self.flattenWeights(attn1_weights)
        hops = self.attention.hops
        l, L = len(dialogue), max([len(line) for line in dialogue])
        Table = np.zeros((l, 1, L))
        Liste = np.zeros((l, 1)) if attn2_weights is not None else None
        count = 0
        count_line = 0
        for i, line in enumerate(dialogue) :
            present = False
            for j, word in enumerate(line) :
                if word in self.lang.index2word.keys():
                    present = True
                    Table[i, 0, j] = sum([attn1_weights[k][0, 0, count].data for k in range(hops)])
                    count += 1
            if present and Liste is not None :
                Liste[i] = sum([attn2_weights[k][count_line].data for k in range(hops)])
                count_line += 1
        return Table, Liste
    
    def showWeights(self, dialogue, attn1_weights, attn2_weights, maxi):
        table, liste = self.formatWeights(dialogue[:-2], attn1_weights, attn2_weights)
        l = table.shape[0]
        L = table.shape[2]
        fig = plt.figure(figsize = (l, L))
        for i, line in enumerate(dialogue[:-2]):
            ligne = [self.lang.index2word[int(word)] for word in line]
            ax = fig.add_subplot(l, 1, i+1)
            vals = table[i]
            text = [' '] + ligne + [' ' for k in range(L-len(ligne))] if L>len(ligne) else [' '] + ligne
            if liste is not None :
                vals = np.concatenate((np.zeros((1, 1)) , vals), axis = 1)  
                vals = np.concatenate((np.reshape(liste[i], (1, 1)) , vals), axis = 1)
                turn = 'User' if i % 2 == 0 else 'Bot'
                text = [turn] + [' '] + text
            cax = ax.matshow(vals, vmin=0, vmax=maxi, cmap='YlOrBr')
            ax.set_xticklabels(text, ha='left')
            ax.set_yticklabels(' ')
            ax.tick_params(axis=u'both', which=u'both',length=0, labelrotation = 30, labelright  = True)
            ax.grid(b = False, which="minor", color="w", linestyle='-', linewidth=1)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            plt.subplots_adjust(hspace=0, wspace = 0.1)
        plt.show()
    
    def showAttention(self, dialogue, n_col = 1, maxi = None):
        answer, decoder_outputs, attn1_w, attn2_w, _ = self.answerTrain(dialogue)
        dialogue = self.flattenDialogue(dialogue)
        if len(dialogue) > 1 : self.showWeights(dialogue, attn1_w, attn2_w, maxi)
        print('User : ', ' '.join([self.lang.index2word[int(word)] for word in dialogue[-2][:-1]]))
        print('target : ', ' '.join([self.lang.index2word[int(word)] for word in dialogue[-1][:-1]]))
        print('predic : ', ' '.join([self.lang.index2word[int(word)] for word in answer]))
        return
    
    
    
    # ------------------- Process methods ------------------------
    def initMemory(self):
        """Initialize memory slots"""
        self.memory = {}
        self.memory_queries = {}
        self.query_hidden = self.encoder.initHidden()
        self.memory_length = 0
        
    def updateMemory(self, last_words, query_hidden):
        """Update memory with a list of word vectors 'last_words' and the last query vector 'last_query'"""
        self.memory[self.memory_length] = last_words
        self.memory_queries[self.memory_length] = query_hidden
        self.query_hidden = query_hidden
        self.memory_length += 1
        
    def readSentence(self, utterance):
        """Perform reading of an utterance, returning created word vectors
           and last hidden states of teh encoder bi-GRU
        """
        utterance = utterance.to(self.device)
        last_words, query_hidden = self.encoder(utterance, self.query_hidden)
        return last_words, query_hidden
        
    def readDialogue(self, dialogue):
        """Loop of readUtterance over a whole dialogue
        """
        for i in range(len(dialogue)) :
            if type(dialogue[i]) == list :
                for utterance in dialogue[i]:
                    last_words, query_hidden = self.readSentence(utterance)
                    self.updateMemory(last_words, query_hidden)
            else :
                utterance = dialogue[i]
                last_words, query_hidden = self.readSentence(utterance)
                self.updateMemory(last_words, query_hidden)
   
    def tracking(self, query_vector = None):
        """Détermine un vecteur d'attention sur les éléments du registre de l'agent,
        sachant un vecteur 'very_last_hidden', et l'accole à ce vecteur """
        decision_vector, attn1_weights, attn2_weights = self.attention(words_memory = self.memory, 
                                                                       query = query_vector)
        return decision_vector, attn1_weights, attn2_weights

    def generateAnswer(self,last_words, query_vector, decision_vector, target_answer = None) :
        """Génère une réponse à partir d'un état caché initialisant le décodeur,
        en utilisant une réponse cible pour un mode 'teacher forcing-like' si celle-ci est fournie """
        answer, decoder_outputs = self.decoder(last_words, query_vector, decision_vector, target_answer)
        return answer, decoder_outputs
    
    def generateQuery(self,last_words, query_vector, decision_vector, target_answer = None) :
        """Génère une réponse à partir d'un état caché initialisant le décodeur,
        en utilisant une réponse cible pour un mode 'teacher forcing-like' si celle-ci est fournie """
        if self.autoencoder is not None : 
            query, autoencoder_outputs = self.autoencoder(last_words, query_vector, decision_vector, target_answer)
            return query, autoencoder_outputs
        else :
            return None, None
        
        
        
    # ------------ 1st working mode : training mode ------------
    def answerTrain(self, input, target_answer = None):
        """Parameters are a complete dialogue, containing the current query,
           
           - either of the form :

                    [['query 1', 'answer 1'],
                     ['query 2', 'answer 2'],
                             ..........
                     ['current query', 'current answer']]
                     
           - or :
           
                    ['query 1',
                     'query 2',
                       ....
                     'current query'] 

           The model learns to generate the current answer. 
           Teacher forcing can be enabled by passing the ground answer though the 'target_answer' option.
        """
        # 1) initiates memory instance
        self.initMemory()
        
        # 2) reads historical part of dialogue (if applicable),
        # word vectors and last hidden states of encoder bi-GRU are stored in memory
        dialogue = input[:-1]
        self.readDialogue(dialogue)
        
        # 3) reads current utterance,
        # returns word vectors of query and query vector
        query = input[-1]
        query = query[0] if type(query) == list else query
        last_words, query_hidden = self.readSentence(query)
        query_hidden = query_hidden.view(1, 1, -1)
        
        # 4) performs tracking
        # returns decision vector
        if self.attention is not None :
            decision_vector, attn1_weights, attn2_weights = self.tracking(query_hidden)
        else :
            decision_vector = query_hidden
            attn1_attention_weights = None
            attn2_attention_weights = None
            
        # 5) response generation
        # returns list of indices
        answer, decoder_outputs = self.generateAnswer(last_words, query_hidden, decision_vector, target_answer)
        pred_query, autoencoder_outputs = self.generateQuery(last_words, query_hidden, decision_vector, target_answer)    
        # 6) returns answer
        return answer, decoder_outputs, attn1_weights, attn2_weights, autoencoder_outputs

        
        
    # ------------ 2nd working mode : test mode ------------
    def forward(self, input):
        """Parameters are a single current query as string, and the model attempts to generate the current answer.
        """
        
        # 1) initiates memory and hidden states of encoder bi-GRU if conversation starts
        if self.memory_length == 0 : self.initMemory()
            
        # 2) reads current utterance,
        # returns word vectors of query and query vector
        sentence = self.variableFromSentence(input)
        if sentence is None :
            return "Excusez-moi je n'ai pas compris", None, None
        else :
            last_words, query_hidden = self.readSentence(sentence)
            q_hidden = query_hidden.view(1, 1, -1)

            # 3) performs tracking
            # returns decision vector
            if self.attention is not None :
                decision_vector, attn1_weights, attn2_weights = self.tracking(q_hidden)
            else :
                decision_vector = q_hidden
                attn1_attention_weights = None
                attn2_attention_weights = None

            # 4) response generation
            # returns list of indices
            answer, decoder_outputs = self.generateAnswer(last_words, q_hidden, decision_vector)
            
            # 5) updates memory with current query and answer
            self.updateMemory(last_words, query_hidden)
            answer_var = Variable(torch.LongTensor([[i] for i in answer]))
            last_words, query_hidden = self.readSentence(answer_var)
            self.updateMemory(last_words, query_hidden)

            # 6) returns answer
            answer = ' '.join([self.lang.index2word[int(word)] for word in answer])
            return answer, attn1_weights, attn2_weights
    
    

    
#-------------------------------------------------------------------#
#                         Instanciator                              #
#-------------------------------------------------------------------#
    
    
    
def CreateBot(lang,                     ###
              embedding,                  # --- Encoder options
              hidden_dim,                 #
              n_layers,                 ###

              sentence_hidden_dim,      ###
              hops,                       #
              share,                      # --- Hierarchical encoder options
              transf,                     #
              dropout,                  ###
              
              attn_decoder_n_layers,    ###
              language_model_n_layers,    #
              tf_ratio,                   # --- decoder options
              bound,                      #
              autoencoding,             ###
              
              device
             ):
    '''Create an agent with specified dimensions and specificities'''
    # 1) ----- encoding -----
    EOS_token = lang.word2index['EOS'] if 'EOS' in lang.word2index.keys() else 1
    if type(embedding) == torch.nn.modules.sparse.Embedding : 
        for param in embedding.parameters() : param.requires_grad = False
    elif type(embedding) == int : 
        embedding = nn.Embedding(lang.n_words, embedding) 
    else : 
        embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding), freeze=True)

    encoder = RecurrentWordsEncoder(device, 
                                    embedding, 
                                    hidden_dim, 
                                    n_layers, 
                                    dropout) # embedding, hidden_dim, n_layers = 1, dropout = 0
    # 2) ----- attention -----
    word_hidden_dim = encoder.output_dim
    attention = RecurrentHierarchicalAttention(device,
                                               word_hidden_dim,
                                               sentence_hidden_dim, 
                                               query_dim = word_hidden_dim,
                                               n_heads = 1,
                                               n_layers = n_layers,
                                               hops = hops,
                                               share = share,
                                               transf = transf,
                                               dropout = dropout)
    # 3) ----- decoding -----
    tracking_dim = attention.output_dim
    autoencoder = None
    if language_model_n_layers > 0 :
        language_model = LanguageModel(device, 
                                       lang,
                                       embedding = embedding, 
                                       hidden_dim = hidden_dim,
                                       n_layers = language_model_n_layers, 
                                       dropout = dropout)
        decoder = LMWordsDecoder(device,
                                 language_model,                                   
                                 word_hidden_dim,
                                 tracking_dim,
                                 dropout = dropout,
                                 tf_ratio = tf_ratio,
                                 bound = bound)
        if autoencoding :
            autoencoder = LMWordsDecoder(device,
                                         language_model,                                   
                                         word_hidden_dim,
                                         tracking_dim,
                                         dropout = dropout,
                                         tf_ratio = tf_ratio,
                                         bound = bound)
        
    elif attn_decoder_n_layers >= 0 :
        decoder = AttnWordsDecoder(device,
                                   embedding,
                                   word_hidden_dim,
                                   tracking_dim,
                                   dropout = dropout,
                                   n_layers = attn_decoder_n_layers,
                                   tf_ratio = tf_ratio,
                                   bound = bound)
        if autoencoding :
            autoencoder = AttnWordsDecoder(device,
                                           embedding,
                                           word_hidden_dim,
                                           tracking_dim,
                                           dropout = dropout,
                                           n_layers = attn_decoder_n_layers,
                                           tf_ratio = tf_ratio,
                                           bound = bound)
    else :
        decoder = WordsDecoder(device,
                               embedding,                                   
                               word_hidden_dim,
                               tracking_dim,
                               dropout = dropout,
                               tf_ratio = tf_ratio,
                               EOS_token = EOS_token,
                               bound = bound)       
        if autoencoding :
            autoencoder = WordsDecoder(device,
                                       embedding,                                   
                                       word_hidden_dim,
                                       tracking_dim,
                                       dropout = dropout,
                                       tf_ratio = tf_ratio,
                                       EOS_token = EOS_token,
                                       bound = bound) 
    # 4) ----- model -----
    chatbot = Chatbot(device, lang, encoder, attention, decoder, autoencoder)
    chatbot = chatbot.to(device)
    return chatbot



#-------------------------------------------------------------------#
#                             Trainer                               #
#-------------------------------------------------------------------#


class BotTrainer(object):
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
        
        
    def distance(self, agent_outputs, target_answer) :
        """ Compute cumulated error between predicted output and ground answer."""
        loss = 0
        loss_diff_mots = 0
        agent_outputs_length = len(agent_outputs)
        target_length = len(target_answer)
        Max = max(agent_outputs_length, target_length)
        Min = min(agent_outputs_length, target_length)   
        for i in range(Min):
            agent_output = agent_outputs[i]
            target_word = target_answer[i]
            #print(agent_output.size(), target_answer.size())
            factor = (1 + Max - Min) if i == Min -1 else 1
            loss += factor * self.criterion(agent_output, target_word)
            topv, topi = agent_output.data.topk(1)
            ni = topi[0][0]
            if ni != target_word.data[0]:
                loss_diff_mots += 1
        loss_diff_mots += Max - Min
        return loss, loss_diff_mots
        
        
    def trainLoop(self, agent, dialogue, target_answer, optimizer):
        """Performs a training loop, with forward pass and backward pass for gradient optimisation."""
        optimizer.zero_grad()
        query = dialogue[-1][0].to(self.device)
        target_answer = target_answer.to(self.device)
        answer, agent_outputs, attn1_w, attn2_w, _ = agent.answerTrain(dialogue, target_answer) 
        loss, loss_diff_mots = self.distance(agent_outputs, target_answer)
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(agent.parameters(), self.clip)
        optimizer.step()
        return loss.data[0] / len(target_answer), loss_diff_mots
        
        
    def train(self, agent, dialogues, n_iters = 10000, learning_rate=0.01, dic = None, per_dialogue = False, return_errors = False):
        """Performs training over a given dataset and along a specified amount of loops."""
        if type(dialogues[0][0]) == list :
            debut = 0
            double = True
        else :
            debut = 1
            double = False
        start = time.time()
        optimizer = self.optimizer([param for param in agent.parameters() if param.requires_grad == True], lr=learning_rate)
        print_loss_total = 0  
        print_loss_diff_mots_total = 0
        if return_errors : errors = {}
        iter = 1
        while iter < n_iters :
            if dic is not None :
                j = int(random.choice(list(dic.keys())))
                training_dialogue = dialogues[j]
                i = random.choice(dic[j])
                partie_dialogue = training_dialogue[:i+1-debut]
                target_answer = training_dialogue[i][1] if double else training_dialogue[i]
                loss, loss_diff_mots = self.trainLoop(agent, partie_dialogue, target_answer, optimizer)
                if return_errors and loss_diff_mots > 0 :
                    if j not in list(errors.keys()) : errors[j] = []
                    if i not in errors[j] : errors[j].append(i)
                # quantité d'erreurs sur la réponse i
                print_loss_total += loss
                print_loss_diff_mots_total += loss_diff_mots 
                iter += 1
                if iter % self.print_every == 0:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_diff_mots_avg = print_loss_diff_mots_total / self.print_every
                    print_loss_total = 0
                    print_loss_diff_mots_total = 0
                    print('%s (%d %d%%) %.4f %.2f' % (self.timeSince(start, iter / n_iters), iter, iter / n_iters * 100, 
                                                                  print_loss_avg, print_loss_diff_mots_avg))
            elif per_dialogue :
                j = int(random.choice(range(len(dialogues))))
                training_dialogue = dialogues[j]
                indices = list(range(debut, len(training_dialogue)))
                random.shuffle(indices)
                for i in indices :
                    partie_dialogue = training_dialogue[:i+1]
                    target_answer = training_dialogue[i][1] if double else training_dialogue[i]
                    loss, loss_diff_mots = self.trainLoop(agent, partie_dialogue, target_answer, optimizer)
                    if return_errors and loss_diff_mots > 0 :
                        if j not in list(errors.keys()) : errors[j] = []
                        if i not in errors[j] : errors[j].append(i)
                    # quantité d'erreurs sur la réponse i
                    print_loss_total += loss
                    print_loss_diff_mots_total += loss_diff_mots
                    iter += 1
                    if iter % self.print_every == 0:
                        print_loss_avg = print_loss_total / self.print_every
                        print_loss_diff_mots_avg = print_loss_diff_mots_total / self.print_every
                        print_loss_total = 0
                        print_loss_diff_mots_total = 0
                        print('%s (%d %d%%) %.4f %.2f' % (self.timeSince(start, iter / n_iters), iter, iter / n_iters * 100, 
                                                                      print_loss_avg, print_loss_diff_mots_avg))
            else :
                j = int(random.choice(range(len(dialogues))))
                training_dialogue = dialogues[j]
                i = random.choice(range(debut, len(training_dialogue)))
                partie_dialogue = training_dialogue[:i+1]
                target_answer = training_dialogue[i][1] if double else training_dialogue[i]
                loss, loss_diff_mots = self.trainLoop(agent, partie_dialogue, target_answer, optimizer)
                if return_errors and loss_diff_mots > 0 :
                    if j not in list(errors.keys()) : errors[j] = []
                    if i not in errors[j] : errors[j].append(i)
                # quantité d'erreurs sur la réponse i
                print_loss_total += loss
                print_loss_diff_mots_total += loss_diff_mots
                iter += 1
                if iter % self.print_every == 0:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_diff_mots_avg = print_loss_diff_mots_total / self.print_every
                    print_loss_total = 0
                    print_loss_diff_mots_total = 0
                    print('%s (%d %d%%) %.4f %.2f' % (self.timeSince(start, iter / n_iters), iter, iter / n_iters * 100, 
                                                                  print_loss_avg, print_loss_diff_mots_avg))


        if return_errors : return errors
                
                
    def ErrorCount(self, agent, dialogues):
        bound = 10
        ERRORS = [0 for i in range(bound +1)]
        repartitionError = {}
        for i in range(bound +1) :
            repartitionError[i] = []
        liste = []
        for k, input_dialogue in enumerate(dialogues):
            for l in range(len(input_dialogue)):
                if len(input_dialogue[l][1])>0 :
                    dialogue = input_dialogue[:l+1]
                    #target_answer = variableFromSentence(agent.output_lang, input_dialogue[l][1])
                    target_answer = input_dialogue[l][1]
                    target_answer = target_answer.to(self.device)
                    answer, agent_outputs, attn1_w, attn2_w, _ = agent.answerTrain(dialogue)
                    loss, loss_diff_mots = self.distance(agent_outputs, target_answer)
                    if loss_diff_mots > bound :
                        ERRORS = ERRORS + [0 for i in range(loss_diff_mots - bound)]
                        for i in range(bound +1, loss_diff_mots +1) :
                            repartitionError[i] = []
                        bound  = loss_diff_mots
                    ERRORS[loss_diff_mots] += 1
                    if loss_diff_mots > 0 :
                        liste.append([k, l, loss_diff_mots])
        for triple in liste:
            repartitionError[triple[2]].append(triple[:2])
        print("The repartition of errors :", ERRORS)
        return repartitionError


    def DialoguesWithErrors(self, agent, dialogues) :
        '''Returns a dictionnary, with indices of dialogues and index of line in dialogue
           where a mistake was made.
        '''
        start = time.time()
        Sortie = {}
        L = len(dialogues)
        for i, dialogue in enumerate(dialogues) :
            errs = []
            for j in range(len(dialogue)) :
                target_answer = dialogue[j][1]
                target_answer = target_answer.to(self.device)
                answer, agent_outputs, attn1_w, attn2_w, _ = agent.answerTrain(dialogue[:j+1], target_answer)
                loss, loss_diff_mots = self.distance(agent_outputs, target_answer)
                if loss_diff_mots > 0 :
                    errs.append(j)
            if errs != []:
                Sortie[i] = errs
            if (i+1) % self.print_every == 0:
                print('%s (%d %d%%)' % (self.timeSince(start, (i+1) / L),
                                             (i+1), (i+1) / L * 100))
        return Sortie
