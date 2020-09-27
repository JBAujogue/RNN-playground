
from __future__ import unicode_literals, print_function, division
import os
from io import open
import unicodedata
import string
import re
import random
import math
import json 
import time

import requests
import urllib

# package a installer d'abord avec anaconda
#import spacy

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# package a installer d'abord avec anaconda
import gensim
from gensim.models import KeyedVectors

#import nltk
#nltk.download()
#from nltk.tokenize import sent_tokenize, word_tokenize

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import unidecode
from unidecode import unidecode



TOKEN = "654278770:AAE4DqqsWp-dtS5pOxul82FCm484PP4vsw0"
URL = "https://api.telegram.org/bot{}/".format(TOKEN)
timeout = 100



use_cuda = False
print(torch.cuda.is_available())
# %matplotlib inline

import warnings
warnings.filterwarnings("ignore")

Generation = True # False pour du sélectif, True pour du génératif



#######################################################
# -------------------- 1.1 ---------------------------#
#######################################################

# -------------------- enlèvement des stopwords ---------------------------------------
def TrimWordsSentence(sentence, stopwords):
    '''Remove stopwords from a sentence'''
    resultwords = [word for word in sentence.split() if word.lower() not in stopwords]
    resultwords = ' '.join(resultwords)
    return resultwords

def TrimWordsDialogue(dialogue, stopwords):
    '''Remove stopwords from user utterances in a dialogue'''
    for pair in dialogue: 
        pair[0] = TrimWordsSentence(pair[0], stopwords)
        #pair[1] = pair[1].strip()
    return dialogue

def TrimWords(dialogues, stopwords):
    '''Remove stopwords from user utterances in a list of dialogues'''
    return [TrimWordsDialogue(dialogue, stopwords) for dialogue in dialogues ]

    
#-------------------------- Enleve les numéros en début de ligne -------------------
def enleveNumero(s):
    '''Remove line numbers in dialogue bAbI tasks dataset'''
    spl = s.split(' ')
    if spl[0].isdigit():
        spl = spl[1:] 
    return ' '.join(spl)


# --------------------------- Normalisation -------------------------------
def unicodeToAscii(s):
    """Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    '''Remove rare symbols from a string'''
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"[^a-zA-Z0-9?&\%\-\_]+", r" ", s) 
    return s 



#--------------------- import des dialogues --------------------
def importDialogues(fichier, limite = None):
    '''Import a textfile containing dialogues and returns a list, each element 
       corresponding to a dialogue and also being under the form of a list, with 
       each element being a list of two elements : an element giving a user 
       utterance and another element giving the bot response. Both elements are 
       normalized strings.
       Ex. The dialogue :
       
               hi    hello what can i help you with today
               can you book a table    i m on it
               
       now becomes :
       
              [['hi', 'hello what can i help you with today'], 
               ['can you book a table', 'i m on it']]
               
       Lines corresponding to user utterance with no bot response are discarted.
    '''
    dialogues_import = open(fichier, encoding='utf-8').read().strip().split('\n\n')
    dialogues = []
    for i, d in enumerate(dialogues_import):
        dialogue = []
        lines = d.split('\n')
        for l in lines:
            if len(l.split('\t')) == 2 :
                pair = [normalizeString(s) for s in l.split('\t')]
                pair[0] = enleveNumero(pair[0])
                dialogue.append(pair)
        dialogues.append(dialogue)
        if limite is not None and i == limite -1 :
            break

    return dialogues


def importVraisDialogues(fichier, limite = None):
    '''Applies to the dialogue bAbI tasks dataset. Import a textfile containing 
       dialogues in the form of a list, each element corresponding to a dialogue 
       and also being under the form of a list, with each element being a list 
       containing an element containing the user utterance and, whenever existant, 
       another element containing the bot response. All elements are normalized 
       strings.
       Ex. The dialogue :
       
               hi    hello what can i help you with today
               can you book a table    i m on it
               
       now becomes :
       
              [['hi', 'hello what can i help you with today'], 
               ['can you book a table', 'i m on it']]
    '''
    dialogues_import = open(fichier, encoding='utf-8').read().strip().split('\n\n')
    dialogues = []
    for i, d in enumerate(dialogues_import):
        dialogue = []
        lines = d.split('\n')
        for l in lines:
            if len(l.split('\t')) == 1 :
                pair0 = normalizeString(l)
                pair0 = enleveNumero(pair0)
                pair = [pair0, '' ]
                dialogue.append(pair)
            elif len(l.split('\t')) == 2 :
                pair = [normalizeString(s) for s in l.split('\t')]
                pair[0] = enleveNumero(pair[0])
                dialogue.append(pair)
        dialogues.append(dialogue)
        if limite is not None and i == limite -1 :
            break

    return dialogues



def importBabiTasks(fichier, limite = None):
    '''Applies to the bAbI tasks dataset. Import a textfile containing short texts
       in the form of a list, each element corresponding to a text and also being 
       under the form of a list, with each element containing an utterance of the text. 
       All elements are normalized strings.
       Ex. The text :
       
                The triangle is above the pink rectangle.
                The blue square is to the left of the triangle.
               
       now becomes :
       
              ['the triangle is above the pink rectangle', 
               'the blue square is to the left of the triangle']
    '''
    lines_import = open(fichier, encoding='utf-8').read().strip().split('\n')
    sortie = []
    text = []
    for i, line in enumerate(lines_import):
        sline= line.split('\t')
        if len(sline) == 1 :
            l = normalizeString(sline[0])
            l = enleveNumero(l)
            text.append(l)
        else :
            l = [normalizeString(el) for el in sline[ :2]]
            qa = [enleveNumero(el) for el in l]
            text_with_qa = text + qa
            sortie.append(text_with_qa)
            
        next_line_number = lines_import[i+1].split(' ')[0] if i < len(lines_import)-1 else '0'
        if next_line_number == '1' :
            text = []
                
    return sortie



# --------------------- Insertion des mots variables---------------------------
def modify(dialogues) :
    '''Applies to the dialogue bAbI tasks dataset. Formatting function 
       replacing all restaurant names, phone and address by one of the 
       three tokens 'option_name', 'option_phone' and 'option_address'.
    '''
    copie = list(dialogues)
    for dialogue in copie :
        optNum = 0
        for i in range(len(dialogue)) :
            utterance = dialogue[i][1]
            if utterance.startswith('what do you think of this option') :
                optNum += 1
                utterance = 'what do you think of this option ' + 'option' + str(optNum) + '_name'
            elif utterance.startswith('here it is') and utterance.endswith('phone') :
                utterance = 'here it is ' + 'option' + str(optNum) + '_phone'
            elif utterance.startswith('here it is') and utterance.endswith('address') :
                utterance = 'here it is ' + 'option' + str(optNum) + '_address'
            dialogue[i][1] = utterance
    return copie

#------------------ Dictionnaire des mots variables -----------------------------
def motVar(file):
    '''Applies to the Master's program dataset.
       Import the collection of pairs token-content for a set of variable words.
    '''
    lines = open(file, encoding='utf-8').read().strip().split('\n')
    motsVar = {}
    for l in lines :
        cle, valeur = l.split('\t')
        motsVar[cle.lower()] = valeur
    return motsVar


def ReplaceMotVar(motsVar, list_of_string):
    sentence = []
    for word in list_of_string :
        if word in motsVar.keys() :
            sentence.append(motsVar[word])
        else :
            sentence.append(word)
    return sentence


# ----------------------- Création de la liste des dialogues--------------------
def prepareData(opt):
    '''Import dialogue from text file and apply some formatting operations,
       as described in the functions 
               - importDialogues
               - modify
               - TrimWords
               - filterDialogues
    '''
    dialogues = importDialogues(fichier = opt['fichier'], 
                                limite = opt['limite'])
    dialogues = modify(dialogues) if opt['modify'] else dialogues
    dialogues = TrimWords(dialogues, opt['stopwords']) # on enlève les stopwords de chaque question
    print(" %s dialogues ..." % len(dialogues))
    print(dialogues[0])
    if opt['filtre'] :
        #for pair in [pair for pair in pairs if not filterPair(pair)]:
        #    print('%s (%d) -> %s (%d)' % (pair[0],len(pair[0].split()),pair[1],len(pair[1].split())))  
        dialogues = filterDialogues(dialogues, opt['max_length'])
        print('')
        print("... reduced to %s dialogues" % len(dialogues))

    return dialogues


def prepareVraieData(opt):
    '''Applies to the dialogue bAbI tasks dataset. Import dialogue from text 
       file and apply some formatting operations, as described in the functions 
               - importVraisDialogues
               - modify
               - TrimWords
               - filterDialogues
    '''
    dialogues = importVraisDialogues(opt['fichier'])
    dialogues = modify(dialogues) if opt['modify'] else dialogues
    dialogues = TrimWords(dialogues, opt['stopwords']) # on enlève les stopwords de chaque question
    print(" %s dialogues ..." % len(dialogues))
    print(dialogues[0])
    if opt['filtre'] :
        #for pair in [pair for pair in pairs if not filterPair(pair)]:
        #    print('%s (%d) -> %s (%d)' % (pair[0],len(pair[0].split()),pair[1],len(pair[1].split())))  
        dialogues = filterDialogues(dialogues, opt['max_length'])
        print('')
        print("... reduced to %s dialogues" % len(dialogues))

    return dialogues
    
    
motsVar = motVar('C:data/chatbot-M2-DS-Variables.txt')

#######################################################
# -------------------- 1.2 ---------------------------#
#######################################################


SOS_token = 0
EOS_token = 1
UNK_token = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"} if Generation else {}
        self.n_words = 3 if Generation else 0  # Counts SOS and EOS and UNK

        
    def addWord(self, word):
        '''Add a word to the language'''
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
            
    def addSentence(self, sentence):
        '''Add to the language all words of a sentence'''
        for word in sentence.split():
            self.addWord(word)
            
            
    def addDialogues(self, dialogues, i):
        '''Add to the language all words contained into : either all user utterances 
          (if i = 0) or all bot utterances (if i = 1), of a list of dialogues'''
        for dialogue in dialogues :
            for pair in dialogue:
                try :
                    self.addSentence(pair[i])
                except IndexError:
                    print("Problem with {}".format(pair))
					
					
					

def generateLanguages(dialogues):
    '''Generate three languages classes out of a list of dialogues :
            - input_lang containing the user's vocabulary
            - output lang containing the bot vocabulary
            - output_sentence_lang containing the bot answers as words of a vocabulary
    '''
    input_lang = Lang('questions')
    output_lang = Lang('answers')
    output_sentences_lang = Lang('sentences')
    
    input_lang.addDialogues(dialogues, 0)
    output_lang.addDialogues(dialogues, 1)
    for dialogue in dialogues :
        for pair in dialogue:
            output_sentences_lang.addWord(pair[1])
    print("Mots comptés :")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    print(output_sentences_lang.name, output_sentences_lang.n_words)
    
    return input_lang, output_lang, output_sentences_lang


def ajout(dialogues, lang, i= 1):
    '''addDialogues method of the Lang class with prints.'''
    lang.addDialogues(dialogues, i)
    print(lang.name, lang.n_words)
    return lang 


def ajoutSentences(dialogues, sentences_lang, i = 1) :
    '''Add sentences as words to a given language'''
    for dialogue in dialogues :
        for pair in dialogue :
            try :
                sentences_lang.addWord(pair[i])
            except IndexError:
                print("Problem with {}".format(pair))
                
    return sentences_lang


#######################################################
# -------------------- 1.3 ---------------------------#
#######################################################



# ----------------- Remplacement des mots par leurs indices  ---------------------

def indexesFromSentence(lang, sentence, max_length = 50):
    '''Turn a given sentence into a list of indices according to a given language'''
    indexes=[]
    unknowns = 0
    for word in sentence.split(' '):
        if word in set(lang.word2index):
            indexes.append(lang.word2index[word])
        else:
            #print("Unknwon word in model vocab: ",word)
            indexes.append(UNK_token)
    # remove exceeding words, exept first word and the two last words or symbols
    while len(indexes) > max_length:
        indexes.pop(random.randint(1,len(indexes)-2))
    return indexes


def variableFromSentence(lang, sentence, max_length = 50):
    '''Turn a sentence into a torch variable, containing a list of indices according
       to a given language.
    '''
    indexes = indexesFromSentence(lang, sentence, max_length) # turn string into list of indices
    indexes.append(EOS_token)                                 # append EOS_token index
    result = Variable(torch.LongTensor(indexes).view(-1, 1))  # wrap the list into torch variable
    return result


def selectedVariableFromSentence(lang, sentence):
    '''Turn a sentence into a torch variable, containing the index of this sentence
       according to a given language.
    '''
    index = []
    index.append(lang.word2index[sentence])
    result = Variable(torch.LongTensor(index).view(-1, 1))
    return result


def variablesFromDialogue(input_lang, output_lang, dialogue, max_length = 50):
    '''Turn a whole dialogue into a list of torch tensors, according to an input
       and an output languages'''
    sortie = []
    for pair in dialogue :
        input_variable = variableFromSentence(input_lang, pair[0], max_length)
        target_variable = variableFromSentence(output_lang, pair[1], max_length)
        sortie.append((input_variable, target_variable))
    return sortie



# ----------------- Récupérer une phrase a partir des indices ---------------------
def SentenceFromIndexes(input_lang, indexes):
    '''Turn a list of indices into a sentence according to a given language.'''
    sentence = []
    for i in indexes:
        sentence.append(input_lang.index2word[i])
    sentence = ' '.join(sentence)
    return sentence



#######################################################
# --------------------- 2 ----------------------------#
#######################################################




#######################################################
# -------------------- 3.1 ---------------------------#
#######################################################



class Encoder_biGRU(nn.Module):
    '''Module performing reading of a sentence, and returns :
    
            - embedding vector of each word
            - hidden vector of each word
            - the last two hidden states of the bi-GRU module
    '''
    def __init__(self, 
                 input_lang_size, 
                 embedding_dim, 
                 hidden_dim, 
                 pre_entrainement = None, 
                 freeze = False, 
                 tag = False, 
                 n_layers=1
                ): 
        super(Encoder_biGRU, self).__init__()
        
        # relevant quantities
        self.bi_gru = True
        self.input_lang_size = input_lang_size # size of word vocabulary
        self.embedding_dim = embedding_dim     # dimension of embedded words
        self.output_dim = hidden_dim * 2       # dimension of contextual rep. of words and utterance
        self.hidden_dim = hidden_dim           # dimension of hidden state of GRUs
        self.tag = tag
        self.n_layers = n_layers

        
        # embedding module, performing both one-hot encoding and embedding of a word index
        self.embedding = nn.Embedding(input_lang_size, embedding_dim)
        if pre_entrainement is not None :
            self.embedding.weight = nn.Parameter(torch.Tensor(pre_entrainement)) 
        if freeze :
            self.embedding.weight.requires_grad = False
        
        # bidirectionnal GRU module
        if tag :
            # TODO
            self.tag_dim = tag_dim
            self.embedding_tag = nn.Embedding(tag_dim, tag_dim)
            self.embedding_tag.weight = nn.Parameter(torch.eye(tag_dim))
            self.gru = nn.GRU(embedding_dim + tag_dim, hidden_dim)
            self.gru2 = nn.GRU(embedding_dim + tag_dim, hidden_dim)
        else :
            self.gru = nn.GRU(embedding_dim, hidden_dim)
            self.gru2 = nn.GRU(embedding_dim, hidden_dim)
          
        
    def initHidden(self): 
        result = Variable(torch.zeros(1, 1, self.hidden_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result
 
    def forward(self, utterance, hidden, hidden2):
        '''takes as parameters : 
        
                an utterance as list of indices,            dim = (input_length, 1)
                a state initializing the forward GRU        dim = (1, 1, hidden_dim)
                a state initializing the backward GRU       dim = (1, 1, hidden_dim)
        
           returns the tensors : 
        
                the word embeddings,                        dim = (input_length, 1, embedding_dim)
                the word hidden representations,            dim = (input_length, 2*embedding_dim)
                the last hidden state of the forward GRU    dim = (1, 1, hidden_dim)
                the last hidden state of the backward GRU   dim = (1, 1, hidden_dim)
        '''
        input_length = utterance.size()[0]
        embeddings = self.embedding(utterance)                          # dim = (input_length, 1, embedding_dim)
        hiddens1 = Variable(torch.zeros(input_length, self.hidden_dim)) # dim = (input_length, embedding_dim)
        hiddens2 = Variable(torch.zeros(input_length, self.hidden_dim)) # dim = (input_length, embedding_dim)
        for i in range(input_length):
            k = input_length - 1 - i
            for j in range(self.n_layers):
                word = embeddings[i].view(1, 1, -1)
                output, hidden = self.gru(word, hidden)
                hiddens1[i] = hidden
                word2 = embeddings[k].view(1, 1, -1)
                output2, hidden2 = self.gru2(word2, hidden2)
                hiddens2[k] = hidden2
        hiddens = torch.cat((hiddens1, hiddens2), dim = 1)              # dim = (input_length, 2*embedding_dim)
        return embeddings, hiddens, hidden, hidden2


#######################################################
# -------------------- 3.2 ---------------------------#
#######################################################


class R_Attn_plus(nn.Module):
    '''Module performing reccurent attention over a sequence of vectors stored in
       a memory block, conditionned by some vector. At instanciation it takes as imput :
       
                - input_variable_dim : the dimension of the conditionning vector
                - attention_targets_dim : the dimension of vectors stored in memory
    '''
    def __init__(self, input_variable_dim, attention_targets_dim): 
        super(R_Attn_plus, self).__init__()
        
        # relevant quantities
        self.input_variable_dim = input_variable_dim           # dimension of the conditionning variable
        self.attention_targets_dim = attention_targets_dim     # dimension of vectors forming the attention target
        self.output_dim = attention_targets_dim                # dimension of output vector
        self.version = 'A_GRU_plus'
        
        # modules
        self.W_1 = nn.Linear(input_variable_dim + attention_targets_dim, attention_targets_dim)
        self.W_r = nn.Linear(input_variable_dim + attention_targets_dim, attention_targets_dim)
        self.U   = nn.Linear(input_variable_dim, attention_targets_dim, bias = False)
        self.W_2 = nn.Linear(attention_targets_dim, 1)
        self.W   = nn.Linear(attention_targets_dim, attention_targets_dim)

        
    def initHidden(self): 
        result = Variable(torch.zeros(1, self.output_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result
        

    def forward(self, input, attention_targets = None):
        '''takes as parameters : 
        
                an input tensor conditionning the attention,    dim = (1, 1, input_variable_dim)
                a tensor containing attention targets           dim = (attention_targets_length, attention_targets_dim)
        
           returns : 
        
                the resulting tensor of the attention process,  dim = (1, 1, attention_targets_dim)
                the attention weights,                          dim = (1, attention_targets_length)
        '''
        if attention_targets is not None :
            L = attention_targets.size()[0]
            
            # calcul des portes
            input = input.squeeze(0)                                       # dim = (1, input_variable_dim)
            expand_input = input.repeat(L, 1)                              # dim = (L, input_variable_dim)
            z = torch.cat((expand_input, attention_targets), 1)            # dim = (L, input_variable_dim + attention_targets_dim)
            z_1 = self.W_1(z)
            z_2 = F.tanh(z_1)
            z_3 = self.W_2(z_2)
            attn = F.softmax(z_3, dim = 0)                                 # dim = (L, 1)
            attn_weights = torch.transpose(attn, 0,1)                      # dim = (1, L)
            
            # lecture du GRU
            hidden = self.initHidden()
            for i in range(L):
                r = F.sigmoid(self.W_r(z[i]))                              # dim = (1, attention_targets_dim)
                l = F.tanh(self.W(attention_targets[i])+ r*self.U(input )) # dim = (1, attention_targets_dim)
                p = attn[i]
                hidden = p*l + hidden - p*hidden                           # dim = (1, attention_targets_dim)
            hidden = hidden.unsqueeze(0)                                   # dim = (1, 1, attention_targets_dim)
            
            return hidden, attn_weights
            
        else :
            return input, None


class A_Attn(nn.Module):
    '''Module performing additive attention over a sequence of vectors stored in
       a memory block, conditionned by some vector. At instanciation it takes as imput :
       
                - version : 1 (location) or 2 (general) or 2.5 (multidim general) or 3 (concat)
                - input_variable_dim : the dimension of the conditionning vector
                - attention_targets_dim : the dimension of vectors stored in memory
    '''
    def __init__(self, version, input_variable_dim, attention_targets_dim): 
        super(A_Attn, self).__init__()
        
        # relevant quantities
        self.version = version
        self.input_variable_dim = input_variable_dim
        self.attention_targets_dim = attention_targets_dim
        self.output_dim = attention_targets_dim
        
        # modules
        if version == 1:
            self.attn = nn.Linear(input_variable_dim, 50) #attention_targets_dim_length)
        elif version == 2:
            self.attn = nn.Linear(input_variable_dim + attention_targets_dim, attention_targets_dim)
            self.attn_v = nn.Linear(attention_targets_dim, 1, bias = False)
        elif version == 2.5:
            self.attn = nn.Linear(input_variable_dim + attention_targets_dim, attention_targets_dim)
            self.attn_v = nn.Linear(attention_targets_dim, attention_targets_dim)            
        elif version == 3 :
            self.attn = nn.Linear(input_variable_dim, attention_targets_dim)
        self.act = F.softmax
        

    def forward(self, input, attention_targets = None):
        '''takes as parameters : 
        
                an input tensor conditionning the attention,    dim = (1, 1, input_variable_dim)
                a tensor containing attention targets           dim = (attention_targets_length, attention_targets_dim)
        
           returns : 
        
                the resulting tensor of the attention process,  dim = (1, 1, attention_targets_dim)
                the attention weights,                          dim = (1, attention_targets_length)
        '''
        if attention_targets is not None :
            if self.version == 1 :
                attn_weights = self.attn(input).squeeze(0)
                attn_weights = self.act(attn_weights)
                attn_applied = torch.bmm(attn_weights.unsqueeze(0), attention_targets.unsqueeze(0))

            elif self.version == 2:
                input = input.squeeze(0)
                expand_input = input.repeat(attention_targets.size()[0], 1)       #(L, input_variable_dim)
                poids = torch.cat((expand_input, attention_targets), 1)           #(L, input_variable_dim + attention_targets_dim)
                poids = self.attn(poids)                                          #(L, attention_targets_dim)
                poids = F.tanh(poids)
                attn_weights = self.attn_v(poids)                                 #(L, 1)
                attn_weights = torch.transpose(attn_weights, 0,1)                 #(1, L)
                attn_weights = self.act(attn_weights)
                attn_applied = torch.bmm(attn_weights.unsqueeze(0), attention_targets.unsqueeze(0))

            elif self.version == 2.5:
                input = input.squeeze(0)
                expand_input = input.repeat( attention_targets.size()[0] , 1)
                poids = torch.cat((expand_input , attention_targets), 1)
                poids = self.attn(poids)
                poids = F.tanh(poids)
                attn_weight_vectors = self.attn_v(poids)
                attn_weight_vectors = torch.transpose(attn_weight_vectors, 0,1)
                attn_weight_vectors = self.act(attn_weight_vectors)
                attn_applied = torch.diag(torch.mm(attn_weight_vectors, attention_targets)).view(1,1, -1)
                attn_weights = attn_weight_vectors

            elif self.version == 3 :
                attn_weights = torch.mm(self.attn(input).squeeze(0), torch.transpose(attention_targets, 0,1))
                attn_weights = self.act(attn_weights)
                attn_applied = torch.bmm(attn_weights.unsqueeze(0), attention_targets.unsqueeze(0))
        else :
            attn_applied = input
            attn_weights = None
        
        return attn_applied, attn_weights



#######################################################
# -------------------- 3.3 ---------------------------#
#######################################################


class Tracker_biGRU(nn.Module):
    '''Performs information tracking over memory blocks given a query vector h_q,
       and returns a decision vector h'_q
    '''
    def __init__(self, 
                 query_dim, 
                 memory_content_dim,
                 Hencoder_hidden_dim,
                 v_attn1 = 0,
                 v_attn2 = 0,
                 mix = True
                ):
        super(Tracker_biGRU, self).__init__()
        
        # relevant quantities
        self.query_dim = query_dim
        self.memory_content_dim = memory_content_dim
        self.Hencoder_input_dim = self.memory_content_dim if v_attn1 is not 0 else self.query_dim
        self.Hencoder_hidden_dim = Hencoder_hidden_dim
        self.context_vector_dim = Hencoder_hidden_dim * 2
        self.mix = mix
        self.output_dim = query_dim if mix else self.context_vector_dim
        
        
        # first attention module
        if type(v_attn1) is not float and type(v_attn1) is not int :
            self.attn1 = v_attn1
            self.v_attn1 = self.attn1.version
        elif v_attn1 != 0 :
            self.v_attn1 = v_attn1
            self.attn1 = A_Attn(v_attn1, 
                                self.query_dim, 
                                self.memory_content_dim)
        else :
            self.v_attn1 = v_attn1
        
        
        # intermediate encoder module
        self.Hencoder  = nn.GRU(self.Hencoder_input_dim, self.Hencoder_hidden_dim)
        self.Hencoder2 = nn.GRU(self.Hencoder_input_dim, self.Hencoder_hidden_dim)
        
        
        # second attention module
        if type(v_attn2) is not float and type(v_attn2) is not int :
            self.attn2 = v_attn2
            self.v_attn2 = self.attn2.version
        else :
            self.v_attn2 = v_attn2
            self.attn2 = A_Attn(v_attn2, 
                                self.query_dim, 
                                self.Hencoder_hidden_dim*2)
        
        
        # module performing mixing between query and context vectors
        if mix :
            self.H = nn.Linear(self.context_vector_dim, self.query_dim, bias = False)
            #self.H = nn.Linear(computeHanDim(han_list) + 2 * encoder.hidden_dim, 2 * encoder.hidden_dim)
            #self.H = nn.Linear(computeHanDim(han_list), 2 * encoder.hidden_dim)
        
                
    def initHidden(self): 
        result = Variable(torch.zeros(1, 1, self.Hencoder_hidden_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result
        
        
    def forward(self, 
                query_vector, 
                memory = None, 
                memory_queries = None
               ):
        '''takes as parameters : 
        
                a query vector,                                 dim = (1, 1, query_dim)
                a tensor containing memory vectors              dim = (memory_length, memory_content_dim)
                a tensor containing past queries                dim = (memory_length, query_dim)
        
           returns : 
        
                the resulting decision vector                   dim = (1, 1, query_dim)
                the weights of first attention layer (dict)     
                the weights of second attention layer (dict)
        '''
        memory_length = len(memory)
        if memory_length == 0 :
            decision_vector = Variable(torch.zeros(1, 1, self.output_dim))
            attn1_attention_weights = None
            attn2_attention_weights = None
        else :
            L = memory_length
            memory_queries_V = Variable(torch.zeros(L, self.memory_content_dim))
            for i in range(memory_length) :
                memory_queries_V[i] = memory_queries[i]
            attn1_attention_weights = {}
            attn2_attention_weights = torch.zeros(L)
            query_vector = query_vector.view(1, 1, -1)
            Hencoder_hidden = self.initHidden()
            Hencoder_hidden2 = self.initHidden()
            Hencoder_inputs = Variable(torch.zeros(L, self.Hencoder_input_dim))
            attn2_inputs = Variable(torch.zeros(L, self.Hencoder_hidden_dim))
            
            
            # first attention layer
            for i in range(L) :
                if self.v_attn1 != 0:
                    attn1_output, attn1_weights = self.attn1(input = query_vector, attention_targets = memory[i])
                    attn1_attention_weights[i] = attn1_weights.data
                    Hencoder_inputs[i] = attn1_output
                else :
                    Hencoder_inputs[i] = memory_queries_V[i]
                    attn1_attention_weights = None

                        
            # intermediate biGRU
            Hencoder_hiddens = Variable(torch.zeros(L, self.Hencoder_hidden_dim))
            Hencoder_hiddens2 = Variable(torch.zeros(L, self.Hencoder_hidden_dim))
            for i in range(L) :
                Hencoder_input = Hencoder_inputs[i].view(1, 1, -1)
                Hencoder_input2 = Hencoder_inputs[L - 1 - i].view(1, 1, -1)
                Hencoder_input, Hencoder_hidden = self.Hencoder(Hencoder_input, Hencoder_hidden)
                Hencoder_hiddens[i] = Hencoder_hidden.view(-1)
                Hencoder_input2, Hencoder_hidden2 = self.Hencoder(Hencoder_input2, Hencoder_hidden2)
                Hencoder_hiddens2[L - 1 - i] = Hencoder_hidden2.view(-1)   
                

            # second attention layer
            if self.v_attn2 != 0:
                attn2_inputs = torch.cat((Hencoder_hiddens, Hencoder_hiddens2), dim = 1) 
                context_vector, attn2_weights = self.attn2(input = query_vector, attention_targets = attn2_inputs)
                attn2_attention_weights = attn2_weights.data
            else :
                context_vector = torch.cat((Hencoder_hidden, Hencoder_hidden2), dim = 2) 
                attn2_attention_weights = None

                
            # form decision vector
            if self.mix :
                decision_vector = query_vector + self.H(context_vector)
                #decision_vector = F.tanh(self.H(torch.cat((decision_vector, context_vector), 2)))
                #decision_vector = (decision_vector + F.tanh(self.H(context_vector)))/2  
            else :
                decision_vector = context_vector
  
        
        # output decision vector
        return decision_vector, attn1_attention_weights, attn2_attention_weights





#######################################################
# -------------------- 3.4 ---------------------------#
#######################################################



class Decoder(nn.Module):
    '''Performs answer decoding given a decision vector yield by the tracker'''
    def __init__(self, 
                 query_dim,
                 decision_dim, 
                 output_dim, 
                 output_lang_size,
                 generation = True,
                 pre_entrainement = None, 
                 freeze = False,
                 tag = False, 
                 dropout_p=0.1, 
                 n_layers = 1
                ):
        super(Decoder, self).__init__()
        
        
        # relevant quantities
        self.generation = generation
        self.n_layers = n_layers    
        self.decision_dim = decision_dim
        
        
        # modules
        self.embedding = nn.Embedding(output_lang_size, output_dim)
        self.dropout = nn.Dropout(dropout_p)
        if generation :
            self.gru = nn.GRU(output_dim + query_dim, decision_dim)
            self.out = nn.Linear(decision_dim, output_lang_size)
        else :
            self.out = nn.Linear(decision_dim, output_lang_size)
        if pre_entrainement is not None :
            self.embedding.weight = nn.Parameter(torch.Tensor(pre_entrainement))   
        if freeze :
            self.embedding.weight.requires_grad = False 
            
            
    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.decision_dim))
        return result
        
        
    def forward(self, input, query_vector, decoder_hidden_vector):
        '''takes parameters : 
        
                the index of last decoded word as input                  
                the query vector                                dim = (1, 1, decision_dim)
                the last hidden decoder state                   dim = (1, 1, decision_dim)
        
           returns : 
        
                the index of next decoded word
                the updated hidden decoder state                 dim = (1, 1, decision_dim)
        '''
        if self.generation :
            embedded = self.embedding(input)
            embedded = torch.cat((embedded, query_vector), dim = 2)
            output = self.dropout(embedded)
            for i in range(self.n_layers):
                output, hidden = self.gru(output, decoder_hidden_vector)
            output = F.log_softmax(self.out(output[0]))
        else :
            output = F.log_softmax(self.out(decoder_hidden_vector), dim = 2)
            output = output.squeeze(0)
            hidden = None
        
        return output, hidden


#######################################################
# -------------------- 3.5 ---------------------------#
#######################################################


class Agent_biGRU(nn.Module):
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
    def __init__(self, 
                 input_lang, 
                 output_lang, 
                 encoder, 
                 tracker, 
                 decoder, 
                 output_sentences_lang = None, 
                 hops = 0,
                 separation = False
                ):
        
        super(Agent_biGRU, self).__init__()
        
        # relevant quantities
        self.hops = hops
        self.memory_dim = encoder.output_dim
        self.separation = separation # A boolean setting whether past utterances are encoded jointly or separately, that is,
                                     # whether last hidden states of the encoder after an utterance encoding is re-used for
                                     # next utterance encoding.
        self.memory_length = 0
        self.input_lang = input_lang                         # input language
        self.output_lang = output_lang                       # output language
        self.output_sentences_lang  = output_sentences_lang  # output sentences as language
        
        # modules        
        self.encoder = encoder
        self.tracker = tracker
        self.decoder = decoder
        
        
    def initMemory(self):
        """Initialize memory slots"""
        self.memory = {}
        self.memory_queries = {}
        self.query_hidden1 = self.encoder.initHidden()
        self.query_hidden2 = self.encoder.initHidden()
        self.memory_length = 0
        
    
    def updateMemory(self, last_words, query_hidden1, query_hidden2):
        """Update memory with a list of word vectors 'last_words' and the last query vector 'last_query'"""
        self.memory[self.memory_length] = last_words
        self.memory_queries[self.memory_length] = torch.cat((query_hidden1, query_hidden2), dim = 2)
        if self.separation == False :
            self.query_hidden1 = query_hidden1
            self.query_hidden2 = query_hidden2
        self.memory_length += 1
        
        
    def readUtterance(self, utterance):
        """Perform reading of an utterance, returning created word vectors
           and last hidden states of teh encoder bi-GRU
        """
        embeddings, last_words, query_hidden1, query_hidden2 = self.encoder(utterance, self.query_hidden1, self.query_hidden2)
        return last_words, query_hidden1, query_hidden2
        
        
    def readDialogue(self, dialogue):
        """Loop of readUtterance over a whole dialogue
        """
        for i in range(len(dialogue)) :
            for j in range(2):
                utterance = dialogue[i][j]
                last_words, query_hidden1, query_hidden2 = self.readUtterance(utterance)
                self.updateMemory(last_words, query_hidden1, query_hidden2)
   
        
    def tracking(self, query_hidden):
        """Détermine un vecteur d'attention sur les éléments du registre de l'agent,
        sachant un vecteur 'very_last_hidden', et l'accole à ce vecteur """
        decision_vector = query_hidden
        attn1_attention_weights = [] 
        attn2_attention_weights = []
        for h in range(self.hops) :
            decision_vector, attn1, attn2 = self.tracker(decision_vector, self.memory, self.memory_queries)
            attn1_attention_weights.append(attn1)
            attn2_attention_weights.append(attn2)
    
        return decision_vector, attn1_attention_weights, attn2_attention_weights

    
    def generateAnswer(self, query_vector, decision_vector, target_answer = None) :
        """Génère une réponse à partir d'un état caché initialisant le décodeur,
        en utilisant une réponse cible pour un mode 'teacher forcing-like' si celle-ci est fournie """
        bound = 30
        decoder_outputs = []
        answer = []
        di = 0
        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_hidden = decision_vector
        for di in range(bound) :
            decoder_output, decoder_hidden = self.decoder(decoder_input,  # index of last generated word
                                                          query_vector,   # query_vector
                                                          decoder_hidden) # last decoder hidden state 
            topv, topi = decoder_output.data.topk(1)
            decoder_outputs.append(decoder_output)
            ni = topi[0][0] # index of current generated word
            answer.append(ni)
            if ni == EOS_token :
                break
            elif target_answer is not None : # Teacher forcing
                if di < target_answer.size()[0] :
                    decoder_input = target_answer[di].view(1,-1)  
                else :
                    break
            else :
                decoder_input = Variable(torch.LongTensor([[ni]])) # !!! Il y a une cassure dans la variable
 
        return answer, decoder_outputs


    def selectAnswer(self, decision_vector) :
        decoder_outputs = []
        decoder_output, decoder_hidden = self.decoder(None, decision_vector, None)   
        decoder_outputs.append(decoder_output)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        answer =[ni] 
        
        return answer, decoder_outputs
        
        
    # ------------ 1st working mode : training mode ------------
    def answerTrain(self, input, target_answer = None):
        """Parameters are a complete dialogue, containing the current query and answer, and of the form

                    [['query 1', 'answer 1'],
                     ['query 2', 'answer 2'],
                             ..........
                     ['current query', 'current answer']]

           The model learns to generate the current answer. 
           Teacher forcing can be enabled by passing the ground answer though the 'target_answer' option. 
           Attention weights over words and past utterances can be provided with the 'provideAttention' option."""
        
        # 1) initiates memory instance
        #
        self.initMemory()
        
        
        # 2) reads historical part of dialogue (if applicable),
        # word vectors and last hidden states of encoder bi-GRU are stored in memory
        #
        dialogue = variablesFromDialogue(self.input_lang, self.output_lang, input[:-1])
        self.readDialogue(dialogue)
            
            
        # 3) reads current utterance,
        # returns word vectors of query and query vector
        #
        query = input[-1][0] # a string
        query = variableFromSentence(self.input_lang, query) # torch variable containing list of word indices
        last_words, query_hidden1, query_hidden2 = self.readUtterance(query)
        query_hidden = torch.cat((query_hidden1, query_hidden2), dim = 2)
        
        
        # 4) performs tracking
        # returns decision vector
        #
        if self.tracker is not None :
            decision_vector, attn1_attention_weights, attn2_attention_weights = self.tracking(query_hidden)
        else :
            decision_vector = query_hidden
            attn1_attention_weights = None
            attn2_attention_weights = None
            
            
        # 5) response generation
        # returns list of indices
        #
        if Generation :
            answer, decoder_outputs = self.generateAnswer(query_hidden, decision_vector, target_answer)
        else :
            answer, decoder_outputs = self.selectAnswer(decision_vector)
            
            
        # 6) returns answer
        #
        return answer, decoder_outputs, attn1_attention_weights, attn2_attention_weights   

        
        
    # ------------ 2nd working mode : test mode ------------
    def forward(self, input):
        """Parameters are a single current query as string, and the model learns to generate the current answer. 
           Attention weights over words and past utterances can be provided with the 'provideAttention' option."""
        
        # 1) initiates memory and hidden states of encoder bi-GRU if conversation starts
        #
        if self.memory_length == 0 :
            self.initMemory()
            
            
        # 2) reads current utterance,
        # returns word vectors of query and query vector
        #
        query = input # a string
        query = variableFromSentence(self.input_lang, query) # torch variable containing list of word indices
        last_words, query_hidden1, query_hidden2 = self.readUtterance(query)
        query_hidden = torch.cat((query_hidden1, query_hidden2), dim = 2)
        
        
        # 3) performs tracking
        # returns decision vector
        #
        if self.tracker is not None :
            decision_vector, attn1_attention_weights, attn2_attention_weights = self.tracking(query_hidden)
        else :
            decision_vector = query_hidden
            attn1_attention_weights = None
            attn2_attention_weights = None
            
            
        # 4) response generation
        # returns list of indices
        #
        if Generation :
            answer, decoder_outputs = self.generateAnswer(query_hidden, decision_vector)
        else :
            answer, decoder_outputs = self.selectAnswer(decision_vector)
            
            
        # 5) updates memory with current query and answer
        #
        self.updateMemory(last_words, query_hidden1, query_hidden2)
        answer_var = Variable(torch.LongTensor(answer).view(-1, 1))  # !!! Cassure dans la variable
        last_words, query_hidden1, query_hidden2 = self.readUtterance(answer_var)
        self.updateMemory(last_words, query_hidden1, query_hidden2)
            
        
        # 6) returns answer
        #
        return answer, decoder_outputs, attn1_attention_weights, attn2_attention_weights


		
		
		
#######################################################
# -------------------- 3.6 ---------------------------#
#######################################################	
		
		
def create_agent(input_lang,               ###
                 embedding_dim,              #
                 hidden_dim,                 # --- Encoder options
                 biGRU_encoding,             #
                 separation,               ###
                 
                 v_attn1,                  ###
                 Hencoder_hidden_dim,        #
                 biGRU_tracking,             #
                 Tracker_opt,                # --- Tracker options
                 v_attn2,                    #
                 mix,                        #
                 hops,                     ###
                 
                 output_dim,               ###
                 output_lang,                # --- Decoder options
                 output_sentences_lang,      # 
                 generation                ###
                ):
    '''Create an agent with specified dimensions and specificities'''
    
    input_lang_size = input_lang.n_words
    if biGRU_encoding :
        encoder = Encoder_biGRU(input_lang_size, 
                                embedding_dim, 
                                hidden_dim, 
                                pre_entrainement = None, 
                                freeze = False, 
                                tag = False, 
                                n_layers=1) 
    else :
        encoder = Encoder_monoGRU(input_lang_size, 
                                  embedding_dim, 
                                  hidden_dim, 
                                  pre_entrainement = None, 
                                  freeze = False, 
                                  tag = False, 
                                  n_layers=1)   
    
    query_dim = 2 * hidden_dim if biGRU_encoding else hidden_dim
    memory_content_dim = 2 * hidden_dim if biGRU_encoding else hidden_dim
    
    if v_attn1 == 'R_Attn_plus' :
        v_attn1 = R_Attn_plus(query_dim, memory_content_dim)
    if v_attn2 == 'R_Attn_plus' :
        v_attn2 = R_Attn_plus(query_dim, memory_content_dim)
    
    if biGRU_tracking : 
        tracker = Tracker_biGRU(query_dim,
                                memory_content_dim,  
                                Hencoder_hidden_dim, 
                                v_attn1, 
                                v_attn2,
                                mix)
    else :
        tracker = Tracker_monoGRU(query_dim,
                                  memory_content_dim,  
                                  Hencoder_hidden_dim, 
                                  v_attn1,
                                  v_attn2,
                                  mix)
        
    decision_dim = tracker.output_dim 
    output_lang_size = output_lang.n_words
    decoder = Decoder(query_dim, 
                      decision_dim,
                      output_dim, 
                      output_lang_size,
                      generation,
                      pre_entrainement = None, 
                      freeze = False, 
                      tag = False,
                      dropout_p=0.1, 
                      n_layers = 1)
    
    if biGRU_encoding :
        agent = Agent_biGRU(input_lang, 
                            output_lang, 
                            encoder, 
                            tracker, 
                            decoder,
                            output_sentences_lang,
                            hops,
                            separation)
    else :
        agent = Agent_monoGRU(input_lang, 
                              output_lang, 
                              encoder, 
                              tracker, 
                              decoder,
                              output_sentences_lang,
                              hops,
                              separation)
        
        
    print("Agent's number of parameters : {} ".format(nbParametres(agent)))
    return agent

    
#######################################################
# -------------------- 5.2 ---------------------------#
#######################################################


def nbParametres(model) :
    count = 0
    for p in model.parameters():
        if p.requires_grad == True :
            count += p.data.nelement()
    return count


def computeHanNbParameters(han_list) :
    nbParametres_han_list = 0
    for han in han_list :
        nbParametres_han_list += nbParametres(han)
    return nbParametres_han_list


#######################################################
# --------------------- 6 ----------------------------#
#######################################################


MAX_LENGTH = 30
max_length = MAX_LENGTH
tag_dim = 17
output_dim = 200
embedding_dim = 200
stopwords = []

stopwords = []

Master_train =   {'fichier': 'C:data/Liste_Dialogues_trn.txt', #dialog-babi-task1-API-calls-trn.txt
                   'modify' : False,
                   'stopwords' : stopwords,
                   'max_length' : MAX_LENGTH ,
                   'limite' : None,
                   'filtre' : False}

dialogues_Master = prepareData(Master_train)

Master_test  =   {'lang1': 'lang_client',
                   'lang2' : 'lang_agent',
                   'fichier': 'C:data/Liste_Dialogues_tst.txt', #dialog-babi-task1-API-calls-trn.txt
                   'modify' : False,
                   'stopwords' : stopwords,
                   'max_length' : MAX_LENGTH ,
                   'limite' : None,
                   'reverse' : False,
                   'filtre' : False}

dialogues_Master_test = prepareData(Master_test)


input_Master_l, output_Master_l, output_sentences_Master_l = generateLanguages(dialogues_Master)
input_Master_l = ajout(dialogues_Master_test, input_Master_l, 0)
Master_l = ajout(dialogues_Master_test, input_Master_l, 1)



# HDMN (our model)
agent_Master = create_agent(input_lang = Master_l,       
                     embedding_dim = 200,              
                     hidden_dim = 100,         
                     biGRU_encoding = True,             
                     separation = False,            
                 
                     v_attn1 =  'R_Attn_plus',                  
                     Hencoder_hidden_dim = 100,        
                     biGRU_tracking = True,             
                     Tracker_opt = None,              
                     v_attn2 =  'R_Attn_plus',                    
                     mix = True,                        
                     hops = 3,                
                 
                     output_dim = 200,           
                     output_lang = Master_l,        
                     output_sentences_lang = output_sentences_Master_l, 
                     generation = True      
                    )
					
					
agent_Master_add = create_agent(input_lang = Master_l,       
                     embedding_dim = 200,              
                     hidden_dim = 100,         
                     biGRU_encoding = True,             
                     separation = False,            
                 
                     v_attn1 = 2, # 'R_Attn_plus',                  
                     Hencoder_hidden_dim = 100,        
                     biGRU_tracking = True,             
                     Tracker_opt = None,              
                     v_attn2 = 2, # 'R_Attn_plus',                    
                     mix = True,                        
                     hops = 3,                
                 
                     output_dim = 200,           
                     output_lang = Master_l,        
                     output_sentences_lang = output_sentences_Master_l, 
                     generation = True      
                    )



#agent_Master.load_state_dict(torch.load('saves/agent_Master.pth'))
#agent_Master_add.load_state_dict(torch.load('saves/agent_Master_add.pth'))

agent_Master.load_state_dict(torch.load('saves/agent_Master_renforced.pth'))
agent_Master_add.load_state_dict(torch.load('saves/agent_Master_add_renforced.pth'))

def answer(text) :
    question = normalizeString(text)
    #generated_answer, decoder_outputs, attn1_attention_weights, attn2_attention_weights = agent_Master(question)
    generated_answer_add, decoder_outputs, attn1_attention_weights, attn2_attention_weights = agent_Master_add(question)
    if Generation :
        #reponse_generee =[agent_Master.output_lang.index2word[ind.item()] for ind in generated_answer[:-1]]
        reponse_generee_add =[agent_Master_add.output_lang.index2word[ind.item()] for ind in generated_answer_add[:-1]]
        
    else :
        #reponse_generee =[agent_Master.output_sentences_lang.index2word[ind.item()] for ind in generated_answer]
        reponse_generee_add =[agent_Master_add.output_sentences_lang.index2word[ind.item()] for ind in generated_answer_add]

    #reponse = ReplaceMotVar(motsVar, reponse_generee)
    reponse_add = ReplaceMotVar(motsVar, reponse_generee_add)
    #answer_text = ' '.join(reponse) # 'rec : ' + ' '.join(reponse)
    answer_text_add = ' '.join(reponse_add) # 'add : ' + ' '.join(reponse_add)
    return answer_text_add # answer_text + '\n \n' + answer_text_add
    


#######################################################
# --------------------- 7 ----------------------------#
#######################################################


# ****************** parse URL ***********************
def get_url(url):
    response = requests.get(url)
    content = response.content.decode("utf8")
    return content


def get_json_from_url(url):
    content = get_url(url)
    js = json.loads(content) #json -> dictionnaire 
    return js


def get_updates(offset=None):
    url = URL + "getUpdates?timeout={}".format(timeout)
    if offset:
        url += "&offset={}".format(offset)
    js = get_json_from_url(url)
    return js

def get_last_update_id(updates):
    update_ids = []
    for update in updates["result"]:
        update_ids.append(int(update["update_id"]))
    return max(update_ids)


def get_last_chat_id_and_text(updates):
    num_updates = len(updates["result"])
    last_update = num_updates - 1
    text = updates["result"][last_update]["message"]["text"]
    chat_id = updates["result"][last_update]["message"]["chat"]["id"]
    return (text, chat_id)


#************ Answer ****************
def echo_all(updates):
    for update in updates["result"]:
        try:
            text = update["message"]["text"]
            chat = update["message"]["chat"]["id"]
            print('User : {}'.format(text))
            if text != '/start' :
                answer_text = answer(text)
                print('Bot  : {}'.format(answer_text))
                send_message(answer_text, chat) #text
        except Exception as e:
            print(e)

def send_message(text, chat_id):
    text = urllib.parse.quote_plus(text) # + '\n timeout = ' + str(timeout)
    url = URL + "sendMessage?text={}&chat_id={}".format(text, chat_id)
    get_url(url)
    

#text, chat = get_last_chat_id_and_text(get_updates())
#send_message(text, chat)

def main():
    last_update_id = None
    while True:
        updates = get_updates(last_update_id)
        if len(updates["result"]) > 0:
            print('message reçu')
            last_update_id = get_last_update_id(updates) + 1
            echo_all(updates)
        time.sleep(1)


if __name__ == '__main__':
    main()

os.system("pause")
