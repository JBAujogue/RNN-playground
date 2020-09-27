
# ------------------------ Libraries import --------------------------
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
import pickle

import requests
import urllib

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import unidecode
from unidecode import unidecode

use_cuda = False

import warnings
warnings.filterwarnings("ignore")




TOKEN = "654278770:AAE4DqqsWp-dtS5pOxul82FCm484PP4vsw0"
URL = "https://api.telegram.org/bot{}/".format(TOKEN)
timeout = 100



# ------------------------ Load Chatbot Model ----------------------
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
motsVar = motVar('C:\\Users\Jb\Desktop\Scripts\data\Conversations_M2DS\\chatbot-M2-DS-Variables.txt')


SOS_token = 0
EOS_token = 1
UNK_token = 2
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS" : 0, "EOS" : 1, "UNK" : 2}
        self.word2count = {"SOS" : 0, "EOS" : 0, "UNK" : 0}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 3  # Counts SOS and EOS and UNK

        
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
        for word in sentence:
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
def importLang(name, n_words):
    lang = Lang(name)
    lang.n_words = n_words
    fil = open(r'C:\Users\Jb\Desktop\Scripts\saves\\'+name+'.file', 'rb')
    return pickle.load(fil)
lang = importLang('lang_M2DS', 1062)

    
from libNLP.models import Chatbot, CreateBot
chatbot = CreateBot(lang,
                    embedding_dim = 150,
                    hidden_dim = 100,
                    pre_entrainement = None,
                    freeze = False,

                    sentence_hidden_dim = 100,
                    hops = 3,
                    share = True,
                    transf = None,
                    dropout = 0.2
                    )
print(chatbot.nbParametres())
chatbot.load_state_dict(torch.load('C:\\Users\Jb\Desktop\Scripts\saves\chatbot.pth'))



def ReplaceMotVar(motsVar, raw_sentence):
        sentence = []
        word_list = raw_sentence.split(' ')
        for word in word_list :
            if word in motsVar.keys() :
                sentence.append(motsVar[word])
            else :
                sentence.append(word)
        return ' '.join(sentence)
def answer(question) :
    reponse, decoder_outputs, attn1_weights, attn2_weights = chatbot(question)
    reponse = ReplaceMotVar(motsVar, reponse)
    return reponse
    


# ----------------------- Main Back Functions --------------------------


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
