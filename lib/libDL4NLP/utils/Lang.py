

import nltk


class Lang:
    def __init__(self, corpus = None, base_tokens = ['UNK'], min_count = None):
        self.base_tokens = base_tokens
        self.initData(base_tokens)
        if    corpus is not None : self.addCorpus(corpus)
        if min_count is not None : self.removeRareWords(min_count)

        
    def initData(self, base_tokens) :
        self.word2index = {word : i for i, word in enumerate(base_tokens)}
        self.index2word = {i : word for i, word in enumerate(base_tokens)}
        self.word2count = {word : 0 for word in base_tokens}
        self.n_words = len(base_tokens)
        return
    
    def getIndex(self, word) :
        if    word in self.word2index : return self.word2index[word]
        elif 'UNK' in self.word2index : 
            #print('word ' + word + ' is unknown')
            return self.word2index['UNK']
        return
        
    def addWord(self, word):
        '''Add a word to the language'''
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
        return 
            
    def addSentence(self, sentence):
        '''Add to the language all words of a sentence'''
        words = sentence if type(sentence) == list else nltk.word_tokenize(sentence)
        for word in words : self.addWord(word)          
        return
            
    def addCorpus(self, corpus):
        '''Add to the language all words contained into a corpus'''
        for text in corpus : self.addSentence(text)
        return 
                
    def removeRareWords(self, min_count):
        '''remove words appearing lesser than a min_count threshold'''
        kept_word2count = {word: count for word, count in self.word2count.items() if count >= min_count}
        self.initData(self.base_tokens)
        for word, count in kept_word2count.items(): 
            self.addWord(word)
            self.word2count[word] = kept_word2count[word]
        return
