# Deep Learning for NLP
A series of Jupyter Notebooks describing Pytorch implementations of models solving a large set of NLP tasks


## Part I 

### I - 1 Word Embedding 

This notebook presents a Pytorch Word2Vec model, trainable following either a CBOW or a Skip-Gram objective, along with a demonstration of Gensim's Word2Vec and FastText models

### I - 2 Sentence Classification

This notebook presents a Sentence Classification model, with word embedding performed by either pretrained custom, Gensim or FastText Word2Vec models. Contextualization is done by multiple stacked GRUs, and important parts of the sentence are identified through single or multi-head self-attention. Applications are :
    
 - Extractive Summarization
 - Sentiment Analysis
 - Text Segmentation


### I - 3 Language Modeling

This notebook presents a Language Model on top of pretrained custom, Gensim and FastText Word2Vec models.


### I - 4 Sequence labelling

Applications are :
    
 - Contextual Word Embedding pre-training
 - Part-of-speech Tagging
 - Named Entity Recognition
 - Automatic Value Extraction
    
