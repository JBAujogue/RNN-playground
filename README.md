# Deep Learning for NLP
A series of Jupyter Notebooks describing Pytorch implementations of models solving a large set of NLP tasks.



# Part I 

## I - 1 Word Embedding 

This notebook presents a Pytorch Word2Vec model, trainable following either a CBOW or a Skip-Gram objective, along with a demonstration of Gensim's Word2Vec and FastText models

## I - 2 Sentence Classification

This notebook presents a Sentence Classification model, with word embedding performed by either pretrained custom, Gensim or FastText Word2Vec models. Contextualization is done by multiple stacked GRUs, and important parts of the sentence are identified through self-attention. 

| Tasks |
|-----|
|Binary and Multi-Class Classification|

Implemented Features :

 - Minibatch training
 - Bidirectional GRUs
 - Self-Attention & Multi-Head Self-Attention
 - Highway Connections in (MH) Self-Attention
 - Penalization over distinct heads
 - Collaborative vs. Competitive head behavior
 - 2 methods for Attention Visualization


## I - 3 Language Modeling

This notebook presents a Language Model on top of pretrained custom, Gensim and FastText Word2Vec models.
Tasks :

- **Next Word Prediction**

Implemented Features :

 - Minibatch training
 - Unidirectionnal GRUs

## I - 4 Sequence labelling

Tasks :

- **Masked Language Modeling**
- **Part Of Speech Tagging**

Implemented Features :
   
 - POS Tagging
 - Masked Language Modeling
 - Minibatch training
 - Bidirectionnal GRUs
 - Highway Connections
 
 
 # Part II 

## II - 1 Text Classification

## II - 2 Sequence to sequence

Tasks :

- **Machine Translation**
