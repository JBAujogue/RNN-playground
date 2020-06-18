# Deep Learning for NLP
A series of Jupyter Notebooks describing Pytorch implementations of models solving a large set of NLP tasks.



# Part I 

## I - 1 Word Embedding 

This notebook presents a Pytorch Word2Vec model, trainable following either a CBOW or a Skip-Gram objective, along with a demonstration of Gensim's Word2Vec and FastText models

## I - 2 Sentence Classification

This notebook presents a Sentence Classification model, with word embedding performed by either pretrained custom, Gensim or FastText Word2Vec models. Contextualization is done by multiple stacked GRUs, and important parts of the sentence are identified through self-attention. Experimentations include :
 
 - Binary and Multi-Class Classification
 - Minibatch-enabled training
 - Bidirectional GRUs & Self-Attention
 - Highway Connections in Self-Attention
 - Multi-Head Self-Attention
 - Penalization over distinct heads
 - Collaborative vs. Competitive head behavior
 - 2 methods for Attention Visualization


## I - 3 Language Modeling

This notebook presents a Language Model on top of pretrained custom, Gensim and FastText Word2Vec models.

 - Next Word Prediction
 - Minibatch-enabled training
 - Unidirectionnal GRUs

## I - 4 Sequence labelling

Experimentations include :
   
 - POS Tagging
 - Token Auto-encoding following Cloze Task
 - Minibatch-enabled training
 - Bidirectionnal GRUs
 - Highway Connections
 
 
 # Part II 

## II - 1 Text Classification

## II - 2 Sequence to sequence

\t a - Machine Translation
