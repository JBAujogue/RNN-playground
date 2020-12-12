# _Deep Learning for NLP_
A series of Jupyter Notebooks describing Pytorch implementations of models solving a large set of NLP tasks.
_This word is still under development._


## I - 1 [Word Embedding](https://github.com/JBAujogue/Deep-Learning-for-NLP/tree/master/DL4NLP%20notebooks)

 Tasks :

- **Word Embedding**

This notebook presents a Pytorch Word2Vec model, trainable following either a CBOW or a Skip-Gram objective, along with a demonstration of Gensim's Word2Vec and FastText models

## I - 2 [Sentence Classification](https://github.com/JBAujogue/Deep-Learning-for-NLP/tree/master/DL4NLP%20notebooks)

 Tasks :

- **Binary and Multi-Class Sequence Classification**

This notebook presents a Sentence Classification model, with word embedding performed by either pretrained custom, Gensim or FastText Word2Vec models. Contextualization is done by multiple stacked GRUs, and important parts of the sentence are identified through Multi-head self-attention.

| Features |
|-----|
| Minibatch training |
| Bidirectional GRUs |
| Self-Attention & Multi-Head Self-Attention |
| Highway Connections in Self-Attention |
| Penalization over distinct heads |
| Collaborative vs. Competitive head behavior |
| 2 methods for Attention Visualization |


## I - 3 [Language Modeling](https://github.com/JBAujogue/Deep-Learning-for-NLP/tree/master/DL4NLP%20notebooks)

Tasks :

- **Next Word Prediction**

This notebook presents a Language Model on top of pretrained custom, Gensim and FastText Word2Vec models.

| Features |
|-----|
| Minibatch training |
| Unidirectionnal GRUs |

## I - 4 [Sequence labelling](https://github.com/JBAujogue/Deep-Learning-for-NLP/tree/master/DL4NLP%20notebooks)

Tasks :

- **Masked Language Modeling**
- **Part Of Speech Tagging**


| Features |
|-----|
| Minibatch training |
| Bidirectionnal GRUs |
| Highway Connections |
 


## II - 1 [Text Classification](https://github.com/JBAujogue/Deep-Learning-for-NLP/tree/master/DL4NLP%20notebooks)

## II - 2 [Sequence to sequence](https://github.com/JBAujogue/Deep-Learning-for-NLP/tree/master/DL4NLP%20notebooks)

Tasks :

- **Machine Translation**
- **Open Domain Chatbot**

| Features |
|-----|
| Minibatch training |
| Bidirectional GRUs |
| Attention Mechanism |
| Highway Connections in Attention |
| Decoder with Smoothed Past Token Feeding |
| Decoder with Content-based Past Attention Feeding |
| Decoder with Position-based Past Attention Feeding |
| 2 methods for Attention Visualization |
