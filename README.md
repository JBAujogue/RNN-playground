# _Deep Learning for NLP_
A series of Jupyter Notebooks describing Pytorch implementations of models solving a large set of NLP tasks.



# I - 1 Word Embedding 

 Tasks :

- **Word Embedding**

This notebook presents a Pytorch Word2Vec model, trainable following either a CBOW or a Skip-Gram objective, along with a demonstration of Gensim's Word2Vec and FastText models

# I - 2 Sentence Classification

 Tasks :

- **Binary and Multi-Class Sequence Classification**

This notebook presents a Sentence Classification model, with word embedding performed by either pretrained custom, Gensim or FastText Word2Vec models. Contextualization is done by multiple stacked GRUs, and important parts of the sentence are identified through Multi-head self-attention.

| Implemented Features |
|-----|
| Minibatch training |
| Bidirectional GRUs |
| Self-Attention & Multi-Head Self-Attention |
| Highway Connections in Self-Attention |
| Penalization over distinct heads |
| Collaborative vs. Competitive head behavior |
| 2 methods for Attention Visualization |


# I - 3 Language Modeling

Tasks :

- **Next Word Prediction**

This notebook presents a Language Model on top of pretrained custom, Gensim and FastText Word2Vec models.

| Implemented Features |
|-----|
| Minibatch training |
| Unidirectionnal GRUs |

# I - 4 Sequence labelling

Tasks :

- **Masked Language Modeling**
- **Part Of Speech Tagging**


| Implemented Features |
|-----|
| Minibatch training |
| Bidirectionnal GRUs |
| Highway Connections |
 


# II - 1 Text Classification

# II - 2 Sequence to sequence

Tasks :

- **Machine Translation**
- **Open Domain Chatbot**
