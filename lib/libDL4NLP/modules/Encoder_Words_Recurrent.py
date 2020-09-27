
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class RecurrentEncoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, 
                 n_layer = 1, 
                 dropout = 0, 
                 bidirectional = False): 
        super().__init__()
        
        # relevant quantities
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.out_dim = hid_dim # * (2 if bidirectional else 1)
        self.n_layer = n_layer
        self.bidirectional = bidirectional

        # layers
        self.dropout = nn.Dropout(p = dropout)
        self.bigru = nn.GRU(
            emb_dim, 
            hid_dim, 
            n_layer,
            dropout = (0 if n_layer == 1 else dropout), 
            bidirectional = bidirectional,
            batch_first = False)

        
    def forward(self, embeddings, 
                lengths = None, 
                hidden = None, 
                enforce_sorted = True) :
        # embedding
        embeddings = self.dropout(embeddings)    # size (batch_size, input_length, emb_dim)
        embeddings = embeddings.transpose(0, 1)  # size (input_length, batch_size, emb_dim)

        # packing
        if lengths is not None : 
            embeddings = nn.utils.rnn.pack_padded_sequence(
                embeddings, 
                lengths, 
                batch_first = False, 
                enforce_sorted = enforce_sorted)
            
        # GRU pass
        outputs, hidden = self.bigru(embeddings, hidden)
        
        # padding
        if lengths is not None : 
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first = False)
        
        # Sum bidirectional GRU outputs
        if self.bidirectional : outputs = outputs[:, :, :self.hid_dim] + outputs[:, : ,self.hid_dim:]
        
        # batch first
        outputs = outputs.transpose(0, 1) # size (batch_size, input_length, out_dim)
        
        # dropout
        outputs = self.dropout(outputs) # size (batch_size, input_length, out_dim)
        hidden  = self.dropout(hidden)  # size (n_layer * num_dir, batch_size, hid_dim)
        return outputs, hidden
    
    
    
# -- OLD --
class RecurrentWordsEncoder(nn.Module):
    def __init__(self, 
                 device, 
                 embedding, 
                 hidden_dim, 
                 n_layers = 1, 
                 dropout = 0
                ): 
        super(RecurrentWordsEncoder, self).__init__()
        
        # relevant quantities
        self.device = device
        self.hidden_dim = hidden_dim           # dimension of hidden state of GRUs 
        self.dropout_p = dropout
        self.n_layers = n_layers               # number of stacked GRU layers
        self.output_dim = hidden_dim * 2       # dimension of outputed rep. of words and utterance
        
        # parameters
        self.embedding = embedding
        for p in embedding.parameters() :
            embedding_dim = p.data.size(1)
        self.dropout = nn.Dropout(p = dropout)
        self.bigru = nn.GRU(embedding_dim, 
                            hidden_dim, 
                            n_layers,
                            dropout=(0 if n_layers == 1 else dropout), 
                            bidirectional=True)

        
    def initHidden(self): 
        return Variable(torch.zeros(2 * self.n_layers, 1, self.hidden_dim)).to(self.device)

    def forward(self, utterance, hidden = None):
        if hidden is None : hidden = self.initHidden()
        embeddings = self.embedding(utterance)                          # dim = (input_length, 1, embedding_dim)
        embeddings = self.dropout(embeddings)                           # dim = (input_length, 1, embedding_dim)
        outputs, hidden = self.bigru(embeddings, hidden)
        outputs = self.dropout(outputs)
        hidden = self.dropout(hidden)
        return outputs, hidden                                          # dim = (input_length, 1, hidden_dim * 2)
