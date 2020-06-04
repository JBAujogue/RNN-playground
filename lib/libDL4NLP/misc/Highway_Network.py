
import math
import time
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable



class Highway(nn.Module):
    def __init__(self, dim, 
                 dropout = 0, 
                 act = F.tanh):
        super().__init__()
        
        # relevant quantities
        self.dim = dim
        
        # layers
        self.transf  = nn.Linear(dim, dim)
        self.gate    = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(p = dropout)
        self.act     = act

    def forward(self, vect):
        transf = self.act(self.transf(vect))
        gate   = F.sigmoid(self.gate(vect))
        vect   = gate * transf + (1 - gate) * vect
        vect   = self.dropout(vect)
        return vect



class HighwayQ(nn.Module):
    def __init__(self, dim, 
                 query_dim = 0, 
                 dropout = 0,
                 act = F.tanh):
        super().__init__()
        
        # relevant quantities
        self.dim      = dim + query_dim
        self.transf   = nn.Linear(self.dim, dim)
        self.gate     = nn.Linear(self.dim, dim)
        self.dropout  = nn.Dropout(p = dropout)
        self.act      = act

    def forward(self, vect, 
                query = None):
        '''vect and (optional) query must be 3D tensors with same size along dim 0 and 1'''
        if query is not None : merge = torch.cat((vect, query), dim = 2)
        else                 : merge = vect
        transf = self.act(self.transf(merge))
        gate   = F.sigmoid(self.gate(merge))
        vect   = gate * transf + (1 - gate) * vect
        vect   = self.dropout(vect)
        return vect
