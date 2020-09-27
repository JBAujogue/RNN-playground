
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassDecoder(nn.Module):
    
    def __init__(self, text_dim, n_classes) :
        super(ClassDecoder, self).__init__() 
        self.version = 'class'
        self.n_classes = n_classes
        self.classes_decoder = nn.Linear(text_dim, n_classes)

    def forward(self, text_vector, train_mode = False):
        classes_vector = self.classes_decoder(text_vector).view(-1)
        if train_mode :
            return classes_vector
        else :
            classes = F.softmax(classes_vector) 
            topv, topi = classes.data.topk(1)
            result = topi[0][0].numpy()
            return result   
