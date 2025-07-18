import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoImageProcessor, AutoModel


class DINOStyle(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DINOStyle, self).__init__()

        # carico il modello
        self.model = AutoModel.from_pretrained('facebook/dinov2-small') # carica il modello DINOv2 versione small

        # freeze di dino
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()

        self.num_classes = num_classes
        self.in_channels = in_channels

        self.linear = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.model(x) # capire che ritorna il modello DINOv2
        x = x.last_hidden_state

        # prendo solo le patch
        # x = x[:, 1:, :]

        # prendo il token CLS
        x = x[:, 0, :]
        x = x.unsqueeze(1)

        x = x.mean(dim=1)
        x = self.linear(x)
        return x
    