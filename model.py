import torch
import torch.nn as nn
import torchvision
import numpy as np

class mn_model(nn.Module):
    
    def __init__(self, first_layer_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(first_layer_size,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.Linear(64,32),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.Linear(256,first_layer_size),
            nn.Sigmoid()
        )

    def encode(self,x):

        x = self.encoder(x)
        return x
    
    def decode(self,x):
        x = self.decoder(x)
        return x

    def forward(self,x):

        x = self.encode(x)
        x = self.decode(x)
        return x
    