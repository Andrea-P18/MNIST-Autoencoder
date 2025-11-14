import torch
import torchvision
import torch.nn as nn
class conv_AE(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2,padding=0),
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2,padding=0),
            nn.Conv2d(in_channels=16,out_channels=8,kernel_size=1,stride=1,padding=0),
            nn.ReLU()
        )
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(8,16,3,2,1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,3,2,1,output_padding=1),
            nn.Sigmoid()
        )
    def encoder(self,x):
        return self.encode(x)
    def decoder(self,x):
        return self.decode(x)
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x