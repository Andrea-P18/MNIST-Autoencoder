from torch import nn
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from conv_ed import conv_AE

transf = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(10)
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 16

mnist_train = datasets.MNIST(root="",train=True,download=True,transform=transf)
mnist_test = datasets.MNIST(root="",train=False,download = True, transform = transforms.ToTensor())

train_dataloader = DataLoader(mnist_train,batch_size=batch_size,shuffle=True, num_workers=3,pin_memory=True)
test_dataloader = DataLoader(mnist_test,batch_size=batch_size,shuffle = False, num_workers=3,pin_memory=True)

epochs = 6
lr = 0.1


model = conv_AE()
model = model.to(device)
model.compile()
optimizer = torch.optim.SGD(model.parameters(),lr=lr)
criterion = nn.MSELoss()

for e in range(epochs):

    loss_train = 0.0

    for(im,_) in train_dataloader:

        im = im.to(device)

        optimizer.zero_grad()

        out = model(im)
        
        loss = criterion(out,im)
        loss_train+=loss.item()
        loss.backward()
        optimizer.step()
    
    loss_test = 0.0
    with torch.no_grad():   
        for ( im, _ ) in test_dataloader:

            im = im.to(device)

            out = model(im)

            loss = criterion(out,im)
            loss_test += loss.item()

    loss_train /= len(train_dataloader)
    loss_test /= len(test_dataloader)
    print(f'loss train: {loss_train},loss test: {loss_test}')

torch.save(model,'autoencoder.pth')   


    

