import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

transf = transforms.Compose([
    transforms.ToTensor()
])

mnist_train = datasets.MNIST(root="",train=False,download=True,transform=transf)

model = torch.load("autoencoder.pth",weights_only=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

model.eval()

num_images = 10 
imgs = [mnist_train[i][0] for i in range(10)]

fig, axes = plt.subplots(2, num_images, figsize=(num_images*2, 4))

with torch.no_grad():
    for i, img in enumerate(imgs):
        img = img.unsqueeze(0).to(device)  # batch dimension
        encoded = model.encoder(img)
        decoded = model.decoder(encoded)

        axes[0, i].imshow(img[0,0].cpu().numpy(), cmap='gray')
        axes[0, i].axis('off')

        axes[1, i].imshow(decoded[0,0].cpu().numpy(), cmap='gray')
        axes[1, i].axis('off')

axes[0,0].set_ylabel('Originali', fontsize=12)
axes[1,0].set_ylabel('Ricostruite', fontsize=12)

plt.tight_layout()
plt.savefig("mnist_10_images.png")
plt.close()
