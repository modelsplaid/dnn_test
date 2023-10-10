#Import the packages needed.
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from torchvision.transforms import ToTensor

data_train = torchvision.datasets.MNIST('./data',
        download=True,train=True,transform=ToTensor())
data_test = torchvision.datasets.MNIST('./data',
        download=True,train=False,transform=ToTensor())

fig,ax = plt.subplots(1,7)

print(type(data_train))
print(type(data_train[0][0]))
print(type(data_train[0][0].view(28,28)))
print(data_train[0][0].view(28,28))
for i in range(7):
    ax[i].imshow(data_train[i][0].view(28,28))
    ax[i].set_title(data_train[i][1])
    #ax[i].axis('off')
fig.show()
#input("in: ")


