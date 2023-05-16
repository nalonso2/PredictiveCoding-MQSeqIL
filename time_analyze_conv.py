import IL_Conv
import torch
import torchvision
import torch.optim as optim
import utilities
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
import numpy as np
import copy
import math
import torch.nn.functional as F
import time as timer


relu = torch.nn.ReLU()

# Load MNIST Data
def get_data(batch_size=64, data=0):
    if data == 3:
        d_name = 'SVHN'
        num_train = 50000

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        trainset = torchvision.datasets.SVHN(root='./data', download=False, split='train', transform=transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

        testset = torchvision.datasets.SVHN(root='./data', download=False, split='test', transform=transform)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=5000, shuffle=False)


    elif data == 2:
        d_name = 'CIFAR'
        num_train = 50000

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=False, transform=transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   shuffle=True)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=False, transform=transform)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=5000,
                                                  shuffle=False)

    else:
        d_name = 'CIFAR (normalize)'
        num_train = 50000

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=False, transform=transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   shuffle=True)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=False, transform=transform)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=5000,
                                                  shuffle=False)


    return train_loader, test_loader, d_name, num_train



def compute_means(data):
    with torch.no_grad():
        d_tensor = torch.tensor(data[0]).view(1, -1)
        for m in range(1, len(data)):
            d_tensor = torch.cat((d_tensor, torch.tensor(data[m]).view(1, -1)), dim=0)
        return torch.mean(d_tensor, dim=0)



def train_model(train_loader, model, epochs, dev):
    for ep in range(epochs):
        for batch_idx, (images, y) in enumerate(train_loader):
            images = images.to(dev)
            y = y.to(dev)
            target = F.one_hot(y, num_classes=10)
            _, _ = model.train_wts(images.detach(), target.detach(), y)




def train(model, batch_size, data, dev, epochs):
    train_loader, _, d_name, num_train = get_data(batch_size, data=data)

    start = timer.perf_counter()
    train_model(train_loader, model, epochs, dev)
    end = timer.perf_counter()
    return end - start



def training_run(epochs=20, batch_size=64, data=2, dev='cuda', model_type=0, n_hlayers=3):

    # Create Model
    model_dim = [3072]
    for ln in range(n_hlayers):
        model_dim.append(1024)
    model_dim.append(10)

    # BP-SGD
    if model_type == 0:
        model = IL_Conv.IL(type=0, lr=.014)

    # BP-Adam
    elif model_type == 1:
        model = IL_Conv.IL(type=1, lr=.0012)

    # IL
    elif model_type == 2:
        model = IL_Conv.IL(n_iter=3, gamma=.04, alpha=1.5, type=2, beta=50)

    # IL-MQ
    elif model_type == 3:
        model = IL_Conv.IL(n_iter=3, gamma=.04, alpha=.00022, type=3, beta=100, lr_min=.001, r=.0000025)

    # IL-Adam
    elif model_type == 4:
        model = IL_Conv.IL(n_iter=3, gamma=.04, alpha=.000075, type=4, beta=100)

    # IL long
    elif model_type == 5:
        model = IL_Conv.IL(n_iter=15, gamma=.04, alpha=1.5, type=2, beta=50)

    # IL-MQ long
    elif model_type == 6:
        model = IL_Conv.IL(n_iter=15, gamma=.04, alpha=.00022, type=3, beta=100, lr_min=.001, r=.0000025)

    # IL-Adam long
    elif model_type == 7:
        model = IL_Conv.IL(n_iter=15, gamma=.04, alpha=.000075, type=4, beta=100)

    # To Device
    model.to(dev)

    #################################################
    # Train
    time = train(model, batch_size, data, dev, epochs)
    print(f'Convolutional, Model Type:{model_type}  Time:{time}')

    with open(f'data/TimeConv_Type{model_type}_data{data}_epochs{epochs}.data','wb') as filehandle:
        pickle.dump(time, filehandle)