import BP
import IL
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
import time as timer

relu = torch.nn.ReLU()


# Load MNIST Data
def get_data(batch_size=64, data=0):
    if data == 0:
        d_name = 'SVHN'
        num_train = 50000

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        trainset = torchvision.datasets.SVHN(root='./data', download=True, split='train', transform=transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   shuffle=True)

        testset = torchvision.datasets.SVHN(root='./data', download=True, split='test', transform=transform)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=5000, shuffle=False)


    elif data == 1:
        d_name = 'CIFAR'
        num_train = 50000

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=5000, shuffle=False)



    return train_loader, test_loader, d_name, num_train


def train_model(train_loader, model, epochs, dev):
    for ep in range(epochs):
        for batch_idx, (images, y) in enumerate(train_loader):
            images = images.view(y.size(0), -1).to(dev) * .99 + .005
            _, _ = model.train_wts(images.detach(), images.detach(), y.detach())


def train(model, batch_size, data, dev, epochs):
    train_loader, test_loader, d_name, num_train = get_data(batch_size, data=data)

    start = timer.perf_counter()
    train_model(train_loader, model, epochs, dev)
    end = timer.perf_counter()

    return end - start




def training_run(epochs=100, batch_size=64, data=1, model_type=0):

    # Create Models
    model_dim = [3072, 1024, 256, 20, 256, 1024, 3072]

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    #BP-SGD
    if model_type == 0:
        model = BP.BP(model_dim, type=0, alpha=.0013, smax=False)

    #BP-Adam
    elif model_type == 1:
        model = BP.BP(model_dim, type=0, alpha=.0007, smax=False)

    #IL
    elif model_type == 2:
        model = IL.IL(model_dim, smax=False, n_iter=6, gamma=.035, alpha=.1, type=0, beta=10)

    #IL-MQ
    elif model_type == 3:
        model = IL.IL(model_dim, smax=False, n_iter=6, gamma=.035, alpha=.000012, type=2, beta=100, lr_min=.001)

    #IL-Adam
    elif model_type == 4:
        model = IL.IL(model_dim, smax=False, n_iter=6, gamma=.035, alpha=.000025, type=3, beta=100)

    # IL
    elif model_type == 5:
        model = IL.IL(model_dim, smax=False, n_iter=15, gamma=.03, alpha=.075, type=0, beta=10)

    # IL-MQ
    elif model_type == 6:
        model = IL.IL(model_dim, smax=False, n_iter=15, gamma=.03, alpha=.0000135, type=2, beta=100, lr_min=.001)

    # IL-Adam
    elif model_type == 7:
        model = IL.IL(model_dim, smax=False, n_iter=15, gamma=.03, alpha=.000025, type=3, beta=100)


    model.to(dev)
    #################################################
    # Train
    time = train(model, batch_size, data, dev, epochs)
    print(f'Autoencoder, Model Type:{model_type}  Time:{time}')

    #Store Data
    with open(f'data/TimeAE_Type{model_type}_data{data}_epochs{epochs}.data', 'wb') as filehandle:
        pickle.dump(time, filehandle)