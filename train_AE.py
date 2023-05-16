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

bce = torch.nn.BCELoss(reduction='none')
mse = torch.nn.MSELoss(reduction='none')
softmax = torch.nn.Softmax(dim=1)
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
relu = torch.nn.ReLU()


# Load Data
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



def test(test_losses, model, test_loader, seed, lr, dev):
    with torch.no_grad():
        test_losses[lr][seed].append(0)
        testn = 0
        for batch_idx, (images, y) in enumerate(test_loader):
            images = images.view(images.size(0), -1).to(dev) * .99 + .005

            # Test and record losses over test set
            h = model.initialize_values(images)

            ##Check if output is nan. This conditional is needed to prevent program from aborting when Nan value produced
            if torch.isnan(torch.sigmoid(h[-1])).sum().item() > 0:
                print('Nan')
                test_losses[lr][seed][-1] = float("nan")
            else:
                global_loss = torch.mean(bce(torch.sigmoid(h[-1]), images.detach()), dim=1).sum()
                test_losses[lr][seed][-1] += global_loss.item()

            testn += images.size(0)

        test_losses[lr][seed][-1] /= testn




def train_model(train_loader, test_loader, model, seed, lr, test_losses, epochs, dev, b_size):
    test(test_losses, model, test_loader, seed, lr, dev)

    for ep in range(epochs):
        for batch_idx, (images, y) in enumerate(train_loader):
            images = images.view(y.size(0), -1).to(dev) * .99 + .005

            if images.size(0) == b_size:
                _, _ = model.train_wts(images.detach(), images.detach(), y.detach())

        # Test every epoch
        test(test_losses, model, test_loader, seed, lr, dev)
        #print(ep, test_losses[lr][seed][-1])





def train(models, batch_size, data, dev, epochs, test_losses):

    for l in range(len(models)):
        print(f'Alpha: {models[l][0].alpha}')
        for m in range(len(models[0])):
            train_loader, test_loader, d_name, num_train = get_data(batch_size, data=data)

            train_model(train_loader, test_loader, models[l][m], m, l, test_losses, epochs, dev, batch_size)

            print(f'Seed:{m}', f'MinLoss:{min(test_losses[l][m])}',
                  f'MinEarLoss:{min(test_losses[l][m][0:10])}',
                  f'LastLoss:{test_losses[l][m][-1]}')





def training_run(epochs=100, batch_size=64, data=0, num_seeds=1, alpha=[.015], model_type=0, beta=100, gamma=.035):

    # Create Models
    model_dim = [3072, 1024, 256, 20, 256, 1024, 3072]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = []

    for l in range(len(alpha)):
        #Add list of seeds at this learning rate
        models.append([])
        for m in range(num_seeds):
            #BP-SGD
            if model_type == 0:
                models[-1].append(BP.BP(model_dim, type=0, alpha=alpha[l], smax=False))

            #BP-Adam
            elif model_type == 1:
                models[-1].append(BP.BP(model_dim, type=0, alpha=alpha[l], smax=False))

            #SeqIL
            elif model_type == 2:
                models[-1].append(IL.IL(model_dim, smax=False, n_iter=6, gamma=gamma, beta=beta, type=0, alpha=alpha[l]))

            #SeqIL-MQ
            elif model_type == 3:
                models[-1].append(IL.IL(model_dim, smax=False, n_iter=6, gamma=gamma, beta=beta, type=1, alpha=alpha[l]))

            #SeqIL-Adam
            elif model_type == 4:
                models[-1].append(IL.IL(model_dim, smax=False, n_iter=6, gamma=gamma, beta=beta, type=2, alpha=alpha[l]))


        # To Device
        for i in range(len(models[-1])):
            models[-1][i].to(dev)

    #################################################
    # Create Containers
    test_losses = [[[] for m in range(num_seeds)] for m in range(len(models))]  # [model_lr][model_seed]

    #################################################

    # Train
    print(f'MODEL TYPE: {model_type}  Gamma:{gamma}')
    train(models, batch_size, data, dev, epochs, test_losses)


    # Store Data
    best_test_loss = torch.mean(torch.tensor([min(test_losses[0][x]) for x in range(len(test_losses[0]))])).item()
    best_lr = 0
    for l in range(1, len(models)):
        ls = torch.mean(torch.tensor([min(test_losses[l][x]) for x in range(len(test_losses[0]))])).item()
        if best_test_loss > ls:
            best_test_loss = ls
            best_lr = l

    print(f'Best Learning Rate, Model Type{model_type}:', best_lr)
    with open(f'data/AE_Type{model_type}_data{data}_epochs{epochs}.data', 'wb') as filehandle:
        pickle.dump([test_losses[best_lr], alpha[best_lr]], filehandle)

