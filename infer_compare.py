import IL_Stnd
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
import torch.nn.functional as F

bce = torch.nn.BCELoss(reduction='none')
mse = torch.nn.MSELoss(reduction='none')
softmax = torch.nn.Softmax(dim=1)
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
relu = torch.nn.ReLU()

# Load MNIST Data
def get_data(batch_size=64, data=0):

    if data == 0:
        d_name = 'SVHN'
        num_train = 50000

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        trainset = torchvision.datasets.SVHN(root='./data', download=True, split='train', transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

        testset = torchvision.datasets.SVHN(root='./data', download=True, split='test', transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=5000, shuffle=False)

    else:
        d_name = 'CIFAR'
        num_train = 50000

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=5000, shuffle=False)

    return train_loader, test_loader, d_name, num_train



def compute_means(data):
    with torch.no_grad():
        d_tensor = torch.tensor(data[0]).view(1, -1)
        for m in range(1, len(data)):
            d_tensor = torch.cat((d_tensor, torch.tensor(data[m]).view(1, -1)), dim=0)
        return torch.mean(d_tensor, dim=0)


def test(test_losses, test_accuracies, model, test_loader, seed, lr, g, dev):
    with torch.no_grad():
        test_accuracies[lr][g][seed].append(0)
        test_losses[lr][g][seed].append(0)
        testn = 0
        for batch_idx, (images, y) in enumerate(test_loader):
            images = images.view(y.size(0), -1).to(dev)
            y = y.to(dev)
            target = F.one_hot(y, num_classes=10).to(dev)

            # Test and record losses and accuracy over whole test set
            h = model.initialize_values(images)
            global_loss = torch.mean(mse(softmax(h[-1]), target).sum(1))
            test_accuracies[lr][g][seed][-1] += utilities.compute_num_correct(softmax(h[-1]), y)
            test_losses[lr][g][seed][-1] += global_loss.item()
            testn += images.size(0)

        test_accuracies[lr][g][seed][-1] /= testn
        test_losses[lr][g][seed][-1] /= testn



def train_model(train_loader, test_loader, model, seed, lr, gamma, test_losses, test_accuracies, epochs, dev):
    #Initial test
    test(test_losses, test_accuracies, model, test_loader, seed, lr, gamma, dev)

    #Train and test each epoch
    for ep in range(epochs):
        for batch_idx, (images, y) in enumerate(train_loader):
            images = images.view(y.size(0), -1).to(dev)
            y = y.to(dev)
            target = F.one_hot(y, num_classes=10)
            _, _ = model.train_wts(images.detach(), target.detach(), y)

        test(test_losses, test_accuracies, model, test_loader, seed, lr, gamma, dev)



def train(models, batch_size, data, dev, epochs, test_losses, test_accuracies):

    for a in range(len(models)):
        for g in range(len(models[a])):
            print(f'Training Alpha:{models[a][g][0].alpha}   Gamma:{models[a][g][0].gamma}')
            for m in range(len(models[a][g])):
                train_loader, test_loader, d_name, num_train = get_data(batch_size, data=data)
                train_model(train_loader, test_loader, models[a][g][m], m, a, g, test_losses, test_accuracies, epochs, dev)

                print(f'Seed:{m}', f'MaxAcc:{max(test_accuracies[a][g][m])}  LastAcc:{test_accuracies[a][g][m][-1]}')




def training_run(epochs=50, batch_size=64, data=2, num_seeds=1, alpha=[.0001], model_type=0, n_iter=5, beta=100,
                 n_hlayers=4, gamma=[.05]):

    # Create Models
    model_dim = [3072]
    for ln in range(n_hlayers):
        model_dim.append(1024)
    model_dim.append(10)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = []

    #Add list of seeds at this learning rate
    for a in range(len(alpha)):
        #Add list models at this learning rate
        models.append([])
        for g in range(len(gamma)):
            # Add list of seeds at this learning rate and gamma value
            models[-1].append([])
            for m in range(num_seeds):
                # IL-sequential
                if model_type == 0:
                    models[-1][-1].append(IL.IL(model_dim, smax=True, n_iter=n_iter, gamma=gamma[g], beta=beta, type=2, alpha=alpha[a]))

                # IL-simultaneous
                elif model_type == 1:
                    models[-1][-1].append(IL_Stnd.IL(model_dim, smax=True, n_iter=n_iter, gamma=gamma[g],  beta=beta, type=2, alpha=alpha[a]))

                models[-1][-1][-1].to(dev)

    #################################################
    # Create Containers
    test_losses = [[[[] for s in range(num_seeds)] for g in range(len(gamma))] for a in range(len(alpha))]  # [model_lr][model_seed]
    test_accs = [[[[] for s in range(num_seeds)] for g in range(len(gamma))] for a in range(len(alpha))]  # [model_lr][model_seed]

    #################################################
    # Train
    train(models, batch_size, data, dev, epochs, test_losses, test_accs)

    # Store Data
    best_test_acc = torch.mean(torch.tensor([max(test_accs[0][0][x]) for x in range(len(test_accs[0][0]))])).item()
    best_lr = 0
    best_g = 0
    for a in range(len(models)):
        for g in range(len(models[a])):
            ac = torch.mean(torch.tensor([max(test_accs[a][g][x]) for x in range(len(test_accs[a][g]))])).item()
            if best_test_acc < ac:
                best_test_acc = ac
                best_lr = a
                best_g = g

    # Store Data
    print(f'Best LR: {alpha[best_lr]}       Best Gamma: {gamma[best_g]}')
    with open(f'data/Infer_compare_type{model_type}_data{data}_{n_iter}.data','wb') as filehandle:
        pickle.dump([test_accs[best_lr][best_g], test_losses[best_lr][best_g], alpha[best_lr], gamma[best_g]], filehandle)