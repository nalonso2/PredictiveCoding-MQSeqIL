import BP
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

        trainset = torchvision.datasets.SVHN(root='./data', download=False, split='train', transform=transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

        testset = torchvision.datasets.SVHN(root='./data', download=False, split='test', transform=transform)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=5000, shuffle=False)

    else:
        d_name = 'CIFAR'
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


    print(d_name)



    return train_loader, test_loader, d_name, num_train


def compute_means(data):
    with torch.no_grad():
        d_tensor = torch.tensor(data[0]).view(1, -1)
        for m in range(1, len(data)):
            d_tensor = torch.cat((d_tensor, torch.tensor(data[m]).view(1, -1)), dim=0)
        return torch.mean(d_tensor, dim=0)


def test(test_losses, test_accuracies, model, test_loader, seed, lr, dev):
    with torch.no_grad():
        test_accuracies[lr][seed].append(0)
        test_losses[lr][seed].append(0)
        testn = 0
        for batch_idx, (images, y) in enumerate(test_loader):
            images = images.to(dev)
            y = y.to(dev)
            target = F.one_hot(y, num_classes=10).to(dev)

            # Test and record losses and accuracy over whole test set
            h = model.initialize_values(images)
            global_loss = torch.mean(mse(softmax(h[-1]), target).sum(1))
            test_accuracies[lr][seed][-1] += utilities.compute_num_correct(softmax(h[-1]), y)
            test_losses[lr][seed][-1] += global_loss.item()
            testn += images.size(0)

        test_accuracies[lr][seed][-1] /= testn
        test_losses[lr][seed][-1] /= testn


def train_model(train_loader, test_loader, model, seed, lr, test_losses, test_accuracies, epochs, dev, b_size):
    test(test_losses, test_accuracies, model, test_loader, seed, lr, dev)

    for ep in range(epochs):
        for batch_idx, (images, y) in enumerate(train_loader):
            images = images.to(dev)
            y = y.to(dev)
            target = F.one_hot(y, num_classes=10)
            if images.size(0) == b_size:
                _, _ = model.train_wts(images.detach(), target.detach(), y)

        test(test_losses, test_accuracies, model, test_loader, seed, lr, dev)
        print(ep+1, 'Acc:', test_accuracies[lr][seed][-1] * 100)





def train(models, batch_size, data, dev, epochs, test_losses, test_accuracies):

    for l in range(len(models)):
        print(f'Training Alpha:{models[l][0].alpha}')
        for m in range(len(models[0])):

            train_loader, test_loader, d_name, num_train = get_data(batch_size, data=data)
            train_model(train_loader, test_loader, models[l][m], m, l, test_losses, test_accuracies, epochs, dev, batch_size)

            print(f'Seed:{m}', f'MaxAcc:{max(test_accuracies[l][m])}',
                  f'LastAcc:{test_accuracies[l][m][-1]}')




def training_run(epochs=50, batch_size=64, data=2, num_seeds=1, alpha=[1], model_type=0, beta=100, gamma=.04):

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = []

    for l in range(len(alpha)):
        #Add list of seeds at this learning rate
        models.append([])
        for m in range(num_seeds):
            # BP-SGD
            if model_type == 0:
                models[-1].append(IL_Conv.IL(type=0, lr=alpha[l]))

            # BP-Adam
            elif model_type == 1:
                models[-1].append(IL_Conv.IL(type=1, lr=alpha[l]))

            # IL n_iter = 7, gamma=.03
            elif model_type == 2:
                models[-1].append(IL_Conv.IL(n_iter=3, gamma=gamma, beta=beta, type=2, alpha=alpha[l]))

            # IL-MQ
            elif model_type == 3:
                models[-1].append(IL_Conv.IL(n_iter=3, gamma=gamma,  beta=beta, type=3, alpha=alpha[l], lr_min=.001))

            # IL-Adam
            elif model_type == 4:
                models[-1].append(IL_Conv.IL(n_iter=3, gamma=gamma, beta=beta, type=4, alpha=alpha[l]))


        # To Device
        for i in range(len(models[-1])):
            models[-1][i].to(dev)

    #################################################
    # Create Containers
    test_losses = [[[] for m in range(num_seeds)] for m in range(len(models))]  # [model_lr][model_seed]
    test_accs = [[[] for m in range(num_seeds)] for m in range(len(models))]  # [model_lr][model_seed]

    #################################################
    # Train
    print(f'\nTRAINING MODEL TYPE {model_type}')
    train(models, batch_size, data, dev, epochs, test_losses, test_accs)

    # Store Data
    best_test_acc = torch.mean(torch.tensor([max(test_accs[0][x]) for x in range(len(test_accs[0]))])).item()
    best_lr = 0
    for l in range(1, len(models)):
        ac = torch.mean(torch.tensor([max(test_accs[l][x]) for x in range(len(test_accs[0]))])).item()
        if best_test_acc < ac:
            best_test_acc = ac
            best_lr = l

    #print(f'Best Learning Rate, at Iterations{max_iters}, Model Type{model_type}:', best_lr)
    '''with open(f'data/Conv_Type{model_type}_data{data}_epochs{epochs}_{batch_size}.data','wb') as filehandle:
        pickle.dump([test_accs[best_lr], test_losses[best_lr], alpha[best_lr]], filehandle)'''