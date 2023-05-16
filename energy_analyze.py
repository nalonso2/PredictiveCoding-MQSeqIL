import BP
import BayesIL
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
    if data == 3:
        d_name = 'SVHN'
        num_train = 50000

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        trainset = torchvision.datasets.SVHN(root='./data', download=True, split='train', transform=transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   shuffle=True)

        testset = torchvision.datasets.SVHN(root='./data', download=True, split='test', transform=transform)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False)


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

        test_loader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                                  shuffle=False)



    elif data == 1:
        num_train = 60000
        d_name = 'fashion'

        train_loader = DataLoader(
            torchvision.datasets.FashionMNIST('./data', train=True, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                              ])), batch_size=batch_size, shuffle=True, pin_memory=False)

        test_loader = DataLoader(
            torchvision.datasets.FashionMNIST('./data', train=False, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                              ])), batch_size=10000, shuffle=False, pin_memory=False)

    else:
        num_train = 60000
        d_name = ''

        train_loader = DataLoader(
            torchvision.datasets.MNIST('./data', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                       ])), batch_size=batch_size, shuffle=True, pin_memory=False)

        test_loader = DataLoader(
            torchvision.datasets.MNIST('./data', train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                       ])), batch_size=10000, shuffle=False, pin_memory=False)

    return train_loader, test_loader, d_name, num_train



def record_energy(model, test_loader, Engy, Engy_std, seed, dev):
    with torch.no_grad():
        for batch_idx, (images, y) in enumerate(test_loader):
            images = images.view(y.size(0), -1).to(dev)
            y = y.to(dev)
            global_target = F.one_hot(y, num_classes=10).to(dev)

            # Initialize targets to FF values
            h = model.initialize_values(images)
            targ = [h[i].clone() for i in range(model.num_layers)]
            targ[-1] = (1 - model.mod_prob) * global_target.clone() + model.mod_prob * softmax(targ[-1])  # global_target.clone()

            # Update for 150
            for i in range(150):
                for layer in reversed(range(1, model.num_layers - 1)):
                    p = model.wts[layer](targ[layer])

                    # Compute error
                    if layer < model.num_layers - 2:
                        err = targ[layer + 1] - p  # MSE gradient
                    else:
                        err = targ[-1] - softmax(p)  # Cross-ent w/ softmax gradient

                    # Record local mse
                    Engy[seed, i] += torch.mean(torch.square(err).sum(1))
                    Engy_std[seed, i] += torch.std(torch.square(err).sum(1))

                    # Update Targets Hidden
                    decay = (1 / (1 + i))
                    if model.nm_err:
                        n = torch.square(model.wts[layer][1].weight.data).sum(0) + 1
                        dfdt = err.matmul(model.wts[layer][1].weight / n) * model.func_d(targ[layer])
                    else:
                        dfdt = err.matmul(model.wts[layer][1].weight) * model.func_d(targ[layer])

                    e_top = targ[layer] - model.wts[layer - 1](targ[layer - 1])
                    dt = decay * model.gamma * (dfdt - e_top)
                    targ[layer] = targ[layer] + dt

                Engy[seed, i] += torch.mean(torch.square(e_top).sum(1))
                Engy_std[seed, i] += torch.std(torch.square(e_top).sum(1))

                p = softmax(model.wts[-1](targ[-2]))
                targ[-1] = (1 - model.mod_prob) * global_target + model.mod_prob * p


            return


def train_model(train_loader, test_loader, model, Engy, Engy_std, max_iters, dev, seed):
    with torch.no_grad():
        iter = 0
        while iter < max_iters:
            for batch_idx, (images, y) in enumerate(train_loader):
                images = images.view(y.size(0), -1).to(dev)
                y = y.to(dev)
                target = F.one_hot(y, num_classes=10)

                # Store initial weights and update
                _, _ = model.train_wts(images.detach(), target.detach(), y)

                # End if max training iterations reached
                iter += 1
                '''if iter % 1000 == 0:
                    print(iter)'''
                if iter == max_iters:
                    record_energy(model, test_loader, Engy, Engy_std, seed, dev)
                    return


def train(models, batch_size, data, dev, max_iters, Engy, Engy_std):
    with torch.no_grad():
        for s in range(len(models)):
            print(f'Training Seed:{s}')
            train_loader, test_loader, _, _ = get_data(batch_size, data=data)
            train_model(train_loader, test_loader, models[s], Engy, Engy_std, max_iters, dev, s)



def training_run(max_iters=10000, batch_size=64, data=2, num_seeds=2, smax=True, model_type=0, n_hlayers=3, n_iter=5):
    with torch.no_grad():
        # Create Models
        model_dim = [3072]
        for ln in range(n_hlayers):
            model_dim.append(1024)
        model_dim.append(10)

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        models = []
        for m in range(num_seeds):
            # IL
            if model_type == 0:
                models.append(BayesIL.IL(model_dim, crossEnt=True, smax=smax, n_iter=n_iter, gamma=.05, eps=.2,
                                         type=0, alpha=100, lr_decay=False).to(dev))

            # IL with GN update
            elif model_type == 1:
                models.append(BayesIL.IL(model_dim, crossEnt=True, smax=smax, n_iter=n_iter, gamma=.1, eps=.2,
                                         type=0, alpha=100, lr_decay=False, nm_err=True).to(dev))

            # IL with GN update
            elif model_type == 2:
                models.append(BayesIL.IL(model_dim, crossEnt=True, smax=smax, n_iter=n_iter, gamma=.05, eps=.2,
                                         type=0, alpha=100, lr_decay=False, nm_err=True).to(dev))

            # IL with GN update
            elif model_type == 3:
                models.append(BayesIL.IL(model_dim, crossEnt=True, smax=smax, n_iter=n_iter, gamma=.07, eps=.2,
                                         type=0, alpha=100, lr_decay=False, nm_err=True).to(dev))

        #################################################
        # Create Containers
        Engy = torch.zeros(num_seeds, 150).to(dev)
        Engy_std = torch.zeros(num_seeds, 150).to(dev)

        #################################################
        # Train
        print(f'TRAINING MODEL TYPE {model_type}')
        train(models, batch_size, data, dev, max_iters, Engy, Engy_std)

        engy_m = torch.mean(Engy, dim=0).to('cpu')
        engy_std = torch.mean(Engy_std, dim=0).to('cpu')

        with open(f'data/energyAnalyze_type{model_type}_data{data}_max_iter{max_iters}_niter{n_iter}_btchsz{batch_size}.data',
                'wb') as filehandle:
            pickle.dump([engy_m, engy_std], filehandle)


training_run(num_seeds=2)
#training_run(model_type=1, num_seeds=2)
#training_run(model_type=2, num_seeds=7)
#training_run(model_type=3, num_seeds=7)
