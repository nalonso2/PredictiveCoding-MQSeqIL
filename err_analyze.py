import IL
import IL_Stnd
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

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   shuffle=True)

        testset = torchvision.datasets.SVHN(root='./data', download=True, split='test', transform=transform)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=5000, shuffle=False)


    else:
        d_name = 'CIFAR'
        num_train = 50000

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False)

    return train_loader, test_loader, d_name, num_train



def record_infer_seq(model, images, global_target, local_mse, seed):
    with torch.no_grad():
        # Initialize targets to FF values
        h = model.initialize_values(images)
        targ = [h[i].clone() for i in range(model.num_layers)]
        targ[-1] = (1 - model.mod_prob) * global_target.clone() + model.mod_prob * softmax(targ[-1])


        for i in range(local_mse[0].size(1)):
            #decay = (1 / (1 + i))
            decay = 1
            for layer in reversed(range(1, model.num_layers - 1)):
                p = model.wts[layer](targ[layer])

                # Compute error
                if layer < model.num_layers - 2:
                    err = targ[layer + 1] - p  # MSE gradient
                else:
                    err = targ[-1] - softmax(p)  # Cross-ent w/ softmax gradient

                # Record local mse
                local_mse[layer][seed, i] = torch.mean(torch.square(err))

                # Update Targets Hidden
                dfdt = err.matmul(model.wts[layer][1].weight) * model.func_d(targ[layer])

                e_top = targ[layer] - model.wts[layer - 1](targ[layer - 1])
                dt = decay * model.gamma * (dfdt - e_top)
                targ[layer] += dt

            local_mse[0][seed, i] = torch.mean(torch.square(targ[1] - model.wts[0](targ[0])))
            p = softmax(model.wts[-1](targ[-2]))
            targ[-1] = (1 - model.mod_prob) * global_target + model.mod_prob * p






def record_infer_stnd(model, images, global_target, local_mse, seed):
    with torch.no_grad():
        h = model.initialize_values(images)
        targ = [h[i].clone() for i in range(model.num_layers)]
        targ[-1] = (1 - model.mod_prob) * global_target.clone() + model.mod_prob * softmax(targ[-1])

        eps = [torch.zeros_like(h[i]) for i in range(model.num_layers)]
        p = [model.wts[i](targ[i].detach()) for i in range(model.num_layers - 1)]
        p.insert(0, h[0].clone())

        # Iterative updates
        for i in range(local_mse[0].size(1)):
            # Compute errors
            for layer in range(1, model.num_layers - 1):
                eps[layer] = (targ[layer] - p[layer])  # MSE gradient
                local_mse[layer-1][seed, i] = torch.mean(torch.square(eps[layer]))

            eps[-1] = (targ[-1] - softmax(p[-1]))
            local_mse[-1][seed, i] = torch.mean(torch.square(eps[-1]))


            #Update targets
            for layer in range(1, model.num_layers - 1):
                dfdt = eps[layer+1].matmul(model.wts[layer][1].weight) * model.func_d(targ[layer])
                dt = decay * model.gamma * (dfdt - eps[layer])
                targ[layer] += dt

            # Update output target
            targ[-1] = (1 - model.mod_prob) * global_target + model.mod_prob * p[-1]

            # Compute Predictions
            for layer in range(1, model.num_layers-1):
                p[layer] = model.wts[layer-1](targ[layer-1])





def train_model(train_loader, test_loader, model, mse_stnd, mse_local, max_iters, dev, seed, train_wts):
    with torch.no_grad():
        iter = 0

        if train_wts:
            while iter < max_iters:
                for batch_idx, (images, y) in enumerate(train_loader):
                    images = images.view(y.size(0), -1).to(dev)
                    y = y.to(dev)
                    target = F.one_hot(y, num_classes=10)

                    # Store initial weights and update
                    _, _ = model.train_wts(images.detach(), target.detach(), y)

                    # End if max training iterations reached
                    iter += 1
                    if iter == max_iters:
                        return

        for batch_idx, (images, y) in enumerate(test_loader):
            images = images.view(y.size(0), -1).to(dev)
            y = y.to(dev)
            target = F.one_hot(y, num_classes=10)
            record_infer_stnd(model, images, target, mse_stnd, seed)
            record_infer_seq(model, images, target, mse_local, seed)
            break



def train(models, batch_size, data, dev, max_iters, mse_stnd, mse_local, train_wts):
    with torch.no_grad():
        for s in range(len(models)):
            print(f'Seed:{s}')
            train_loader, test_loader, _, _ = get_data(batch_size, data=data)
            train_model(train_loader, test_loader, models[s], mse_stnd, mse_local, max_iters, dev, s, train_wts)




def training_run(max_iters=5000, batch_size=64, data=1, num_seeds=10, n_hlayers=4, n_iter=8, train_wts=False):
    with torch.no_grad():
        # Create Models
        model_dim = [3072]
        for ln in range(n_hlayers):
            model_dim.append(1024)
        model_dim.append(10)

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        models = []
        for m in range(num_seeds):
            models.append(IL.IL(model_dim, smax=True, n_iter=n_iter, gamma=.02, alpha=.5, type=0, beta=100))

        # To Device
        for i in range(len(models)):
            models[i].to(dev)

        ######################################################
        # Create Containers
        mse_stnd = [torch.zeros(num_seeds, n_iter) for l in range(n_hlayers + 1)]
        mse_seq = [torch.zeros(num_seeds, n_iter) for l in range(n_hlayers + 1)]

        ######################################################
        # Train
        train(models, batch_size, data, dev, max_iters, mse_stnd, mse_seq, train_wts)

        mse_stnd_m = [torch.mean(mse_stnd[l], dim=0) for l in range(len(mse_stnd))]
        mse_stnd_std = [torch.std(mse_stnd[l], dim=0) for l in range(len(mse_stnd))]

        mse_seq_m = [torch.mean(mse_seq[l], dim=0) for l in range(len(mse_stnd))]
        mse_seq_std = [torch.std(mse_seq[l], dim=0) for l in range(len(mse_stnd))]

        print('Standard:', mse_stnd_m[0], mse_stnd_m[0].shape, mse_stnd_std[0].shape)
        print('Sequential:', mse_seq_m[0], mse_seq_m[0].shape, mse_seq_std[0].shape)

        with open(f'data/errAnalyze_train{train_wts}.data','wb') as filehandle:
            pickle.dump([mse_stnd_m, mse_stnd_std, mse_seq_m, mse_seq_std], filehandle)