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
NLL = torch.nn.NLLLoss(reduction='sum')

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

    return train_loader, test_loader




def vectorize_params(model):
    with torch.no_grad():
        params = model.wts[0][-1].weight.data.clone().view(-1)
        params = torch.cat((params, model.wts[0][-1].bias.data.clone().view(-1)), dim=0)

        for l in range(1,len(model.wts)):
            params = torch.cat((params, model.wts[l][-1].weight.data.clone().view(-1)), dim=0)
            params = torch.cat((params, model.wts[l][-1].bias.data.clone().view(-1)), dim=0)

        return params


def test(model, test_loader, dev):
    with torch.no_grad():
        test_accuracies = 0
        testn = 0

        # Test and record losses and accuracy over test set
        for batch_idx, (images, y) in enumerate(test_loader):
            images = images.view(y.size(0), -1).to(dev)
            y = y.to(dev)
            target = F.one_hot(y, num_classes=10).to(dev)
            h = model.initialize_values(images)
            test_accuracies += utilities.compute_num_correct(softmax(h[-1]), y)
            testn += images.size(0)

        print('Test Acc', test_accuracies / testn)



def get_prox(model, h_hat, target, y, dev='cuda', NLL_loss=True):
    with torch.no_grad():
        betas = [.01, .1, 1, 10, 100]
        proxs = torch.zeros(h_hat[0].size(0), 5).to(dev)

        old_prms = vectorize_params(model)
        for d in range(h_hat[0].size(0)):
            #Get h_hat from mini-batch element d
            h_hat_d = [h_hat[x][d].view(1, -1) for x in range(len(h_hat))]

            #Get new params after update
            temp_model = copy.deepcopy(model)
            if model.type == 0:
                temp_model.LMS_update(h_hat_d)
            else:
                temp_model.MQ_update(h_hat_d)

            # Get loss after weight update
            h_N = temp_model.initialize_values(h_hat_d[0])[-1]
            if NLL_loss:
                L = NLL(torch.log(softmax(h_N)), y[d].view(1).detach())
            else:
                L = .5 * torch.square(target[d] - softmax(h_N)).sum()

            # Get squared parameter changed weighted by beta
            new_prms = vectorize_params(temp_model)
            delta_theta = .5 * torch.square(new_prms - old_prms).sum()
            for b in range(len(betas)):
                proxs[d, b] = L + delta_theta / betas[b]

        return torch.mean(proxs, dim=0).to('cpu')



def get_prox_inference(model, test_loader, dev='cuda', NLL_loss=True):
    with torch.no_grad():
        prox_means = torch.zeros(model.T, 5)
        for batch_idx, (images, y) in enumerate(test_loader):
            images = images.view(y.size(0), -1).to(dev)
            y = y.to(dev)
            target = F.one_hot(y, num_classes=10)

            # Initialize h_hat = h and create error containers
            h = model.initialize_values(images)
            h_hat = [h[i].clone() for i in range(model.num_layers)]
            h_hat[-1] = (1 - model.mod_prob) * target.clone() + model.mod_prob * softmax(h_hat[-1])
            eps = [torch.zeros_like(h[i]) for i in range(model.num_layers)]


            # Iterative updates to activities
            for t in range(model.T):
                # Compute errors at hidden layers
                for layer in range(1, model.num_layers - 1):
                    eps[layer] = (h_hat[layer] - model.wts[layer - 1](h_hat[layer - 1]))  # MSE gradient

                # Compute errors at output layer
                eps[-1] = h_hat[-1] - softmax(model.wts[-1](h_hat[-2]))  # Cross-ent w/ softmax gradient

                px_m = get_prox(model, h_hat, target, y, dev=dev, NLL_loss=NLL_loss)
                prox_means[t] += px_m


                '''Fr = sum([torch.mean(torch.square(eps[x]).sum(1)) for x in range(len(eps))])
                print(t, f'Prox:{px_m}, FreeEnergy:{Fr.item()}')'''

                # Update Hidden Layer Targets
                for layer in range(1, model.num_layers - 1):
                    dfdt = eps[layer + 1].matmul(model.wts[layer][1].weight) * model.func_d(h_hat[layer])
                    dt = model.gamma * (dfdt - eps[layer])
                    h_hat[layer] = h_hat[layer] + dt

                # Update output target
                h_hat[-1] = (1 - model.mod_prob) * target + model.mod_prob * softmax(model.wts[-1](h_hat[-2]))

        return prox_means




def get_prox_seq_inference(model, test_loader, dev='cuda', NLL_loss=True):
    with torch.no_grad():
        prox_means = torch.zeros(model.T, 5)
        for batch_idx, (images, y) in enumerate(test_loader):
            images = images.view(y.size(0), -1).to(dev)
            y = y.to(dev)
            target = F.one_hot(y, num_classes=10)

            # Initialize h_hat = h and create error containers
            h = model.initialize_values(images)
            h_hat = [h[i].clone() for i in range(model.num_layers)]
            h_hat[-1] = (1 - model.mod_prob) * target.clone() + model.mod_prob * softmax(h_hat[-1])

            # Update for T steps
            for t in range(model.T):
                px_m = get_prox(model, h_hat, target, y, dev=dev, NLL_loss=NLL_loss)
                prox_means[t] += px_m

                #print(t, f'Prox:{px_m}')

                # Compute decay, initialize p
                decay = (1 / (1 + t))
                p = model.wts[-1](h_hat[-2])

                # Update each hidden layer target in a serial fashion
                for layer in reversed(range(1, model.num_layers - 1)):
                    # Compute error
                    if layer < model.num_layers - 2:
                        err = h_hat[layer + 1] - p  # MSE gradient
                    else:
                        err = (h_hat[-1] - softmax(p))  # Cross-ent w/ softmax gradient

                    # Update Hidden Activities
                    dfdt = err.matmul(model.wts[layer][1].weight) * model.func_d(h_hat[layer])
                    p = model.wts[layer - 1](h_hat[layer - 1])
                    e_top = h_hat[layer] - p
                    dt = decay * model.gamma * (dfdt - e_top)
                    h_hat[layer] = h_hat[layer] + dt

                # Update output layer
                h_hat[-1] = (1 - model.mod_prob) * target + model.mod_prob * softmax(model.wts[-1](h_hat[-2]))

        return prox_means




def train_model(train_loader, test_loader, model, dev, test_iter=0, model_type=0, NLL_loss=True):
    with torch.no_grad():
        iter = 0
        while iter <= test_iter:
            for batch_idx, (images, y) in enumerate(train_loader):
                images = images.view(y.size(0), -1).to(dev)
                y = y.to(dev)
                target = F.one_hot(y, num_classes=10)

                # Store initial weights and and get initial proximal quantity every 100 iterations
                if iter == test_iter:
                    if model_type == 1:
                        px_ms = get_prox_inference(model, test_loader, dev=dev, NLL_loss=NLL_loss)
                    else:
                        px_ms = get_prox_seq_inference(model, test_loader, dev=dev, NLL_loss=NLL_loss)
                    return px_ms

                _, _ = model.train_wts(images, target, y)
                '''if iter % 50 == 0:
                    test(model, test_loader, dev)'''
                iter += 1



def train(models, batch_size, data, dev, test_iter=0, model_type=0, NLL_loss=True):
    with torch.no_grad():
        prox_means = torch.zeros(len(models), models[0].T, 5)

        for s in range(len(models)):
            train_loader, test_loader = get_data(batch_size, data=data)
            px_ms = train_model(train_loader, test_loader, models[s], dev, test_iter=test_iter, model_type=model_type, NLL_loss=NLL_loss)
            prox_means[s] += px_ms
        return torch.mean(prox_means, dim=0), torch.std(prox_means, dim=0)



def training_run(test_iter=1000, batch_size=64, data=2, num_seeds=1, model_type=1, n_hlayers=3, n_iter=20, alpha=.1,
                 gamma=.012, opt_type=0, NLL_loss=True):

    with torch.no_grad():
        if opt_type == 0:
            opt = ''
        elif opt_type == 1:
            opt = '_MQ'
        else:
            opt = '_Adam'

        ls_name = ''
        if NLL_loss:
            ls_name = '_NLL'

        # Create Models
        model_dim = [3072]
        for ln in range(n_hlayers):
            model_dim.append(1024)
        model_dim.append(10)

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        models = []
        for m in range(num_seeds):
            if model_type == 0:
                models.append(IL.IL(model_dim, smax=True, n_iter=n_iter, gamma=gamma, alpha=alpha, type=opt_type))
            else:
                models.append(IL_Stnd.IL(model_dim, smax=True, n_iter=n_iter, gamma=gamma, alpha=alpha, type=opt_type))
            models[m].to(dev)

        #################################################
        # Train
        print(f'Training   Model:{model_type}')
        px_m, px_std = train(models, batch_size, data, dev, test_iter, model_type, NLL_loss=NLL_loss)
        print(px_m, '\n')


        with open(f'data/proxAnalyze_data{data}_test_iter{test_iter}_niter{n_iter}_mType{model_type}{opt}{ls_name}.data','wb') as filehandle:
            pickle.dump([px_m, px_std], filehandle)