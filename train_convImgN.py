import BP
import IL_ConvBig
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

bce = torch.nn.BCELoss(reduction='none')
mse = torch.nn.MSELoss(reduction='none')
softmax = torch.nn.Softmax(dim=1)
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
relu = torch.nn.ReLU()

# Load MNIST Data
def get_data(batch_size=100):
    traindir = 'C:/Users/nalon/Documents/PythonScripts/tiny-imagenet-200/train'
    testdir = 'C:/Users/nalon/Documents/PythonScripts/tiny-imagenet-200/val'

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.ImageFolder(traindir, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.ImageFolder(testdir, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=True)

    return train_loader, test_loader



def test(test_losses, test_top1, test_top5, model, test_loader, seed, lr, dev):
    with torch.no_grad():
        test_top1[lr][seed].append(0)
        test_top5[lr][seed].append(0)
        test_losses[lr][seed].append(0)

        testn = 0
        for batch_idx, (images, y) in enumerate(test_loader):
            images = images.to(dev)
            y = y.to(dev)
            target = F.one_hot(y, num_classes=200).to(dev)

            # Test and record losses and accuracy over whole test set
            h = model.initialize_values(images)

            global_loss = torch.mean(mse(softmax(h[-1]), target).sum(1))
            test_top1[lr][seed][-1] += utilities.compute_num_correct(softmax(h[-1]), y)
            test_top5[lr][seed][-1] += utilities.compute_num_correct_top_k(softmax(h[-1]), y, k=5)
            test_losses[lr][seed][-1] += global_loss.item()
            testn += images.size(0)

        test_top1[lr][seed][-1] /= testn
        test_top5[lr][seed][-1] /= testn
        test_losses[lr][seed][-1] /= testn



def train_model(train_loader, test_loader, model, seed, lr, test_losses, test_top1, test_top5, epochs, dev, b_size):
    #Initial test
    test(test_losses, test_top1, test_top5, model, test_loader, seed, lr, dev)

    #Train and test each epoch
    hlfway = int(100000 / b_size / 2)
    for ep in range(epochs):
        for batch_idx, (images, y) in enumerate(train_loader):

            images = images.to(dev)
            y = y.to(dev)
            target = F.one_hot(y, num_classes=200)
            if images.size(0) == b_size:
                _, _ = model.train_wts(images.detach(), target.detach(), y)

            #Test halfway through epoch
            if batch_idx == hlfway:
                test(test_losses, test_top1, test_top5, model, test_loader, seed, lr, dev)
                #print(ep + .5, 'Top1:', test_top1[lr][seed][-1] * 100, 'Top5:', test_top5[lr][seed][-1] * 100)

        #Test after epoch
        test(test_losses, test_top1, test_top5, model, test_loader, seed, lr, dev)
        #print(ep+1, 'Top1:', test_top1[lr][seed][-1] * 100, 'Top5:', test_top5[lr][seed][-1] * 100)




def train(models, batch_size, dev, epochs, test_losses, test_top1, test_top5):
    for l in range(len(models)):
        print(f'Training Alpha:{models[l][0].alpha}')
        for m in range(len(models[0])):

            train_loader, test_loader = get_data(batch_size)
            train_model(train_loader, test_loader, models[l][m], m, l, test_losses, test_top1, test_top5, epochs, dev, batch_size)

            print(f'Seed:{m}', f'MaxAcc:{max(test_top1[l][m])} ({max(test_top5[l][m])})',
                  f'MaxEarlyAcc:{max(test_top1[l][m][0:int(epochs/5)])} ({max(test_top1[l][m][0:int(epochs/5)])})',
                  f'LastAcc:{test_top1[l][m][-1]} ({test_top5[l][m][-1]})')



#
def training_run(epochs=120, batch_size=64, num_seeds=1, alpha=[1], model_type=0, beta=100, arch=0, gamma=.05):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = []

    for l in range(len(alpha)):
        #Add list of seeds at this learning rate
        models.append([])
        for m in range(num_seeds):
            # BP-SGD
            if model_type == 0:
                models[-1].append(IL_ConvBig.IL(type=0, alpha=alpha[l], arch=arch))

            # BP-Adam
            elif model_type == 1:
                models[-1].append(IL_ConvBig.IL(type=1, alpha=alpha[l], arch=arch))

            # IL
            elif model_type == 2:
                models[-1].append(IL_ConvBig.IL(n_iter=4, gamma=gamma, beta=beta, type=2, alpha=alpha[l], arch=arch))

            # IL-MQ
            elif model_type == 3:
                models[-1].append(IL_ConvBig.IL(n_iter=4, gamma=gamma, beta=beta, type=3, alpha=alpha[l], arch=arch))

            # IL-Adam
            elif model_type == 4:
                models[-1].append(IL_ConvBig.IL(n_iter=4, gamma=gamma, beta=beta, type=4, alpha=alpha[l], arch=arch))


        # To Device
        for i in range(len(models[-1])):
            models[-1][i].to(dev)

    #################################################
    # Create Containers
    test_losses = [[[] for m in range(num_seeds)] for m in range(len(models))]  # [model_lr][model_seed]
    test_top1 = [[[] for m in range(num_seeds)] for m in range(len(models))]  # [model_lr][model_seed]
    test_top5 = [[[] for m in range(num_seeds)] for m in range(len(models))]  # [model_lr][model_seed]

    #################################################
    # Train
    print(f'\nTRAINING MODEL TYPE {model_type}')
    train(models, batch_size, dev, epochs, test_losses, test_top1, test_top5)

    # Store Data
    best_test_acc = torch.mean(torch.tensor([max(test_top1[0][x]) for x in range(len(test_top1[0]))])).item()
    best_lr = 0
    for l in range(1, len(models)):
        ac = torch.mean(torch.tensor([max(test_top1[l][x]) for x in range(len(test_top1[0]))])).item()
        if best_test_acc < ac:
            best_test_acc = ac
            best_lr = l

    with open(f'data/ConvImgN_Type{model_type}_epochs{epochs}_mbatch{batch_size}_arch{arch}.data','wb') as filehandle:
        pickle.dump([test_top1[best_lr], test_losses[best_lr], test_top5[best_lr], alpha[best_lr]], filehandle)