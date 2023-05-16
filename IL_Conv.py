import torch
from torch import nn
from utilities import sigmoid_d
from utilities import tanh_d
import math

relu = torch.nn.ReLU()
l_relu = nn.LeakyReLU(0.5)
mse = torch.nn.MSELoss(reduction='sum')
bce = torch.nn.BCELoss(reduction='sum')
NLL = nn.NLLLoss(reduction='sum')
softmax = torch.nn.Softmax(dim=1)


class IL(nn.Module):
    def __init__(self, lr=.001, type=0, n_iter=25, beta=100, gamma=.05, alpha=0, lr_min=.001, r=.000001, network=0):
        super().__init__()

        self.network = network
        self.num_layers = 5 + network            # 5 layers for network 1, 6 layers for network 2
        self.conv = True
        self.n_iter = n_iter
        self.l_rate = lr
        self.N = 0
        self.type = type   #BP=0, BP-Adam=1, IL=2, IL-MQ=3
        self.alpha = alpha
        self.beta = beta
        self.mod_prob = 1 / (1 + self.beta)
        self.gamma = gamma
        self.lr_min = lr_min
        self.wt_var = [alpha for x in range(4)]
        self.r = r

        if self.network == 0:
            self.wts = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=(5,5), stride=(2,2), bias=True),
                    nn.ReLU()
                ),

                nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=(5,5), stride=(2,2), bias=True),
                    nn.ReLU()
                ),

                nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=(3,3), stride=(2,2), bias=True),
                    nn.ReLU(),
                    nn.Flatten()
                ),

                nn.Sequential(
                    nn.Linear(1024, 10, bias=True)
                )
            )

        else:
            self.wts = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=True),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2)
                ),

                nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), bias=True),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2)
                ),

                nn.Sequential(
                    nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), bias=True),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Flatten()
                ),

                nn.Sequential(
                    nn.Linear(524, 256, bias=True)
                ),

                nn.Sequential(
                    nn.Linear(256, 10, bias=True)
                )
            )




        if self.type == 1:
            self.bp_optim = torch.optim.Adam(self.wts.parameters(), lr=self.l_rate)
        else:
            self.bp_optim = torch.optim.SGD(self.wts.parameters(), lr=self.l_rate)

        self.optims = self.create_optims()




    ############################## COMPUTE FORWARD VALUES ##################################
    def initialize_values(self, x):
        with torch.no_grad():
            h = [torch.randn(1, 1) for i in range(self.num_layers)]

            #First h is the input
            h[0] = x.clone()

            #Compute FF values
            for i in range(1, self.num_layers):
                h[i] = self.wts[i-1](h[i-1].detach())
            return h


    def create_optims(self):
        if self.type == 4:
            optims = []
            for l in range(0, self.num_layers - 1):
                optims.append(torch.optim.Adam(self.wts[l].parameters(), lr=self.alpha))
            return optims
        else:
            return None


    ############################## Minimize F w.r.t. Neuron Activities ##################################
    def compute_targets(self, h, global_target):
        with torch.no_grad():
            targ = [h[i].clone() for i in range(self.num_layers)]
            targ[-1] = (1 - self.mod_prob) * global_target.clone() + self.mod_prob * softmax(targ[-1])

        # Iterative updates
        for i in range(self.n_iter):
            decay = (1 / (1 + i))
            with torch.no_grad():
                p = self.wts[-1](targ[-2])
            for layer in reversed(range(1, self.num_layers - 1)):
                with torch.no_grad():
                    self.bp_optim.zero_grad()
                    # Compute error
                    if layer < self.num_layers - 2:
                        err = targ[layer + 1] - p  # MSE gradient
                    else:
                        err = targ[-1] - softmax(p)  # Cross-ent w/ softmax gradient

                #Update Targets
                _, dfdt = torch.autograd.functional.vjp(self.wts[layer], targ[layer], err)
                with torch.no_grad():
                    p = self.wts[layer - 1](targ[layer - 1])
                    e_top = targ[layer] - p
                    dt = decay * self.gamma * (dfdt - e_top)
                    targ[layer] = targ[layer] + dt

            with torch.no_grad():
                targ[-1] = (1 - self.mod_prob) * global_target + self.mod_prob * softmax(self.wts[-1](targ[-2]))

        return targ



    ############################## TRAIN ##################################
    def train_wts(self, x, global_target, y):

        # Record params before update
        with torch.no_grad():
            # Get feedforward and target values
            h = self.initialize_values(x)

            # Get targets
            if self.type > 1:
                h_hat = self.compute_targets(h, global_target)

        # Count datapoint
        self.N += 1

        # Update weights
        if self.type < 2:
            self.BP_update(x, y)
        elif self.type == 2:
            self.LMS_update(h_hat, y)
        elif self.type == 3:
            self.MQ_update(h_hat, y)
        elif self.type == 4:
            self.Adam_update(h_hat, y)

        '''# Count datapoint
        self.N += 1'''

        return False, h[-1]



    def BP_update(self, x, y):
        ## Get BP Gradients
        z = x.clone().detach()
        for i in range(0, self.num_layers - 1):
            z = self.wts[i](z)

        #Get loss
        loss = NLL(torch.log(softmax(z)), y.detach()) / z.size(0)

        #Update
        self.bp_optim.zero_grad()
        loss.backward()
        self.bp_optim.step()



    def LMS_update(self, targ, y):

        ## Update each weight matrix
        for i in range(self.num_layers-1):
            #Compute local losses, sum neuron-wise and avg batch-wise
            if i < (self.num_layers - 2):
                p = self.wts[i](targ[i].detach())
                loss = .5 * mse(p, targ[i+1].detach()) / p.size(0)
            else:
                p = self.wts[i](targ[i].detach())
                loss = NLL(torch.log(softmax(p)), y.detach()) / p.size(0)

            #Compute weight gradients
            self.bp_optim.zero_grad()
            loss.backward()
            with torch.no_grad():
                # Update weights with normalized step size and precision weighting
                self.wts[i][0].weight.data -= self.wts[i][0].weight.grad * self.alpha
                self.wts[i][0].bias.data -= self.wts[i][0].bias.grad * self.alpha



    def MQ_update(self, targ, y):
        for i in range(self.num_layers - 1):
            # Compute local losses, sum neuron-wise and avg batch-wise
            if i < (self.num_layers - 2):
                p = self.wts[i](targ[i].detach())
                loss = .5 * mse(p, targ[i + 1].detach()) / p.size(0)
            else:
                p = self.wts[i](targ[i].detach())
                loss = NLL(torch.log(softmax(p)), y.detach()) / p.size(0)
                # loss = -torch.mean((targ[-1].detach() * torch.log(softmax(p))).sum(1))

            # Compute weight gradients
            self.bp_optim.zero_grad()
            loss.backward()
            with torch.no_grad():
                # Update weights
                wtfrac = self.alpha / ((self.wt_var[i]) + self.r) + self.lr_min
                self.wts[i][0].weight.data -= wtfrac * self.wts[i][0].weight.grad
                self.wts[i][0].bias.data -= wtfrac * self.wts[i][0].bias.grad

                # Update variances as moving average of absolute gradient
                self.avgRt = min(self.N / (self.N + 1), .999)
                params = torch.cat((self.wts[i][0].weight.grad.view(-1).clone(), self.wts[i][0].bias.grad.view(-1).clone()), dim=0)
                self.wt_var[i] = self.avgRt * self.wt_var[i] + (1 - self.avgRt) * torch.mean(torch.abs(params))




    def Adam_update(self, targ, y):
        for i in range(self.num_layers - 1):
            # Compute local losses, sum neuron-wise and avg batch-wise
            if i < (self.num_layers - 2):
                p = self.wts[i](targ[i].detach())
                loss = .5 * mse(p, targ[i + 1].detach()) / p.size(0)
            else:
                p = self.wts[i](targ[i].detach())
                loss = NLL(torch.log(softmax(p)), y.detach()) / p.size(0)

            # Compute weight gradients, update with adam
            self.optims[i].zero_grad()
            loss.backward()
            self.optims[i].step()
