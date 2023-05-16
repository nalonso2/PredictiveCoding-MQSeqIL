import torch
from torch import nn
from utilities import relu_d
from utilities import tanh_d
import math

softmax = nn.Softmax(dim=1)
NLL = nn.NLLLoss(reduction='sum')
mse = torch.nn.MSELoss(reduction='sum')
bce = torch.nn.BCELoss(reduction='sum')
cos = torch.nn.CosineSimilarity(dim=0)
relu = nn.ReLU()

class IL(nn.Module):
    def __init__(self, layer_szs, n_iter=25, gamma=.015, type=0, bias=True, smax=True,
                 func=nn.ReLU(), func_d=relu_d, alpha=.5, beta=100, lr_min=.001, r=.000001, v=.9999):
        super().__init__()

        self.num_layers = len(layer_szs)
        self.conv = False
        self.layer_szs = layer_szs
        self.func = func
        self.func_d = func_d
        self.T = n_iter
        self.bias = bias
        self.v = v
        self.type = type      #IL = 0, IL-MQ = 1, IL-Adam = 2
        self.wts = self.create_wts()
        self.gamma = gamma
        self.smax = smax
        self.N = 0
        self.alpha = alpha
        self.beta = beta
        self.mod_prob = 1 / (1 + self.beta)
        self.lr_min = lr_min
        self.wt_var = [1 for x in range(len(self.wts))]
        self.bias_var = [.001 for x in range(len(self.wts))]
        self.r = r
        self.alphas = []
        self.optim = self.create_optims()


    def create_wts(self):
        with torch.no_grad():
            w = nn.ModuleList([nn.Sequential(nn.Linear(self.layer_szs[0], self.layer_szs[1], bias=self.bias))])
            for l in range(1, self.num_layers - 1):
                w.append(nn.Sequential(self.func, nn.Linear(self.layer_szs[l], self.layer_szs[l+1], bias=self.bias)))
            return nn.ModuleList(w)


    def create_optims(self):
        if self.type == 2:
            optims = []
            for l in range(0, self.num_layers - 1):
                optims.append(torch.optim.Adam(self.wts[l].parameters(), lr=self.alpha))
            return optims
        else:
            return None



    ############################## COMPUTE FORWARD VALUES ##################################
    def initialize_values(self, x):
        with torch.no_grad():
            h = [torch.randn(1, 1) for i in range(self.num_layers)]

            # First h is the input
            h[0] = x.clone()

            # Compute all hs except last one.
            for i in range(1, self.num_layers):
                h[i] = self.wts[i - 1](h[i - 1])
        return h




    ############################## Minimize F w.r.t. Neuron Activities ##################################
    def compute_targets(self, h, global_target):
        with torch.no_grad():
            #Initialize h_hat = h and create error containers
            h_hat = [h[i].clone() for i in range(self.num_layers)]
            if self.smax:
                h_hat[-1] = (1 - self.mod_prob) * global_target.clone() + self.mod_prob * softmax(h_hat[-1])
            else:
                h_hat[-1] = (1 - self.mod_prob) * global_target.clone() + self.mod_prob * torch.sigmoid(h_hat[-1])

            eps = [torch.zeros_like(h[i]) for i in range(self.num_layers)]

            # Iterative updates to activities
            for t in range(self.T):
                # Compute errors at hidden layers
                for layer in range(1, self.num_layers - 1):
                    eps[layer] = (h_hat[layer] - self.wts[layer - 1](h_hat[layer-1]))  # MSE gradient


                # Compute errors at output layer
                if self.smax:
                    eps[-1] = h_hat[-1] - softmax(self.wts[-1](h_hat[-2]))  # Cross-ent w/ softmax gradient
                else:
                    eps[-1] = h_hat[-1] - torch.sigmoid(self.wts[-1](h_hat[-2]))  # BCE


                # Update Hidden Layer Targets
                for layer in range(1, self.num_layers - 1):
                    dfdt = eps[layer + 1].matmul(self.wts[layer][1].weight) * self.func_d(h_hat[layer])
                    dt = self.gamma * (dfdt - eps[layer])
                    h_hat[layer] = h_hat[layer] + dt / (t+1)


                #Update output target
                if self.smax:
                    h_hat[-1] = (1 - self.mod_prob) * global_target + self.mod_prob * softmax(self.wts[-1](h_hat[-2]))
                else:
                    h_hat[-1] = (1 - self.mod_prob) * global_target + self.mod_prob * torch.sigmoid(self.wts[-1](h_hat[-2]))

        return h_hat



    ############################## TRAIN ##################################
    def train_wts(self, x, global_target, y):

        # Record params before update
        with torch.no_grad():
            # Get feedforward and target values
            h = self.initialize_values(x)

            # Get targets
            h_hat = self.compute_targets(h, global_target)

            # Update weights
            if self.type == 0:
                self.LMS_update(h_hat)
            elif self.type == 1:
                self.MQ_update(h_hat)
            elif self.type == 2:
                self.Adam_update(h_hat)

            # Count datapoint
            self.N += 1

            return False, h[-1]




    #Best performing, variance computed on normalized gradient w/o epsilon.
    def LMS_update(self, h_hat):
        # Update each weight matrix
        #prec = torch.zeros(0).to('cuda')
        for i in range(self.num_layers - 1):
            # Compute local losses, sum neuron-wise and avg batch-wise
            with torch.no_grad():
                if i < (self.num_layers - 2):
                    p = self.wts[i](h_hat[i])
                    e = h_hat[i + 1] - p
                elif self.smax:
                    p = self.wts[i](h_hat[i])
                    e = h_hat[i + 1] - softmax(p)
                else:
                    p = self.wts[i](h_hat[i])
                    e = h_hat[i + 1] - torch.sigmoid(p)

                # Compute weight gradients and norm
                self.wts[i][-1].weight.grad = e.t().matmul(self.func(h_hat[i])) / e.size(0)
                self.wts[i][-1].bias.grad = torch.mean(e, dim=0)

                # Update weights with normalized step size and precision weighting
                nm = torch.mean(torch.square(h_hat[i]).sum(1)) + 1
                self.wts[i][-1].weight.data += self.wts[i][-1].weight.grad / nm #* self.alpha
                self.wts[i][-1].bias.data += self.wts[i][-1].bias.grad / nm # * self.alpha



    #Best performing, variance computed on normalized gradient
    def MQ_update(self, h_hat):
        # Update each weight matrix
        for i in range(self.num_layers - 1):
            # Compute local losses, sum neuron-wise and avg batch-wise
            with torch.no_grad():
                if i < (self.num_layers - 2):
                    p = self.wts[i](h_hat[i])
                    e = h_hat[i + 1] - p
                elif self.smax:
                    p = self.wts[i](h_hat[i])
                    e = h_hat[i + 1] - softmax(p)
                else:
                    p = self.wts[i](h_hat[i])
                    e = h_hat[i + 1] - torch.sigmoid(p)

                #Compute weight gradients and norm
                self.wts[i][-1].weight.grad = e.t().matmul(self.func(h_hat[i])) / e.size(0)
                self.wts[i][-1].bias.grad = torch.mean(e, dim=0)

                #Update
                wtfrac = self.alpha / (self.wt_var[i] + self.r) + self.lr_min
                self.wts[i][-1].weight.data += wtfrac * self.wts[i][-1].weight.grad
                self.wts[i][-1].bias.data += wtfrac * self.wts[i][-1].bias.grad

                # Update variances as moving average of absolute gradient
                self.avgRt = min(self.N / (self.N+1), self.v)
                grads = torch.cat((self.wts[i][-1].weight.grad.view(-1).clone(), self.wts[i][-1].bias.grad.view(-1).clone()), dim=0)
                self.wt_var[i] = self.avgRt * self.wt_var[i] + (1 - self.avgRt) * torch.mean(torch.abs(grads))



    def Adam_update(self, h_hat):
        with torch.no_grad():
            # Update each weight matrix
            for i in range(self.num_layers - 1):

                # Compute local losses, sum neuron-wise and avg batch-wise
                if i < (self.num_layers - 2):
                    p = self.wts[i](h_hat[i])
                    e = h_hat[i + 1] - p
                elif self.smax:
                    p = self.wts[i](h_hat[i])
                    e = h_hat[i + 1] - softmax(p)
                else:
                    p = self.wts[i](h_hat[i])
                    e = h_hat[i + 1] - torch.sigmoid(p)

                # Compute weight gradients and norm
                self.wts[i][-1].weight.grad = -e.t().matmul(self.func(h_hat[i])) / e.size(0)
                self.wts[i][-1].bias.grad = -torch.mean(e, dim=0)
                self.optim[i].step()





    def ImSGD_Update(self, x, y, gl_target, beta=1):
        # Compute an approximate implicit SGD/proximal, by performing SGD on the proximal objective for 500 iterations
        SGD_optim = torch.optim.SGD(self.wts.parameters(), lr=self.alpha)

        with torch.no_grad():
            wts_b = [self.wts[w][-1].weight.clone() for w in range(self.num_layers - 1)]
            bias_b = [self.wts[w][-1].bias.clone() for w in range(self.num_layers - 1)]
            step_sz = .01

        old_prox = 100
        for iter in range(500):
            # Get BP Gradients
            z = x.clone().detach()
            for n in range(0, self.num_layers - 1):
                z = self.wts[n](z)

            # Get loss gradients
            if self.smax:
                loss = NLL(torch.log(softmax(z)), y.detach()) / z.size(0)  # CrossEntropy
            else:
                loss = bce(torch.sigmoid(z), gl_target.detach()) / z.size(0)

            SGD_optim.zero_grad()
            loss.backward()

            # Perform update (either Adam or SGD) with loss gradients and regularization gradients
            with torch.no_grad():
                prox = loss.item()
                for i in range(self.num_layers - 1):
                    prox += .5 / beta * torch.square(self.wts[i][-1].weight - wts_b[i]).sum().item()
                    prox += .5 / beta * torch.square(self.wts[i][-1].bias - bias_b[i]).sum().item()
                    self.wts[i][-1].weight -= step_sz * (self.wts[i][-1].weight.grad + .5 / beta * (self.wts[i][-1].weight - wts_b[i]))
                    self.wts[i][-1].bias -= step_sz * (self.wts[i][-1].bias.grad + .5 / beta * (self.wts[i][-1].bias - bias_b[i]))

                if old_prox < prox:
                    step_sz *= .5
                old_prox = prox

            SGD_optim.zero_grad()


    def SGD_Update(self, x, y, gl_target):
        with torch.no_grad():
            SGD_optim = torch.optim.SGD(self.wts.parameters(), lr=self.alpha)
        z = x.clone().detach()
        for n in range(self.num_layers - 1):
            z = self.wts[n](z)

        # Get loss
        if self.smax:
            loss = NLL(torch.log(softmax(z)), y.detach()) / z.size(0)  # CrossEntropy
        else:
            loss = bce(torch.sigmoid(z), gl_target.detach()) / z.size(0)


        # SGD Update
        SGD_optim.zero_grad()
        loss.backward()
        SGD_optim.step()