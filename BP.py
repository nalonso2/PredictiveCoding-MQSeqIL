import torch
from torch import nn
from utilities import relu_d
from utilities import tanh_d
from torch.optim.lr_scheduler import StepLR

softmax = nn.Softmax(dim=1)
NLL = nn.NLLLoss(reduction='sum')
mse = torch.nn.MSELoss(reduction='sum')
bce = torch.nn.BCELoss(reduction='sum')
cos = torch.nn.CosineSimilarity(dim=0)
relu = nn.ReLU()


class BP(nn.Module):
    def __init__(self, layer_szs, alpha=.001, func=nn.ReLU(), type=0, n_iter=25, smax=True, bias=True,
                 lr_decay=True, decay_rt=.00003):
        super().__init__()

        self.num_layers = len(layer_szs)
        self.layer_szs = layer_szs
        self.n_iter = n_iter
        self.alpha = alpha
        self.type = type  # 0=BP-SGD, 1=BP-Adam, 2=BP-Nest, 3=BP-RMS
        self.bias = bias
        self.func = func
        self.wts = self.create_wts()
        self.smax= smax
        self.decay_rt = decay_rt
        self.lr_decay = lr_decay
        self.N = 0
        self.mod_prob = 0

        #Vanilla SGD
        if type == 0:
            self.optim = torch.optim.SGD(self.wts.parameters(), lr=self.alpha)
        #BP-Adam
        elif type == 1:
            self.optim = torch.optim.Adam(self.wts.parameters(), lr=self.alpha)
        #BP+Nesterov
        elif type == 2:
            self.optim = torch.optim.SGD(self.wts.parameters(), lr=self.alpha, nesterov=True, momentum=.9)
        #BP+RMS
        elif type == 3:
            self.optim = torch.optim.RMSprop(self.wts.parameters(), lr=self.alpha)



    def create_wts(self):
        w = nn.ModuleList([])
        for l in range(self.num_layers - 2):
            w.append(nn.Sequential(nn.Linear(self.layer_szs[l], self.layer_szs[l + 1], bias=self.bias), self.func))
        w.append(nn.Sequential(nn.Linear(self.layer_szs[-2], self.layer_szs[-1], bias=self.bias)))

        return nn.ModuleList(w)


    ############################## COMPUTE FEEDFORWARD VALUES ##################################
    def initialize_values(self, x):
        with torch.no_grad():
            h = [torch.randn(1, 1) for i in range(self.num_layers)]

            #First h is the input
            h[0] = x.clone()

            #Compute all hs except last one. Use Tanh as in paper
            for i in range(1, self.num_layers):
                h[i] = self.wts[i-1](h[i-1].detach())
        return h




    ############################## TRAIN ##################################
    def train_wts(self, x, global_target, y):
        pred = self.BP_update(x, y, global_target)
        return False, pred


    def BP_update(self, x, y, gl_target):

        ## Get FF values
        z = x.clone().detach()
        for n in range(self.num_layers - 1):
            z = self.wts[n](z)

        #Get loss
        if self.smax:
            loss = NLL(torch.log(softmax(z)), y.detach()) / z.size(0) # CrossEntropy
        else:
            loss = bce(torch.sigmoid(z), gl_target.detach()) / z.size(0)

        #Update weights
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        #Update learning rate
        if self.lr_decay:
            self.N += 1
            with torch.no_grad():
                lr = 1 / (self.N * self.decay_rt + 1) * self.alpha
                self.mod_prob = 1 - 1 / (self.N * self.decay_rt + 1)
                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr

        return z


    '''def prox_BP_update(self, x, y, gl_target):
        # Get FF values
        h = [torch.randn(1, 1) for i in range(self.num_layers)]
        h[0] = x.clone().detach()
        for n in range(self.num_layers - 1):
            h [n+1] = self.wts[n](h[n])

        #Get loss
        if self.smax and self.crossEnt:
            loss = NLL(torch.log(softmax(h[-1])), y.detach()) / h[-1].size(0) # CrossEntropy
        elif not self.smax and self.crossEnt:
            loss = bce(torch.sigmoid(h[-1]), gl_target.detach()) / h[-1].size(0)
        else:
            loss = .5 * mse(h[-1], gl_target.detach()) / h[-1].size(0)
        self.optim.zero_grad()
        loss.backward()

        with torch.no_grad():
            #h = self.compute_values(x)
            for i in range(0, self.num_layers - 1):
                nm = torch.mean(torch.square(self.func(h[i])).sum(1)) + self.bias + self.eps
                self.wts[i][0].weight -= self.l_rate * self.wts[i][0].weight.grad / nm
                if self.bias:
                    self.wts[i][0].bias -= self.l_rate * self.wts[i][0].bias.grad / nm'''
