import pickle
import torch
import numpy as np
from utilities import sigmoid_d
from utilities import tanh_d
import matplotlib
import pylab
import math


matplotlib.rcParams['text.usetex']=False
matplotlib.rcParams['savefig.dpi']=400.
matplotlib.rcParams['font.size']=9.0
matplotlib.rcParams['figure.figsize']=(5.0,3.5)
matplotlib.rcParams['axes.formatter.limits']=[-10,10]
matplotlib.rcParams['axes.labelsize']= 9.0
matplotlib.rcParams['figure.subplot.bottom'] = .2
matplotlib.rcParams['figure.subplot.left'] = .2
matplotlib.rcParams["axes.facecolor"] = (0.8, 0.8, 0.8, 0.5)
matplotlib.rcParams['axes.edgecolor'] = 'white'
matplotlib.rcParams['grid.linewidth'] = 1.2
matplotlib.rcParams['grid.color'] = 'white'
matplotlib.rcParams['axes.grid'] = True




############### HELPER FUNCTIONS ##################
def compute_means(data, scale=1):
    with torch.no_grad():
        d_tensor = torch.zeros((len(data), len(data[0])))

        for m in range(0,len(data)):
            t = torch.tensor(data[m]).view(1,-1).clone()
            d_tensor[m] = t

        return torch.mean(d_tensor*scale, dim=0)



def compute_stds(data, scale=1):
    with torch.no_grad():
        d_tensor = torch.zeros((len(data), len(data[0])))
        for m in range(0, len(data)):
            t = torch.tensor(data[m]).view(1, -1).clone()
            d_tensor[m, 0:t.size(1)] = t

        return torch.std(d_tensor*scale, dim=0)


def compute_means_std_comp(data, scale=1):
    with torch.no_grad():

        d_tensor = torch.tensor(data[0]).view(1,-1)
        for m in range(1,len(data)):
                d_tensor = torch.cat((d_tensor, torch.tensor(data[m]).view(1,-1)), dim=1)

        # return torch.mean(torch.nan_to_num(d_tensor, posinf=pos_inf, nan=nan)).item(), torch.std(torch.nan_to_num(d_tensor, posinf=pos_inf, nan=nan)).item()
        return torch.mean(d_tensor*scale).item(), torch.std(d_tensor*scale).item()


def compute_max_means_std(data, scale=1):
    with torch.no_grad():
        maxs = []
        conv_maxs = []
        for m in range(len(data)):
            maxs.append(torch.max(torch.tensor(data[m]) * scale).item())
            conv_maxs.append(torch.argmax(torch.tensor(data[m])).item())

        return [sum(maxs) / len(maxs), torch.std(torch.tensor(maxs)).item(), sum(conv_maxs) / len(conv_maxs)]




def compute_min_means_std(data, scale=1):
    with torch.no_grad():
        mins = []
        conv_mins = []
        for m in range(len(data)):
            mins.append(torch.min(torch.tensor(data[m]) * scale).item())
            conv_mins.append(torch.argmin(torch.tensor(data[m])).item())

        return [sum(mins) / len(mins), torch.std(torch.tensor(mins)).item(), sum(conv_mins) / len(conv_mins)]









##########################################################################
def plot_conv():
    with torch.no_grad():
        dlabel = ['SVHN', 'CIFAR-10', 'Tiny ImageNet']
        labels = ['BP', 'BP-Adam', 'SeqIL', 'SeqIL-MQ', 'SeqIL-Adam']
        clr = ['black', 'blue', 'purple', 'red', 'brown']
        fig, axs = pylab.subplots(1, 3, figsize=(7, 1.5), constrained_layout=True)

        for d in range(2):
            print(f'\n{dlabel[d]}')
            for t in range(5):
                with open(f'data/Conv_Type{t}_data{d}_epochs45_64.data','rb') as filehandle:
                    testdata = pickle.load(filehandle)
                best_acc = compute_max_means_std(testdata[0], scale=100)
                acc = compute_means(testdata[0], scale=100)
                accstd = compute_stds(testdata[0], scale=100)
                idx = torch.argmax(acc)
                idx = min(idx + 3, 45)
                x = torch.linspace(0, idx, idx)
                acc = acc[0:idx]
                accstd = accstd[0:idx]

                print(f'{labels[t]} Conv', f'Test Acc Best(Mean):{round(best_acc[0], 2)}',
                                      f'  (STD):{round(best_acc[1], 2)}',
                                      f'  (Epoch#):{round(best_acc[2], 2)}',
                                      f'  LR:{testdata[2]}')

                axs[d].plot(x, 100 - acc, label=labels[t], linewidth=2.25, color=clr[t], alpha=.7)
                axs[d].fill_between(x, (100 - acc) - accstd, (100 - acc) + accstd, color=clr[t], alpha=.2)


        print('\nTiny ImageNet')
        for t in range(5):
            with open(f'data/ConvImgN_Type{t}_epochs20_mbatch64_arch0.data', 'rb') as filehandle:
                testdata = pickle.load(filehandle)

            #Avg run over seeds, then truncate at convergence point
            acc = compute_means(testdata[0], scale=100)
            accstd = compute_stds(testdata[0], scale=100)
            idx = torch.argmax(acc)
            idx = min(idx + 2, 40)
            x = torch.linspace(0, idx/2, idx)
            acc = acc[0:idx]
            accstd = accstd[0:idx]

            best_acc1 = compute_max_means_std(testdata[0], scale=100)
            best_acc5 = compute_max_means_std(testdata[2], scale=100)
            print(f'{labels[t]} Conv', f'Test Acc(Top1) Best(Mean):{round(best_acc1[0], 2)}',
                  f'  (STD):{round(best_acc1[1], 2)}',
                  f'   Test Acc(Top5) Best(Mean):{round(best_acc5[0], 2)}',
                  f'  (STD):{round(best_acc5[1], 2)}')

            axs[2].plot(x, 100 - acc, label=labels[t], linewidth=2.25, color=clr[t], alpha=.7)
            axs[2].fill_between(x, (100 - acc) - accstd, (100 - acc) + accstd, color=clr[t], alpha=.2)

        axs[0].legend(ncol=2)
        axs[1].set(xlabel='Epoch')
        axs[0].set(ylim=[10, 50])
        axs[1].set(ylim=[30, 60])
        #axs[2].set(ylim=[39, 60])
        axs[0].set(ylabel='Test Error (%)')
        axs[0].set(title='SVHN')
        axs[1].set(title='CIFAR-10')
        axs[2].set(title='Tiny ImageNet')
        fig.suptitle('Convolutional Networks', fontsize=12)
        # pylab.tight_layout()
        pylab.show()






##########################################################################
def plot_MLP():
    with torch.no_grad():
        dlabel = ['SVHN', 'CIFAR-10']
        labels = ['BP', 'BP-Adam', 'SeqIL', 'SeqIL-MQ', 'SeqIL-Adam']
        clr = ['black', 'blue', 'purple', 'red', 'brown']
        epochs = [70, 110]
        fig, axs = pylab.subplots(1, 4, figsize=(11, 2.5), constrained_layout=True)

        for d in range(2):
            print(f'\n{dlabel[d]}')
            for t in range(5):
                with open(f'data/MLP_Type{t}_data{d}_epochs{epochs[d]}_64.data', 'rb') as filehandle:
                    testdata = pickle.load(filehandle)

                best_acc = compute_max_means_std(testdata[0], scale=100)
                acc = compute_means(testdata[0], scale=100)[0:60]
                accstd = compute_stds(testdata[0], scale=100)[0:60]
                x = torch.linspace(0, 60, acc.size(0))

                print(f'{labels[t]}', f'Test Acc Best(Mean):{round(best_acc[0], 2)}',
                      f'  (STD):{round(best_acc[1], 2)}',
                      f'  (Iter#):{round(best_acc[2], 2)*500}',
                      f'  LR:{testdata[2]}')

                axs[d].plot(x, 100 - acc, label=labels[t], linewidth=2.25, color=clr[t], alpha=.7)
                axs[d].fill_between(x, (100 - acc) - accstd, (100 - acc) + accstd, color=clr[t], alpha=.2)


        for d in range(2):
            print(f'\n{dlabel[d]}')
            for t in range(5):
                with open(f'data/AE_Type{t}_data{d}_epochs100.data','rb') as filehandle:
                    testdata = pickle.load(filehandle)
                best_loss = compute_min_means_std(testdata[0], scale=1)

                loss = compute_means(testdata[0], scale=1)
                lossstd = compute_stds(testdata[0], scale=1)
                idx = torch.argmin(loss)
                idx = min(idx + 1, 100)
                x = torch.linspace(0, idx, idx)
                loss = loss[0:idx]
                lossstd = lossstd[0:idx]

                print(f'{labels[t]} AE', f'Test Loss Best(Mean):{round(best_loss[0], 4)}',
                                      f'  (STD):{round(best_loss[1], 4)}',
                                      f'  (Epoch#):{round(best_loss[2], 4)}')

                axs[d+2].plot(x, loss, label=labels[t], linewidth=2.25, color=clr[t], alpha=.7)
                axs[d+2].fill_between(x, loss - lossstd, loss + lossstd, color=clr[t], alpha=.2)


        axs[0].legend(ncol=1)
        axs[0].set(xlabel='Epoch')
        axs[1].set(xlabel='Epoch')
        axs[2].set(xlabel='Epoch')
        axs[3].set(xlabel='Epoch')
        axs[0].set(ylim=[15, 80])
        axs[1].set(ylim=[43, 70])
        axs[2].set(ylim=[.588, .65])
        axs[3].set(ylim=[.588, .65])
        axs[0].set(ylabel='Test Error (%)')
        #axs[1].set(ylabel='Test Error (%)')
        axs[2].set(ylabel='Test Reconstruction\nLoss (BCE)')
        #axs[3].set(ylabel='Test Reconstruct. Loss (BCE)')
        axs[0].set(title='SVHN (MLP)')
        axs[1].set(title='CIFAR-10 (MLP)')
        axs[2].set(title='SVHN (AE)')
        axs[3].set(title='CIFAR-10 (AE)')
        fig.suptitle('Fully Connected Networks', fontsize=12)
        #pylab.tight_layout()
        pylab.show()






##########################################################################
def plot_speed_analysis():
    with torch.no_grad():
        labels = ['BP', 'BP-Adam', 'IL', 'IL-MQ', 'IL-Adam', 'IL(i=15)', 'IL-MQ(i=15)', 'IL-Adam(i=15)']
        clr = ['black', 'blue', 'purple', 'red', 'brown', 'purple', 'red', 'brown']
        ls = ['-', '-', '-', '-', '-', '--', '--', '--']

        fig, axs = pylab.subplots(1, 3, figsize=(8, 2.5), constrained_layout=True)

        #Plot Fully Connected
        for t in [0,1,3]:
            with open(f'data/MLP_Type{t}_data1_epochs110_64.data', 'rb') as filehandle:
                testdata = pickle.load(filehandle)

            with open(f'data/Time_Type{t}_data1_epochs110.data', 'rb') as filehandle:
                time = pickle.load(filehandle)

            acc = compute_means(testdata[0], scale=100)
            accstd = compute_stds(testdata[0], scale=100)
            idx = torch.argmax(acc)
            idx = min(idx + 2, 110)
            timeConv = (idx / 110) * time
            x = torch.linspace(0, timeConv, idx)
            acc = acc[0:idx]
            accstd = accstd[0:idx]

            print(f'MLP Type:{t}, Time:{time}, TimeConv:{timeConv}')
            axs[0].plot(x, 100 - acc, label=labels[t], linewidth=3, color=clr[t], alpha=.7, linestyle=ls[t])
            axs[0].fill_between(x, (100 - acc) - accstd, (100 - acc) + accstd, color=clr[t], alpha=.2)


        #Plot Conv
        for t in [0,1,3]:
            with open(f'data/Conv_Type{t}_data1_epochs45_64.data', 'rb') as filehandle:
                testdata = pickle.load(filehandle)

            with open(f'data/TimeConv_Type{t}_data1_epochs45.data', 'rb') as filehandle:
                time = pickle.load(filehandle)

            acc = compute_means(testdata[0], scale=100)
            accstd = compute_stds(testdata[0], scale=100)
            idx = torch.argmax(acc)
            idx = min(idx + 2, 45)
            timeConv = (idx / 45) * time
            x = torch.linspace(0, timeConv, idx)
            acc = acc[0:idx]
            accstd = accstd[0:idx]

            print(f'Conv Type:{t}, Time:{time}, TimeConv:{timeConv}')
            axs[1].plot(x, 100 - acc, label=labels[t], linewidth=3, color=clr[t], alpha=.7, linestyle=ls[t])
            axs[1].fill_between(x, (100 - acc) - accstd, (100 - acc) + accstd, color=clr[t], alpha=.2)



        # Plot AE
        for t in [0, 1, 3]:
            with open(f'data/AE_Type{t}_data1_epochs100.data', 'rb') as filehandle:
                testdata = pickle.load(filehandle)

            with open(f'data/TimeAE_Type{t}_data1_epochs100.data', 'rb') as filehandle:
                time = pickle.load(filehandle)

            loss = compute_means(testdata[0], scale=1)
            lossstd = compute_stds(testdata[0], scale=1)
            idx = torch.argmin(loss)
            idx = min(idx + 2, 100)
            timeConv = (idx / 100) * time
            x = torch.linspace(0, timeConv, idx)
            loss = loss[0:idx]
            lossstd = lossstd[0:idx]

            print(f'AE Type:{t}, Time:{time}, TimeConv:{timeConv}')
            axs[2].plot(x, loss, label=labels[t], linewidth=3, color=clr[t], alpha=.7, linestyle=ls[t])
            axs[2].fill_between(x, loss - lossstd, loss + lossstd, color=clr[t], alpha=.2)


        axs[0].legend(ncol=1)
        axs[0].set(xlabel='Time (sec)')
        axs[1].set(xlabel='Time (sec)')
        axs[2].set(xlabel='Time (sec)')
        axs[0].set(ylim=[41, 70])
        axs[1].set(ylim=[30, 70])
        axs[2].set(ylim=[.587, .66])
        axs[0].set(ylabel='Test Error (%)')
        axs[1].set(ylabel='Test Error (%)')
        axs[2].set(ylabel='Reconstruction\nLoss (BCE)')
        axs[0].set(title='CIFAR-10 (FC)')
        axs[1].set(title='CIFAR-10 (Conv)')
        axs[2].set(title='CIFAR-10 (Autoencoder)')
        fig.suptitle('Real Time Comparison', fontsize=12)
        #pylab.tight_layout()
        pylab.show()




def plot_err_analyze():
    with torch.no_grad():
        fig, axs = pylab.subplots(1, 2, figsize=(8, 2.5), constrained_layout=True, sharey=True)
        axs[0].set(yscale='log')
        axs[1].set(yscale='log')

        with open(f'data/errAnalyze_trainFalse.data', 'rb') as filehandle:
            data = pickle.load(filehandle)
            x_axis = torch.linspace(1, 8, steps=8)
            for l in range(5):
                axs[0].errorbar(x_axis, data[0][l], yerr=data[1][l], linewidth=2.25, alpha=.7, label=f'l{l + 1}', marker='.', color=[1-.25*l, 0, l/4])
                axs[1].errorbar(x_axis, data[2][l], yerr=data[3][l], linewidth=2.25, alpha=.7, label=f'l{l + 1}', marker='.', color=[1-.25*l, 0, l/4])


        axs[0].legend(ncol=1, loc='lower left')
        axs[0].set(xlabel='Inference Iteration')
        axs[1].set(xlabel='Inference Iteration')
        axs[0].set(title='Standard/Simultaneous Inference')
        axs[1].set(title=r'Sequential Inference')
        axs[0].set(ylabel='Local Error (MSE)')
        axs[0].set(ylim=[1e-30, 1])
        #axs[1].set(ylim=[30, 60])
        #pylab.tight_layout()
        pylab.show()



def plot_wt_anlz(batch_size=64):
    with torch.no_grad():
        labels = ['IL', 'IL-MQ', 'IL-Adam']
        types = [0, 1, 2]
        colors = ['Orange', 'Cyan', 'Green']

        fig, axs = pylab.subplots(1, 1, figsize=(3, 2.5))
        axs.set(yscale='log')

        #Plot wt magnitudes
        for m in range(3):
            with open(f'data/wtAnalyze_type{types[m]}_data2_max_iter10000_btchsz{batch_size}.data','rb') as filehandle:
                datalw = pickle.load(filehandle)
            axs.plot([0,1,2,3], datalw[0], linewidth=3.25, alpha=.6,
                         label=f'{labels[m]}', marker='.', markersize=9, color=colors[m])


        axs.legend(ncol=1)
        axs.set(xlabel=r'$W_l$')
        axs.set(ylabel=r'$\Delta W_l$ Magnitude $(\frac{1}{ij} \vert \Delta W_l \vert)$')
        pylab.tight_layout()
        pylab.show()



################################################################################
def plot_wt_anlz_conv(batch_size=64):
    with torch.no_grad():
        labels = ['SeqIL', 'SeqIL-MQ', 'SeqIL-Adam']
        types = [3, 4, 5]
        colors = ['Purple', 'Red', 'Brown']

        fig, axs = pylab.subplots(1, 1, figsize=(3, 2.5))
        axs.set(yscale='log')

        #Plot wt magnitudes
        for m in range(3):
            with open(f'data/wtAnalyze_type{types[m]}_data2_max_iter10000_btchsz{batch_size}.data','rb') as filehandle:
                datalw = pickle.load(filehandle)
            axs.errorbar([0,1,2,3], datalw[0], yerr=datalw[1], linewidth=3.25, alpha=.6,
                         label=f'{labels[m]}', marker='.', markersize=9, color=colors[m])

        axs.legend(ncol=1, loc='lower right')
        axs.set(title='Convolutional Update Magnitudes')
        axs.set(xlabel=r'$W_l$')
        axs.set(ylabel=r'$\Delta W_l$ Magnitude $(\frac{1}{IJ} \vert \Delta W_l \vert)$')
        pylab.tight_layout()
        pylab.show()



def plot_T_anlz():
    with torch.no_grad():
        labels = ['SeqIL-Adam', 'IL-Adam']
        colors = ['Brown', 'Green']
        y = [[], []]
        std = [[], []]
        mark = ['^', 's']

        fig, axs = pylab.subplots(1, 1, figsize=(3, 2.5))

        #Plot
        for m in range(2):
            for n_iter in range(1, 6):
                with open(f'data/Infer_compare_type{m}_data1_{n_iter}.data','rb') as filehandle:
                    data = pickle.load(filehandle)

                mx = compute_max_means_std(data[0], scale=100)
                y[m].append(mx[0])
                std[m].append(mx[1])

            print(f'Model:{m}    Acc:{y[m]}   Std:{std[m]}')
            axs.errorbar([1,2,3,4,5], y[m], yerr=std[m], linewidth=3.25, alpha=.6,
                         label=f'{labels[m]}', marker=mark[m], markersize=9, color=colors[m])


        axs.legend(ncol=1)
        axs.set(xlabel='# Iterations During Inference Phase (T)')
        axs.set(ylabel='Test Accuracy (%)')
        pylab.tight_layout()
        pylab.show()




##########################################################################
def plot_prox_Analyze(NLL=True):
    with torch.no_grad():
        opt = ['', '_MQ']
        n_iters = [5, 20]
        beta = [.01, .1, 1, 10, 100]
        fig, axs = pylab.subplots(1, 4, figsize=(12, 3), constrained_layout=True)

        l_type = ''
        if NLL:
            l_type = '_NLL'

        for o in range(2):
            for m in range(2):
                for b in range(5):
                    with open(f'data/proxAnalyze_data2_test_iter1000_niter{n_iters[m]}_mType{m}{opt[o]}{l_type}.data','rb') as filehandle:
                        dta = pickle.load(filehandle)

                    x = torch.linspace(0, dta[0][:,b].size(0)-1, dta[0][:,b].size(0))
                    shift = dta[0][0,b] - 2
                    axs[o*2 + abs(m-1)].plot(x, dta[0][:,b] - shift, linewidth=3.25, alpha=.7, label=r'$\beta$='+f'{beta[b]}', markersize=9, color=[0, 1-.15*b, .15*b])


        axs[1].legend(ncol=2)
        for x in range(4):
            axs[x].set(xlabel='Inference Iteration (t)')
        axs[0].set(ylim=[0, 2.14])
        axs[1].set(ylim=[0, 2.14])
        axs[2].set(ylim=[0, 2.14])
        axs[3].set(ylim=[0, 2.14])
        axs[0].set(ylabel=r'$\mathcal{L}(\theta) + \frac{1}{2 \beta} \Vert \theta - \theta^b \Vert^2$')
        axs[0].set(title='IL')
        axs[1].set(title='SeqIL')
        axs[2].set(title='IL-MQ')
        axs[3].set(title='SeqIL-MQ')
        #axs[1].set(title='SimIL (Train Weights)')
        #fig.suptitle('Convolutional Networks', fontsize=12)
        # pylab.tight_layout()
        pylab.show()




##########################################################################
def plot_prox():
    with torch.no_grad():
        opt = ['', '', '_MQ', '_MQ']
        labels = ['IL', 'SeqIL', 'IL-MQ', 'SeqIL-MQ']
        n_iters = [20, 5, 20, 5]
        colors = ['orange', 'purple', 'cyan', 'red']
        types = [1,0,1,0]
        betas = [4,4,4,4]
        fig, axs = pylab.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

        for m in range(4):
            with open(f'data/proxAnalyze_data2_test_iter1000_niter{n_iters[m]}_mType{types[m]}{opt[m]}_NLL.data','rb') as filehandle:
                dta = pickle.load(filehandle)

            x = torch.linspace(0, dta[0][:, betas[m]].size(0) - 1, dta[0][:, betas[m]].size(0))
            shift = dta[0][0, betas[m]] - 2.2
            axs.plot(x, dta[0][:,betas[m]], linewidth=2.5, alpha=.7, label=labels[m], markersize=9, color=colors[m])
            axs.fill_between(x, dta[0][:,betas[m]] - dta[1][:,betas[m]], dta[0][:,betas[m]] + dta[1][:,betas[m]], color=colors[m], alpha=.2)


        axs.legend(ncol=1)
        #axs.set(ylim=(0, 2.5))
        axs.set(ylabel=r'$\mathcal{L}(\theta) + \frac{1}{2 \beta} \Vert \theta - \theta^b \Vert^2$')
        axs.set(xlabel='Inference Iterations')
        #axs[1].set(title='SimIL (Train Weights)')
        #fig.suptitle('Convolutional Networks', fontsize=12)
        # pylab.tight_layout()
        pylab.show()


##########################################################################
def plot_T_Analyze_training():
    with torch.no_grad():
        colors = ['Brown', 'Green']
        fig, axs = pylab.subplots(1, 2, figsize=(7, 2), constrained_layout=True)

        for t in range(2):
            for n_iter in range(1,6):
                with open(f'data/Infer_compare_type{t}_data1_{n_iter}.data','rb') as filehandle:
                    testdata = pickle.load(filehandle)
                acc = compute_means(testdata[0], scale=100)
                x = torch.linspace(0, 45, acc.size(0))
                axs[t].plot(x, 100 - acc, linewidth=2.25, color=colors[t], alpha=0 + n_iter*.15, label=f'T={n_iter}')

        axs[0].legend()
        axs[1].legend()
        axs[0].set(xlabel='Epoch')
        axs[1].set(xlabel='Epoch')
        axs[0].set(ylim=[40, 90])
        axs[1].set(ylim=[40, 90])
        axs[0].set(ylabel='Test Error (%)')
        axs[0].set(title='SeqIL-Adam')
        axs[1].set(title='IL-Adam')
        #fig.suptitle('Convolutional Networks', fontsize=12)
        # pylab.tight_layout()
        pylab.show()


#plot_conv()
#plot_MLP()
#plot_T_Analyze_training()
#plot_wt_anlz()
#plot_wt_anlz_conv()
#plot_speed_analysis()
#plot_T_anlz()
#plot_prox_Analyze()
plot_prox()