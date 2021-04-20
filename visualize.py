import torch
import matplotlib
import matplotlib.pyplot as plt
from configs import HYPER_PARAMETERS as hp

dataset_name = 'FashionMNIST'


class Animator:
    def __init__(self, x_axis, y_axes, x_label=None, y_label=None, x_lim=None, y_lim=None, legends=None, fmts=None):
        assert len(legends) == len(fmts)
        self.x_label = x_label
        self.y_label = y_label
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.legends = legends
        self.fmts = fmts
        self.x_axis = x_axis
        self.y_axes = [[] for _ in range(len(y_axes))]
        for i, y_axis in enumerate(y_axes):
            self.y_axes[i] = y_axis

    def display(self, fname):
        params = {
            'font.family': 'serif',
            'font.serif': 'Times New Roman',
            'font.size': 15
        }
        matplotlib.rcParams.update(params)
        plt.figure(figsize=(6, 5.3))
        for y_axis, legend, fmt in zip(self.y_axes, self.legends, self.fmts):
            plt.plot(self.x_axis, y_axis, fmt, label=legend)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.legend(loc='best')
        plt.ylim((0.4, 0.9))
        if dataset_name == 'CIFAR10':
            plt.plot((10, 2000), (.8, .8), 'm--')
        elif dataset_name == 'FashionMNIST':
            plt.plot((2, 200), (.8, .8), 'm--')
        plt.savefig(fname)


# experiment 1
x = list(range(10, 2001, 10))
result1 = torch.load('ijcai_results/CIFAR10_DP_SFed_sampling_pr_0.03_participating_ratio_0.5_edge_epochs_10_compression_ratio_0.3_sigma_2_NM.pt')['accuracy']
result4 = torch.load('ijcai_results/CIFAR10_DP_SFed_sampling_pr_0.03_participating_ratio_0.5_edge_epochs_10_compression_ratio_0.3_sigma_2.pt')['accuracy']
fedavg_result_1 = torch.load('ijcai_results/CIFAR10_FedAVG_participating_ratio_0.5.pt')['accuracy']
legend1 = 'H-FL without Bias Corrector'
legend2 = 'H-FL with Bias Corrector'
fedavg_legend_1 = 'FedAVG'
results = [result1, result4, fedavg_result_1]
legends = [legend1, legend2, fedavg_legend_1]
# fmts = ['r-', 'b-', 'g-', 'k-', 'y-', 'm-']
fmts = ['r-', 'b-', 'g--']

# animator = Animator(x, ijcai_results, x_label='communication_rounds', y_label='TestAcc', legends=legends, fmts=fmts)
# animator.display('ijcai_plots/Bias Corrector.png')

# experiment 2
result1 = torch.load('ijcai_results/CIFAR10_DP_SFed_sampling_pr_0.03_participating_ratio_0.5_edge_epochs_10_compression_ratio_0.3_sigma_2.pt')['accuracy']
result2 = torch.load('ijcai_results/CIFAR10_DP_SFed_sampling_pr_0.03_participating_ratio_0.3_edge_epochs_10_compression_ratio_0.3_sigma_2.pt')['accuracy']
result3 = torch.load('ijcai_results/CIFAR10_DP_SFed_sampling_pr_0.03_participating_ratio_0.2_edge_epochs_10_compression_ratio_0.3_sigma_2.pt')['accuracy']
result4 = torch.load('ijcai_results/CIFAR10_DP_SFed_sampling_pr_0.02_participating_ratio_0.5_edge_epochs_10_compression_ratio_0.3_sigma_2.pt')['accuracy']
result5 = torch.load('ijcai_results/CIFAR10_DP_SFed_sampling_pr_0.01_participating_ratio_0.5_edge_epochs_10_compression_ratio_0.3_sigma_2.pt')['accuracy']
result6 = torch.load('ijcai_results/CIFAR10_DP_SFed_sampling_pr_0.03_participating_ratio_0.5_edge_epochs_10_compression_ratio_0.2_sigma_2.pt')['accuracy']
result8 = torch.load('ijcai_results/CIFAR10_DP_SFed_sampling_pr_0.03_participating_ratio_0.5_edge_epochs_10_compression_ratio_0.3_sigma_10.pt')['accuracy']
result9 = torch.load('ijcai_results/CIFAR10_DP_SFed_sampling_pr_0.03_participating_ratio_0.5_edge_epochs_10_compression_ratio_0.3_sigma_20.pt')['accuracy']
legend1 = 'P=0.5, S=0.03, C=0.3, sigma=2'
legend2 = 'P=0.3, S=0.03, C=0.3, sigma=2'
legend3 = 'P=0.2, S=0.03, C=0.3, sigma=2'
legend4 = 'P=0.5, S=0.02, C=0.3, sigma=2'
legend5 = 'P=0.5, S=0.01, C=0.3, sigma=2'
legend6 = 'P=0.5, S=0.03, C=0.2, sigma=2'
legend8 = 'P=0.5, S=0.03, C=0.3, sigma=10'
legend9 = 'P=0.5, S=0.03, C=0.3, sigma=20'

results1 = [result1, result2, result3]
legends1 = [legend1, legend2, legend3]
fmts1 = ['b-', 'r-', 'g-']

results2 = [result1, result4, result5]
legends2 = [legend1, legend4, legend5]
fmts2 = ['b-', 'k-', 'g-']

results3 = [result1, result6]
legends3 = [legend1, legend6]
fmts3 = ['b-', 'm-']

results4 = [result1, result8, result9]
legends4 = [legend1, legend8, legend9]
fmts4 = ['b-', 'y-', 'g-']

# animator = Animator(x, results1, x_label='communication_rounds', y_label='TestAcc', legends=legends1, fmts=fmts1)
# animator.display('ijcai_plots/Participating Ratio.png')
#
# animator = Animator(x, results2, x_label='communication_rounds', y_label='TestAcc', legends=legends2, fmts=fmts2)
# animator.display('ijcai_plots/Sampling Probability.png')
#
# animator = Animator(x, results3, x_label='communication_rounds', y_label='TestAcc', legends=legends3, fmts=fmts3)
# animator.display('ijcai_plots/Compression Ratio.png')

# animator = Animator(x, results4, x_label='communication_rounds', y_label='TestAcc', legends=legends4, fmts=fmts4)
# animator.display('ijcai_plots/Noise Level.png')

# experiment 3
x = list(range(2, 201, 2))
result1 = torch.load('ijcai_results/FashionMNIST_DP_SFed_sampling_pr_0.03_participating_ratio_0.5_edge_epochs_10_compression_ratio_0.3_sigma_2.pt')['accuracy']
result2 = torch.load('ijcai_results/FashionMNIST_FedAVG_participating_ratio_0.5.pt')['accuracy']
result3 = torch.load('ijcai_results/FashionMNIST_STC_participating_ratio_0.5.pt')['accuracy']
result4 = torch.load('ijcai_results/FashionMNIST_DGC_participating_ratio_0.5.pt')['accuracy']
results = [result1, result2, result3, result4]
legends = ['H-FL', 'FedAVG', 'STC', 'DGC']
fmts1 = ['b-', 'r-', 'g-', 'k-']

animator = Animator(x, results, x_label='communication_rounds', y_label='TestAcc', legends=legends, fmts=fmts1)
animator.display('ijcai_plots/FMNIST Methods.png')

# experiment 4
x = list(range(10, 2001, 10))
result1 = torch.load('ijcai_results/CIFAR10_DP_SFed_sampling_pr_0.03_participating_ratio_0.5_edge_epochs_10_compression_ratio_0.3_sigma_2.pt')['accuracy']
result2 = torch.load('ijcai_results/CIFAR10_FedAVG_participating_ratio_0.5.pt')['accuracy']
result3 = torch.load('ijcai_results/CIFAR10_STC_participating_ratio_0.5.pt')['accuracy']
result4 = torch.load('ijcai_results/CIFAR10_DGC_participating_ratio_0.5.pt')['accuracy']
results = [result1, result2, result3, result4]
legends = ['H-FL', 'FedAVG', 'STC', 'DGC']
fmts1 = ['b-', 'r-', 'g-', 'k-']
# animator = Animator(x, ijcai_results, x_label='communication_rounds', y_label='TestAcc', legends=legends, fmts=fmts1)
# animator.display('ijcai_plots/CIFAR10 Methods.png')

# eperiment 5
'''
Dataset    H-FL    FedAVG    DGC    STC
CIFAR10    0.25    2.71      0.13     0.06
FMNIST    16.97     -        16.67     12.24
'''


def plot_fmnist_overhead():
    params = {
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 15
    }
    matplotlib.rcParams.update(params)
    plt.figure(figsize=(6, 5.3))
    methods_list = ['H-FL', 'FedAVG', 'DGC', 'STC']
    overhead_list = [0.25, 2.71, 0.13, 0.06]
    plt.bar(range(len(methods_list)), overhead_list, width=.5, color=['b', 'g', 'r', 'y'], tick_label=methods_list)
    plt.xlabel('Different methods')
    plt.ylabel('Communication Overhead (GB)')
    plt.savefig('ijcai_plots/FMNST Overhead.png')


def plot_cifar10_overhead():
    params = {
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 15
    }
    matplotlib.rcParams.update(params)
    plt.figure(figsize=(6, 5.3))
    methods_list = ['H-FL', 'FedAVG', 'DGC', 'STC']
    overhead_list = [16.97, 0, 16.67, 12.24]
    plt.bar(range(len(methods_list)), overhead_list, width=.5, color=['b', 'w', 'r', 'y'], tick_label=methods_list)
    plt.xlabel('Different methods')
    plt.ylabel('Communication Overhead (GB)')
    plt.savefig('ijcai_plots/CIFAR10 Overhead.png')


plot_fmnist_overhead()
plot_cifar10_overhead()