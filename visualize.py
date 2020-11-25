import torch
import matplotlib.pyplot as plt


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
        plt.figure(figsize=(8, 6))
        for y_axis, legend, fmt in zip(self.y_axes, self.legends, self.fmts):
            plt.plot(self.x_axis, y_axis, fmt, label=legend)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.legend(loc='best')
        plt.savefig(fname)


x = list(range(20, 2001, 20))
result1 = torch.load('results/CIFAR10_DP_SFed_sampling_pr_0.02_participating_ratio_0.3_edge_epochs_20.pt')['accuracy']
result2 = torch.load('results/CIFAR10_DP_SFed_sampling_pr_0.02_participating_ratio_0.5_edge_epochs_20.pt')['accuracy']
result3 = torch.load('results/CIFAR10_DP_SFed_sampling_pr_0.02_participating_ratio_0.7_edge_epochs_20.pt')['accuracy']
fedavg_result_1 = torch.load('results/CIFAR10_FedAVG_participating_ratio_0.3.pt')['accuracy']
fedavg_result_2 = torch.load('results/CIFAR10_FedAVG_participating_ratio_0.5.pt')['accuracy']
fedavg_result_3 = torch.load('results/CIFAR10_FedAVG_participating_ratio_0.7.pt')['accuracy']
legend1 = 'DP_SFed_sampling_pr_0.02_participating_ratio_0.3_edge_epochs_20'
legend2 = 'DP_SFed_sampling_pr_0.02_participating_ratio_0.5_edge_epochs_20'
legend3 = 'DP_SFed_sampling_pr_0.02_participating_ratio_0.7_edge_epochs_20'
fedavg_legend_1 = 'FedAVG_participating_ratio_0.3.pt'
fedavg_legend_2 = 'FedAVG_participating_ratio_0.5.pt'
fedavg_legend_3 = 'FedAVG_participating_ratio_0.7.pt'
results = [result1, result2, result3, fedavg_result_1, fedavg_result_2, fedavg_result_3]
legends = [legend1, legend2, legend3, fedavg_legend_1, fedavg_legend_2, fedavg_legend_3]
fmts = ['r-', 'b-', 'g-', 'k-', 'y-', 'm-']

animator = Animator(x, results, x_label='communication_rounds', y_label='TestAcc', legends=legends, fmts=fmts)
animator.display('plots/participating_ratio.png')
