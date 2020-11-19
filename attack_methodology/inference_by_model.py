import torch
from configs import HYPER_PARAMETERS as hp
from configs import DATA_SET_NAME as data_set_name
from configs import MODEL_NAME as model_name
from models.model_factory import model_factory
import torchvision
from torchvision import transforms
from torch.utils import data
import torchvision.utils as vutils


def show_image(img_data):
    vutils.save_image(img_data, 'raw_input.png', normalize=True)


device = hp['device_for_baseline']

"""Model"""
model = model_factory(data_set_name, model_name).to(device)
partial_model = list(model.children())[0][:1]
"""Data"""
train_transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

cifar_train = torchvision.datasets.CIFAR10(root='~/CODES/Dataset',
                                                train=True, download=True, transform=train_transform)
"""Single example"""
batch_size = 1
train_loader = data.DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=4)
true_input, true_label = None, None
for X, y in train_loader:
    true_input, true_label = X, y
    break
"""Visualize"""
show_image(true_input.squeeze())
"""True output"""
true_output = partial_model(true_input.to(device))
