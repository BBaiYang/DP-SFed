"""
model loader
"""

from forMNIST.cnn import CNN as minist_CNN
from forFashionMNIST.cnn import CNN
from forCIFAR10.vgg16 import VGG16
from forCIFAR10.resnet18 import ResNet18


def model_factory(dataset_name, model_name_string):
    model = None
    if dataset_name == "MNIST":
        if model_name_string == "CNN":
            model = minist_CNN()
    elif dataset_name == "FashionMNIST":
        if model_name_string == "CNN":
            model = CNN()
    elif dataset_name == "CIFAR10":
        if model_name_string == "VGG16":
            model = VGG16()
        elif dataset_name == 'ResNet18':
            model = ResNet18()
    else:
        raise ValueError('No such kind of model!')
    return model
