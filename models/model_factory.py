"""
model loader
"""

from forMNIST.cnn import CNN as minist_cnn
from forFashionMNIST.cnn import CNN as fashion_mnist_cnn
from forCIFAR10.vgg16 import VGG16 as cifar_vgg16


def model_factory(dataset_name, model_name_string):
    model = None
    if dataset_name == "MNIST":
        if model_name_string == "CNN":
            model = minist_cnn()
    elif dataset_name == "FashionMNIST":
        if model_name_string == "CNN":
            model = fashion_mnist_cnn()
    elif dataset_name == "CIFAR10":
        if model_name_string == "VGG16":
            model = cifar_vgg16()
    else:
        raise ValueError('No such kind of model!')
    return model
