# -*- coding: utf-8 -*-
# @Author : Zhang
# @Email : zl16035056@163.com
# @File : dataloader.py


import torch
from torchvision import datasets, transforms


def loader(name):
    print('Using MNIST dataset!\n')
    if name == 'MNIST':
        train_dataloader = datasets.MNIST(root='~/data', train=True, download=False, transform=transforms.ToTensor())
        x_train = train_dataloader.data.float().unsqueeze(1)
        y_train = train_dataloader.targets

        indices_train = torch.argsort(y_train)
        sorted_x_train = x_train[indices_train]
        sorted_y_train = y_train[indices_train]

        test_dataloader = datasets.MNIST(root='~/data', train=False, download=False, transform=transforms.ToTensor())
        x_test = test_dataloader.data.float().unsqueeze(1)
        y_test = test_dataloader.targets

        return sorted_x_train, sorted_y_train, x_test, y_test