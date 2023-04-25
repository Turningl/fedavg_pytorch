# -*- coding: utf-8 -*-
# @Email : zl16035056@163.com
# @File : client.py
import copy

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine


class Client(nn.Module):
    def __init__(self, x_train, y_train, dataset, batch_size, dp, epoch, sigma, grad_norm, fedprox, mu, device):
        super(Client, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.dataset_size = dataset
        self.batch_size = batch_size
        self.epoch = epoch
        self.dp = dp
        self.sigma = sigma
        self.grad_norm = grad_norm
        self.device = device
        self.mu = mu
        self.fedprox = fedprox
        self.model = None

    def download(self, model):
        if self.device:
            self.model = model.to(self.device)
        else:
            self.model = model

    def local_update(self):
        model = self.model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        global_model = copy.deepcopy(model)

        x_batch = self.x_train[self.dataset_size]
        y_batch = self.y_train[self.dataset_size]

        data_batch = TensorDataset(x_batch, y_batch)
        data_loader = DataLoader(data_batch, batch_size=self.batch_size, shuffle=True)

        if self.dp:
            privacy_engine = PrivacyEngine(secure_mode=False)
            model, optimizer, train_loader = privacy_engine.make_private(module=model,
                                                                         optimizer=optimizer,
                                                                         data_loader=data_loader,
                                                                         noise_multiplier=self.sigma,
                                                                         max_grad_norm=self.grad_norm)

        # train
        for epoch in range(self.epoch):
            train_acc = 0
            train_loss = 0
            for x_train, y_train in data_loader:
                x_train, y_train = x_train.to(self.device), y_train.to(self.device)

                y_pred = model(x_train)
                _, test_pred = torch.max(y_pred, 1)
                correct = (test_pred == y_train).sum()

                loss = criterion(y_pred, y_train)

                # fedprox, add proximal term
                if self.fedprox:
                    proximal_term = 0
                    for w, w_global in zip(model.parameters(), global_model.parameters()):
                        proximal_term += self.mu / 2 * torch.norm(w - w_global, 2)
                    loss += proximal_term

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_acc += correct.item()
                train_loss += loss.item()

            print('Epoch is: %d, Train acc: %.4f, Train loss: %.4f' % ((epoch + 1), train_acc / len(self.dataset_size), train_loss / len(self.dataset_size)))

        return [weight.data for weight in model.state_dict().values()]
