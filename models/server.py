# -*- coding: utf-8 -*-
# @Email : zl16035056@163.com
# @File : server.py


import numpy as np
import torch
import torch.nn as nn
from models.cnn import CNN


class Server(nn.Module):
    def __init__(self, num_clients, sample_ratio, device):
        super(Server, self).__init__()
        self.num_clients = num_clients
        self.sample_ratio = sample_ratio
        self.device = device
        self.model_state_dict = []
        self.num_vars = 0
        self.shape_var = 0
        self.model = CNN(input_dim=1, output_dim=10)
        self.state_dict_key = self.model.state_dict().keys()

    def init_global_model(self):
        return self.model

    def sample_clients(self, candidates):
        m = int(self.num_clients * self.sample_ratio)

        if candidates < m:
            return []

        else:
            participants = list(np.random.permutation(candidates))[0:m]
            return participants

    def add_weights(self, num_vars, model_dict, agg_model_dict):
        # for i in range(num_vars):
        #     if not len(agg_model_dict):
        #         a = np.expand_dims(model_dict[i], 0)
        #     else:
        #         b = np.append(agg_model_dict[i], np.expand_dims(model_dict[i], 0), 0)
        return [np.expand_dims(model_dict[i], 0) if not len(agg_model_dict) else
                np.append(agg_model_dict[i], np.expand_dims(model_dict[i], 0), 0) for i in range(num_vars)]

    # fedavg
    def aggregate(self, model_state_dict):
        self.num_vars = len(model_state_dict)
        self.shape_var = [var.shape for var in model_state_dict]

        update_model_dict = [state.cpu().numpy().flatten() for state in model_state_dict]

        self.model_state_dict = self.add_weights(self.num_vars, update_model_dict, self.model_state_dict)

    def update(self):
        mean_updates = [torch.from_numpy(np.average(self.model_state_dict[i], 0).reshape(self.shape_var[i])) for i in range(self.num_vars)]
        mean_updates = dict(zip(self.state_dict_key, mean_updates))

        self.model_state_dict = []

        self.model.load_state_dict(mean_updates)
        return self.model
