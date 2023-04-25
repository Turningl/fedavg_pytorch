# -*- coding: utf-8 -*-
# @Email : zl16035056@163.com
# @File : create_dataset.py

import numpy as np
import math


def create_iid_clients(num_clients, num_examples, num_classes, num_examples_per_client, num_classes_per_client):
    client_set = []

    rounds = math.ceil(num_clients * num_examples_per_client / num_examples)
    client_per_round = int(num_examples / num_examples_per_client)
    client_rount = 0

    for i in range(rounds):
        perm = np.random.permutation(num_examples)
        for j in range(client_per_round):
            if client_rount == num_clients:
                break
            client_rount += 1
            client_set.append(np.array(perm[j * num_examples_per_client: (j+1) * num_examples_per_client]))

    return client_set


def create_noniid_clients(num_clients, num_examples, num_classes, num_examples_per_client, num_classes_per_client):
    num_classes = 10

    buckets = []
    for k in range(num_classes):
        temp = []
        for j in range(int(num_clients / 10)):
            temp = np.hstack((temp, k * int(num_examples/10) + np.random.permutation(int(num_examples/10))))
        # print('temp.len: ', len(temp))
        buckets = np.hstack((buckets, temp))
        # print('buckets.len: ', len(buckets))

    shards = 2 * num_clients # 20
    # ('buckets.shape:', buckets.shape, 'shards', shards) # buckets.shape: (10, 5000*(N/10))
    perm = np.random.permutation(shards)

    # z will be of length N*5000 and each element represents a client.
    z = []
    ind_list = np.split(buckets, shards) # 50000/20 = 2500
    # print('ind_list.len:', len(ind_list))
    for j in range(0, shards, 2):
        # each entry of z is associated to two shards. the two shards are sampled randomly by using the permutation matrix
        # perm and stacking two shards together using vstack. Each client now holds 2500*2 datapoints.
        z.append(np.hstack((ind_list[int(perm[j])], ind_list[int(perm[j + 1])])))
        # shuffle the data in each element of z, so that each client doesn't have all digits stuck together.
        perm_2 = np.random.permutation(int(2 * len(buckets) / shards))
        z[-1] = z[-1][perm_2]

    return z


def check_labels(N, client_set, y_train):
    labels_set = []
    for cid in range(N):
        idx = [int(val) for val in client_set[cid]]
        labels_set.append(set(np.array(y_train)[idx]))

        labels_count = [0]*10
        for label in np.array(y_train)[idx]:
            labels_count[int(label)] += 1
        print('cid: {}, number of labels: {}/10.'.format(cid, len(labels_set[cid])))
        print(labels_count)
