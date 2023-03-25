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
    print('Number of classes per client {}'.format(num_classes_per_client))

    buckets = []
    for k in range(num_classes):
        temp = np.array(k * int(num_examples / num_classes) + np.random.permutation(int(num_examples / num_classes)))
        buckets = np.hstack((buckets, temp))

    shards = num_classes_per_client * num_clients
    perm = np.random.permutation(shards)

    # client_set will be of length num_examples/N and each element represents a client.
    client_set = []
    extra = len(buckets) % shards
    if extra:
        buckets = buckets[:-extra]
    ind_list = np.split(buckets, shards)
    print('ind_list.len:', len(ind_list))

    for j in range(0, shards, num_classes_per_client):
        # each entry of z is associated to two shards. the two shards are sampled randomly by using the permutation matrix
        # perm and stacking two shards together using vstack. Each client now holds 2500*2 datapoints.
        temp = []
        for k in range(num_classes_per_client):
            temp = np.hstack((temp, ind_list[int(perm[j+k])]))
        client_set.append(temp)
        perm_2 = np.random.permutation(len(temp))
        client_set[-1] = client_set[-1][perm_2]

    return client_set


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
