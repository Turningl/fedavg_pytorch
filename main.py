# -*- coding: utf-8 -*-
# @Email : zl16035056@163.com
# @File : main.py


import copy
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from utils.create_dataset import create_iid_clients, create_noniid_clients, check_labels
from models.client import Client
from models.server import Server


def test(model, x_val, y_val):
    model.eval().to(args.device)
    data_loader = TensorDataset(x_val, y_val)
    data_loader = DataLoader(data_loader, batch_size=128, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    test_acc = 0

    with torch.no_grad():
        for x_test, y_test in data_loader:
            x_test, y_test = x_test.to(args.device), y_test.to(args.device)

            output = model(x_test)
            loss = criterion(output, y_test)

            _, test_pred = torch.max(output, 1)

            correct = (test_pred == y_test).sum()

            test_acc += correct.item()
            test_loss += loss.item()

    return test_acc / len(y_val), test_loss / len(y_val)


train_dataloader = datasets.MNIST(root='~/data', train=True, download=False, transform=transforms.ToTensor())
x_train = train_dataloader.data.float().unsqueeze(1)
y_train = train_dataloader.targets

test_dataloader = datasets.MNIST(root='~/data', train=False, download=False, transform=transforms.ToTensor())
x_test = test_dataloader.data.float().unsqueeze(1)
y_test = test_dataloader.targets


def prepare_local_data(noniid, num_clients):
    if noniid:
        dataset = create_iid_clients(num_clients=num_clients,
                                     num_examples=len(x_train),
                                     num_classes=10,
                                     num_examples_per_client=len(x_train) // 10,
                                     num_classes_per_client=10)
    else:
        dataset = create_noniid_clients(num_clients=num_clients,
                                        num_examples=len(x_train),
                                        num_classes=10,
                                        num_examples_per_client=len(x_train) // 10,
                                        num_classes_per_client=10)

    check_labels(10, dataset, y_train)

    return dataset


def main(args):
    # prepare local dataset
    dataset = prepare_local_data(args.noniid, args.num_clients)

    # set server
    server = Server(num_clients=args.num_clients, device=args.device, sample_ratio=args.sample_ratio)
    server_model = server.init_global_model()

    # set clients
    clients = []
    for i in range(args.all_clients):
        client = Client(x_train=x_train,
                        y_train=y_train,
                        dataset=dataset[i],
                        batch_size=args.batch_size,
                        dp=args.dp,
                        epoch=args.epochs,
                        sigma=args.sigma,
                        grad_norm=args.grad_norm,
                        device=args.device,
                        fedprox=args.fedprox,
                        mu=args.mu)

        clients.append(client)

    if args.dp:
        print('Using differential privacy!')
    if args.fedprox:
        print('Using fedprox algorithm!')

    # communication
    for round in range(args.communication_round):
        print('The communication round is: %d' % (round + 1))
        candidates = server.sample_clients(args.all_clients)

        # local update and aggregate
        for participant in candidates:
            # Delivery model
            clients[participant].download(copy.deepcopy(server_model))

            # Update
            model_state_dict = clients[participant].local_update()

            # Aggregate
            server.aggregate(model_state_dict)

        # load average weight
        global_model = server.update()

        # test
        test_acc, test_loss = test(global_model, x_test, y_test)
        print('Test acc: %.4f  Test loss: %.4f' % (test_acc, test_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--communication_round', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--noniid', type=bool, default=True, help='if True, use noniid data')
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dp', type=bool, default=True, help='if True, use differential privacy')
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--grad_norm', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sample_ratio', type=float, default=0.8)
    parser.add_argument('--all_clients', type=int, default=10)
    parser.add_argument('--mu', type=float, default=0.1)
    parser.add_argument('--fedprox', type=bool, default=False)
    args = parser.parse_args()

    main(args)
