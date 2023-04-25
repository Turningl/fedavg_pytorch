# -*- coding: utf-8 -*-
# @Author : Zhang
# @Email : zl16035056@163.com
# @File : plt.py

import matplotlib.pyplot as plt


def plt_curve(algorithm_name, test_acc, test_loss):
    epochs = range(1, len(test_acc) + 1)
    plt.figure(1)
    plt.plot(epochs, test_loss, label=algorithm_name)
    plt.title('test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig('./results/%s_loss.svg' % algorithm_name, dpi=1080)

    plt.figure(2)
    plt.plot(epochs, test_acc, label=algorithm_name)
    plt.title('test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.savefig('./results/%s_acc.svg' % algorithm_name, dpi=1080)
    plt.show()
