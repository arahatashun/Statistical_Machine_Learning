#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
Gaussian Kernel density estimation
"""
import numpy as np

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(0)


def data_generate(n=3000):
    x = np.zeros(n)
    u = np.random.rand(n)
    index1 = np.where((0 <= u) & (u < 1 / 8))
    x[index1] = np.sqrt(8 * u[index1])
    index2 = np.where((1 / 8 <= u) & (u < 1 / 4))
    x[index2] = 2 - np.sqrt(2 - 8 * u[index2])
    index3 = np.where((1 / 4 <= u) & (u < 1 / 2))
    x[index3] = 1 + 4 * u[index3]
    index4 = np.where((1 / 2 <= u) & (u < 3 / 4))
    x[index4] = 3 + np.sqrt(4 * u[index4] - 2)
    index5 = np.where((3 / 4 <= u) & (u <= 1))
    x[index5] = 5 - np.sqrt(4 - 4 * u[index5])
    return x


def gaussian_kernel(x):
    """ x [n ,1] vector

    :param x:
    :return:
    """
    d = x.shape[0]
    tmp = 1 / (np.sqrt(np.power(2 * np.pi, d)))
    res = tmp * np.exp(-1 / 2 * np.dot(x, x))
    # print(res.shape)
    return res


def density(x, data, h):
    """ probability density function
    :param x:
    :param data:
    :param h:  band width
    :return:
    """
    # [#feature, #sample]
    n = data.shape[1]
    d = data.shape[0]
    # print(n, d)
    tmp = np.sum([gaussian_kernel((x - data[:, i]) / h) for i in range(n)])
    res = 1 / (n * pow(h, d)) * tmp
    # print(tmp)
    return res


def lcv(data):
    """ likelihood cross validation
    :return:
    """
    #  calculate log likelihood
    candidate = [0.01, 0.1, 1, 10]
    scores = [0., 0., 0., 0.]
    for i in range(len(candidate)):
        for j in range(6):
            ind = np.ones(3000, dtype=bool)
            ind[500 * j:500 * (j + 1)] = False
            test = data[:, 500 * j:500 * (j + 1)]
            train = data[:, ind]
            # print("train", train.shape)
            # print("test", test.shape)
            scores[i] += 1 / (6 * 500) * np.sum(
                [np.log(density(test[:, k], train, candidate[i])) for k in range(len(test))])
    print(scores)


if __name__ == '__main__':
    data = data_generate()
    data = np.expand_dims(data, axis=0)
    # [1, #sample]
    # print(data.shape)
    fig = plt.figure()
    ax1 = plt.axes()
    ax1.set_ylabel('histogram', color="b")
    ax1.hist(data.T, bins=50)
    x = np.linspace(0, 5.0, num=50)
    lcv(data)
    y = []
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    for i in x:
        y.append(density(i, data, h=0.1))
    ax2.plot(x, y, color="r")
    ax2.set_ylabel('density', color="r")
    plt.title("gaussian kernel density estimation")
    plt.savefig("gaussian_kernel.pdf")
    plt.show()
