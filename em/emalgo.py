#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
EM algorithm for estimation of gaussian mixture model
"""

import numpy as np
import matplotlib
import collections
from sklearn import mixture

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(0)


def data_generate(n):
    """ generate data
    :param n:
    :return: [ndata, nfeatures]
    """
    a = np.random.randn(n)
    b = np.where(np.random.rand(n) > 0.3, 2.0, -2.0)
    return np.array(a + b)[:, np.newaxis]


def phi(x, mu, sigma, d):
    """ gaussian

    :param x:
    :param mu:
    :param sigma:
    :return:
    """
    denom = np.sqrt(pow(2 * np.pi * sigma * sigma, d))
    e = np.exp(-(x - mu).T @ (x - mu) / (2 * sigma * sigma))
    return 1 / denom * e


def eta_ij(i, j, n, m, x, param):
    d = x.shape[1]
    denom = sum([param["w"][j] * phi(x[i], param["mu"][j], param["sigma"][j], d) for j in range(m)])
    return param["w"][j] * phi(x[i], param["mu"][j], param["sigma"][j], d) / denom


def mstep(x, param, n, m):
    """ max step

    :param x:
    :param param:
    :return: newparam
    """
    d = x.shape[1]
    newparam = collections.defaultdict(list)
    for j in range(m):
        newparam["w"].append(1 / n * sum([eta_ij(i, j, n, m, x, param) for i in range(n)]))
        newparam["mu"].append(sum([eta_ij(i, j, n, m, x, param) * x[i] for i in range(n)]) /
                              sum([eta_ij(i, j, n, m, x, param) for i in range(n)]))
        newparam["sigma"].append(np.sqrt(
            sum([eta_ij(i, j, n, m, x, param) * (x[i] - param["mu"][j]).T @ (x[i] - param["mu"][j]) for i in
                 range(n)]) / (
                    d * sum([eta_ij(i, j, n, m, x, param) for i in range(n)]))))
    return newparam


def gaussian_mixture_distribution(x, parameter):
    """

    :param x:
    :param parameter:
    :return:
    """
    mix = param["w"][0] * norm.pdf(x=x, loc=param["mu"][0], scale=param["sigma"][0]) + (1 - param["w"][0]) * norm.pdf(
        x=x, loc=param["mu"][1], scale=param["sigma"][1])
    a = param["w"][0] * norm.pdf(x=x, loc=param["mu"][0], scale=param["sigma"][0])
    b = (1 - param["w"][0]) * norm.pdf(
        x=x, loc=param["mu"][1], scale=param["sigma"][1])
    return [mix, a, b]


if __name__ == '__main__':
    n = 1000
    m = 2
    data = data_generate(n)
    """
    clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
    clf.fit(data)
    print(clf.means_)  # [[ 2.02294145] [-2.17767632]]
    print(clf.covariances_)  # [[[ 0.951965  ]]  [[ 0.96413187]]]
    print(clf.weights_)  # [ 0.7104716  0.2895284]
    """
    param = {}
    param["w"] = [0.6, 0.4]
    param["mu"] = [1.0, -2.0]
    param["sigma"] = [1., 1.]
    print(data.shape)
    for i in range(10):
        param = mstep(data, param, n, m)
        print(param)
    print(data.shape)
    fig, ax = plt.subplots(1, 1)
    ax.hist(data, bins=16)
    x = np.linspace(-5, 5, 100)
    ax.set_xlim([-5, 5])
    res = gaussian_mixture_distribution(x, param)
    ax.plot(x, n * res[0], label=u'Gaussian mixture')
    ax.plot(x, n * res[1], label=u'Gaussian')
    ax.plot(x, n * res[2], label=u'Gaussian')
    plt.title("Gaussian mixture estimation")
    plt.savefig("em.pdf")
    plt.savefig("em.png")
