#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
asymptotic normality of variance of normal distribution
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


def normal(size):
    """
    generate normal distribution
    :param num:
    :return:
    """
    mu, sigma = 0, 1  # mean and standard deviation
    data = np.random.normal(mu, sigma, size)
    mu_ml = data.mean(axis=0)
    var = 1 / size * (data - mu_ml).T @ (data - mu_ml)
    return var


def plot():
    res = []
    size = 1000
    for i in range(10000):
        res.append(normal(size))
    sns.distplot(res)
    plt.title("asymptotic normality")
    mu_, std_ = norm.fit(res)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu_, std_)
    plt.plot(x, p, 'k', linewidth=2, label="Normal Distribution Fitting")
    plt.legend()
    plt.savefig("asymp.png")
    plt.show()


if __name__ == '__main__':
    plot()
