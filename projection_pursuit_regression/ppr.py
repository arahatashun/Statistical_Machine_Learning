#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
projection pursuit regression
(Approx) Newton method
"""
import numpy as np
from scipy.linalg import sqrtm
import matplotlib
from matplotlib.lines import Line2D
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(0)


def generate_data(n=1000):
    x = np.concatenate([np.random.rand(n, 1), np.random.randn(n, 1)], axis=1)
    x[0, 1] = 6  # outlier
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)  # Standardization
    M = np.array([[1, 3], [5, 3]])
    x = x.dot(M.T)
    x = np.linalg.inv(sqrtm(np.cov(x, rowvar=False))).dot(x.T).T
    print("x shape: ", x.shape)
    # [the number of data, dimension]
    return x


def centerandwhite(x):
    """
    centering and whitening
    X = (1/n X^\top H^2 X)^{-1/2} X^\top H
    """
    n = x.shape[0]
    H = np.eye(n) - 1 / n * np.ones([n, n])
    left = sqrtm(1 / n * x.T @ H @ H @ x)
    x = (left @ x.T @ H).T
    print("x shape:", x.shape)
    return x


def newton_step(b, x, g):
    """
    b: [feature, 1]
    x: [#data, #feature]
    g: function
    g.fucntion: g(x)
    g.gradient: g'(x)
    """
    n = x.shape[1]
    newb = np.zeros(b.shape)
    for i in range(n):
        newb += b * g.gradient(np.inner(np.squeeze(b), x[i])) \
                - x[i][:, np.newaxis] * g.function(np.inner(np.squeeze(b), x[i]))
    newb /= n
    newb /= np.linalg.norm(newb)
    # need to fix sign
    newb = np.sign(newb[0]) * newb
    # print("-----------------------")
    # print(b)
    # print(newb)
    # print("-----------------------")
    return newb


def calc_projection(x, g):
    """

    :param x:
    :param g:
    :return:
    """
    threshold = 0.0001
    nfeature = x.shape[1]
    b = np.random.randn(nfeature, 1)
    diff = 1000
    max_iter = 10000
    cnt = 0
    while diff > threshold and cnt < max_iter:
        newb = newton_step(b, x, g)
        diff = np.linalg.norm(newb - b)
        b = newb
        cnt += 1
        print("iteration: ", cnt)
    return b


class Function: pass


if __name__ == '__main__':
    x = generate_data(100)
    x = centerandwhite(x)
    g = Function()
    g.function = lambda t: 4 * pow(t, 3)
    g.gradient = lambda t: 12 * pow(t, 2)
    gb = calc_projection(x, g)
    f = Function()
    f.function = lambda t: np.tanh(t)
    f.gradient = lambda t:1 - np.tanh(t) * np.tanh(t)
    fb = calc_projection(x, f)
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].scatter(x[:, 0], x[:, 1], marker='x')
    axs[0, 0].set_xlim(-3, 5)
    axs[0, 0].set_ylim(-2, 3)
    px = [-10 * gb[0], 0, 10 * gb[0]]
    py = [-10 * gb[1], 0, 10 * gb[1]]
    axs[0, 0].plot(px, py, color='r')
    axs[1, 0].hist(x @ gb, bins=20)
    axs[1, 0].set_title("s^4", y=-0.2)
    axs[0, 1].scatter(x[:, 0], x[:, 1], marker='x')
    axs[0, 1].set_xlim(-3, 5)
    axs[0, 1].set_ylim(-2, 3)
    px = [-10 * fb[0], 0, 10 * fb[0]]
    py = [-10 * fb[1], 0, 10 * fb[1]]
    axs[0, 1].plot(px, py, color='r')
    axs[1, 1].hist(x @ fb, bins=20)
    axs[1, 1].set_title("logcosh(s)", y=-0.2)
    plt.savefig("ppr.png")