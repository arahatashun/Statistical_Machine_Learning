#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
Fisher’s linear discriminant analysis
"""
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(0)


def generate_sample(n, alpha):
    n1 = sum(np.random.rand(n) < alpha)
    n2 = n - n1
    mean1, mean2 = np.array([2, 0]), np.array([-2, 0])
    cov = np.array([[1, 0], [0, 9]])
    x1 = np.random.multivariate_normal(mean1, cov, n1).transpose()
    x2 = np.random.multivariate_normal(mean2, cov, n2).transpose()
    return x1, x2


def calculate_hyper_plane(x1, x2, alpha):
    """ calculate hyper plane

    :param x1:
    :param x2:
    :return: a b (a^Tx + b = 0)
    """
    sigma = alpha * np.cov(x1) + (1 - alpha) * np.cov(x2)
    mu1 = np.average(x1, axis=1)
    # print(mu1.shape)
    mu2 = np.average(x2, axis=1)
    # print(sigma.shape)
    a = np.linalg.solve(sigma, mu1 - mu2)
    b = -1 / 2 * (mu1.T @ np.linalg.solve(sigma, mu1) - mu2.T @ np.linalg.solve(sigma, mu2)) + \
        np.log(len(mu1) / len(mu2))
    return {"a": a, "b": b}


def plot(x1, x2, alpha, ax):
    param = calculate_hyper_plane(x1, x2, alpha)
    ax.scatter(x1[0], x1[1], facecolors='none', edgecolors='r')
    ax.scatter(x2[0], x2[1], marker='X', c='blue')
    intercept = - param["b"] / param["a"][1]
    gradient = -param["a"][0] / param["a"][1]
    x = np.linspace(-5, 5, num=100)
    print("gradient: ", gradient, " intercept: ", intercept)
    y = gradient * x + intercept
    ax.plot(x, y, 'black', label='hyper plane')
    ax.set_ylim(-10, 10)
    ax.set_xlim(-5, 5)
    ax.set_title("alpha = " + str(alpha))


if __name__ == '__main__':
    fig = plt.figure(figsize=(9, 3))
    alpha = 0.005
    ax1 = fig.add_subplot(1, 3, 1)
    x1, x2 = generate_sample(1000, alpha)
    plot(x1, x2, alpha, ax1)
    ax2 = fig.add_subplot(1, 3, 2)
    alpha = 0.5
    x1, x2 = generate_sample(1000, alpha)
    plot(x1, x2, alpha, ax2)
    ax3 = fig.add_subplot(1, 3, 3)
    alpha = 0.999
    x1, x2 = generate_sample(1000, alpha)
    plot(x1, x2, alpha, ax3)
    plt.savefig("fisher.png")
