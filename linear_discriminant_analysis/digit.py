#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
classification of handwritten digits
"""

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
import seaborn as sn
import pandas as pd
from sklearn import metrics


def linear_discriminant_analysis(train, test):
    """

    :param train:
    :param test:
    :return:
    """
    # make list of mu
    mu = list()
    for i in range(10):
        mu.append(np.mean(train[:, :, i], axis=1))
    mu = np.array(mu)
    assert mu.shape == (10, 256)
    # covariance matrix
    sigma = np.zeros((256, 256))
    for i in range(10):
        sigma += np.cov(train[:, :, i])
    sigma = sigma / 10
    assert sigma.shape == (256, 256)
    predictions = []
    for i in range(10):
        test_batch = test[:, :, i]
        # print("test_batch shape", test_batch.shape)
        assert test_batch.shape == (256, 200)
        left = mu @ np.linalg.solve(sigma, test_batch)
        assert left.shape == (10, 200)
        right = - 1 / 2 * np.diag(mu @ np.linalg.solve(sigma, mu.T))[:, np.newaxis]
        # print("left: ", left.shape, " right: ", right.shape)
        prod = left + right
        assert prod.shape == (10, 200)
        predictions.append(np.argmax(prod, axis=0))
    predictions = np.array(predictions)
    y_true = np.array([[j for _ in range(200)] for j in range(10)])
    predictions = predictions.flatten()
    assert len(predictions) == 2000
    y_true = y_true.flatten()
    cm = metrics.confusion_matrix(y_true, predictions)
    df_cm = pd.DataFrame(cm, index=[i for i in "1234567890"],
                         columns=[i for i in "1234567890"])
    plt.figure(figsize=(10, 7))
    ax = sn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt='g', square=True)
    ax.set(xlabel='predicted label', ylabel='true label')
    plt.savefig("digit.png")


def mahalanobis(train, test):
    """ classification usign mahalanobis distance

    :param train:
    :param test:
    :return:
    """
    # make list of mu
    mu = list()
    for i in range(10):
        mu.append(np.mean(train[:, :, i], axis=1))
    mu = np.array(mu)
    assert mu.shape == (10, 256)
    # covariance matrix
    sigma = list()
    for i in range(10):
        S = np.cov(train[:, :, i])
        sigma.append(S)
    predictions = []
    for i in range(10):
        test_batch = test[:, :, i]
        assert test_batch.shape == (256, 200)
        prods = list()
        for k in range(10):
            # print("test_batch shape", test_batch.shape)
            left2, _, _, _ = np.linalg.lstsq(sigma[k], test_batch - mu[k, :].T[:, np.newaxis])
            left = np.diag((test_batch.T - mu[k, :]) @ left2)
            (sign, logdet) = np.linalg.slogdet(sigma[k] + 0.000001 * np.identity(256))
            '''
            if sign == 0:
                (sign, logdet) = np.linalg.slogdet(sigma[k] + 0.000001 * np.identity(256))
            assert sign > 0
            '''
            right = - 1 / 2 * logdet
            # print("left: ", left.shape, " right: ", right.shape)
            prod = left + right
            prods.append(prod)
        prods = np.array(prods)
        print(prods[:, 0])
        assert prods.shape == (10, 200), str(prods.shape)
        predictions.append(np.argmax(prods, axis=0))

    predictions = np.array(predictions)
    y_true = np.array([[j for _ in range(200)] for j in range(10)])
    predictions = predictions.flatten()
    assert len(predictions) == 2000
    y_true = y_true.flatten()
    cm = metrics.confusion_matrix(y_true, predictions)
    df_cm = pd.DataFrame(cm, index=[i for i in "1234567890"],
                         columns=[i for i in "1234567890"])
    plt.figure(figsize=(10, 7))
    ax = sn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt='g', square=True)
    ax.set(xlabel='predicted label', ylabel='true label')
    plt.show()


if __name__ == '__main__':
    data = loadmat('digit.mat')
    train = data['X']
    test = data['T']
    print("Train data: {}".format(train.shape))
    print("Test data:  {}".format(test.shape))
    # Train data: (256, 500, 10)
    # Test data:  (256, 200, 10)
    # mahalanobis(train, test)
    linear_discriminant_analysis(train, test)
