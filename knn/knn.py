#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
K nearest neighborhood classifier
"""
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
import seaborn as sn
import pandas as pd
from sklearn import metrics


def knn(train_x, train_y, test_x, k_list):
    """kNN

    :param train_x: train dataset
    :param train_y: category of train data
    :param test_x: dataset to classify
    :param k_list: list of choices of k
    :return: ret_matrix.shape == (len(k_list), len(test_x))
    """

    train_x = train_x.astype(np.float32)
    test_x = test_x.astype(np.float32)
    tmp = train_x[None] - test_x[:, None]
    dist_matrix = np.sqrt(np.sum((tmp) ** 2, axis=2))
    sorted_index_matrix = np.argsort(dist_matrix, axis=1)
    ret_matrix = None
    for k in k_list:
        knn_label = train_y[sorted_index_matrix[:, :k]]
        label_sum_matrix = None
        for i in range(10):
            predict = np.sum(np.where(knn_label == i, 1, 0), axis=1)[:, None]
            if label_sum_matrix is None:
                label_sum_matrix = predict
            else:
                label_sum_matrix = np.concatenate([label_sum_matrix,
                                                   predict], axis=1)
        if ret_matrix is None:
            ret_matrix = np.argmax(label_sum_matrix, axis=1)[None]
        else:
            ret_matrix = np.concatenate([ret_matrix, np.argmax(
                label_sum_matrix, axis=1)[None]], axis=0)
    return ret_matrix  # ret_matrix.shape == (len(k_list), len(test_x))


def crossvalidate(train_x, train_y):
    klist = [1, 2, 3, 4, 5]
    scores = [0, 0, 0, 0, 0]
    for j in range(50):
        ind = np.ones(5000, dtype=bool)
        ind[100 * j:100 * (j + 1)] = False
        test_x = train_x[100 * j:100 * (j + 1), :]
        train = train_x[ind, :]
        train_label = train_y[ind]
        test_label = train_y[100 * j:100 * (j + 1)]
        res = knn(train, train_label, test_x, klist)
        for i in range(len(klist)):
            print("res", res[i])
            print("label", test_label)
            scores[i] += np.sum(res[i, :] == test_label)
    print(scores)


def plot(train_x, train_y, test_x, k):
    predictions = knn(train_x, train_y, test_x, [k])
    print(predictions.shape)
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
    plt.title("kNN")
    plt.savefig("kNN.png")
    plt.savefig("kNN.pdf")
    plt.show()


if __name__ == '__main__':
    data = loadmat('digit.mat')
    train_x = data['X']
    train_x = np.concatenate(train_x.T, axis=0)
    train_y = np.array([[j for _ in range(500)] for j in range(10)]).flatten()
    test_x = data['T']
    test_x = np.concatenate(test_x.T, axis=0)
    # print(train_x.shape)
    # print(train_y.shape)
    # print(test_x.shape)
    # crossvalidate(train_x, train_y)
    plot(train_x, train_y, test_x, 1)
