#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
Probabilistic Matrix Factorization using Variational Bayesian Inference
"""

import numpy as np
from pystan import StanModel
from tabulate import tabulate
bsm = StanModel(file='model.stan')

R = np.array([
    [1, 1, 3],
    [1, 2, 3],
    [1, 4, 1],
    [2, 1, 3],
    [2, 3, 3],
    [3, 1, 1],
    [3, 4, 3],
    [4, 2, 3],
    [4, 3, 3],
    [5, 3, 1],
    [5, 4, 3],
])

K = 2
J = 5
I = 4
O = R.shape[0]
users = np.copy(R[:, 0])
items = np.copy(R[:, 1])
scores = np.copy(R[:, 2])
stan_data = {'K': K, 'J': J, 'I': I, "O": O, "users": users, "items": items, "scores": scores}

results = bsm.vb(data=stan_data, iter=100, output_samples=1)

sigma = results['mean_pars'][0]
rhou = results['mean_pars'][1:K + 1]
rhov = results['mean_pars'][K + 1:2 * K + 1]
u = np.array(results['mean_pars'][2 * K + 1:2 * K + K * J + 1]).reshape(K, J).T
v = np.array(results['mean_pars'][2 * K + K * J + 1:]).reshape(K, I).T

r = np.zeros((J, I))
for j in range(J):
    for i in range(I):
        r[j, i] = np.random.normal(np.dot(u[j], v[i]), sigma)
print(r)
for i in range(O):
    r[users[i] - 1, items[i] - 1] = scores[i]
print(r)
print(tabulate(r, tablefmt="latex", floatfmt=".2f"))