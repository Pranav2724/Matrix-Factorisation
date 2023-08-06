# https://udemy.com/recommender-systems
# https://deeplearningcourses.com/recommender-systems
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime

# load in the data
import os
if not os.path.exists('large_files/user2movie.json') or \
   not os.path.exists('large_files/movie2user.json') or \
   not os.path.exists('large_files/usermovie2rating.json') or \
   not os.path.exists('large_files/usermovie2rating_test.json'):
   import preprocess2dict


with open('large_files/user2movie.json', 'rb') as f:
  user2movie = pickle.load(f)

with open('large_files/movie2user.json', 'rb') as f:
  movie2user = pickle.load(f)

with open('large_files/usermovie2rating.json', 'rb') as f:
  usermovie2rating = pickle.load(f)

with open('large_files/usermovie2rating_test.json', 'rb') as f:
  usermovie2rating_test = pickle.load(f)

N = np.max(list(user2movie.keys())) + 1
# the test set may contain movies the train set doesn't have data on
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("N:", N, "M:", M)

# initialize variables
K = 10  # latent dimensionality
W = np.random.randn(N, K)
b = np.zeros(N)
U = np.random.randn(M, K)
c = np.zeros(M)
mu = np.mean(list(usermovie2rating.values()))


def get_mse(d):  # Takes input a dictionary with mapping from (user,movie) -> rating
    sse = 0
    N = len(d)
    for usermovie, rating in d.items():
        i, j = usermovie
        p = W[i].dot(U[j]) + b[i] + c[j] + mu
        sse += (rating - p)*(rating - p)
    return sse/N


reg_lambda = 20
n_epochs = 50
train_losses = []
test_losses = []

for epoch in range(n_epochs):
    # Update W and b
    for i in range(N):
        matrix = reg_lambda*np.eye(K)
        vector = np.zeros(K)

        bi = 0
        for j in user2movie[i]:
            r = usermovie2rating[(i,j)]
            matrix += np.outer(U[j],U[j])
            vector += (r - b[i] - c[j] - mu)*U[j]
            bi += (r - W[i].dot(U[j]) - c[j] - mu)
        W[i] = np.linalg.solve(matrix, vector)
        b[i] = bi / (len(user2movie[i]) + reg_lambda)

    # Update U and c
    for j in range(M):
        matrix = reg_lambda * np.eye(K)
        vector = np.zeros(K)

        cj = 0
        try:
            for i in movie2user[j]:
                r = usermovie2rating[(i, j)]
                matrix += np.outer(W[i], W[i])
                vector += (r - b[i] - c[j] - mu) * W[i]
                cj += (r - W[i].dot(U[j]) - b[i] - mu)
            U[j] = np.linalg.solve(matrix, vector)
            c[j] = cj / (len(movie2user[j]) + reg_lambda)

        except KeyError:
            pass

    train_losses.append(get_mse(usermovie2rating))
    test_losses.append(get_mse(usermovie2rating_test))

    print("Train loss for epoch ", epoch, " is ", train_losses[-1])
    print("Test loss for epoch ", epoch, " is ", test_losses[-1])

print("Train loss is ", train_losses[-1])
print("Test loss is ", test_losses[-1])

plt.plot(train_losses, label="train loss")
plt.plot(test_losses, label="test loss")
plt.legend()
plt.show()




