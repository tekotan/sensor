import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os import listdir
import random

import scipy.cluster.hierarchy as sch
from scipy import stats

import tensorflow as tf

from tensorflow.keras.layers import Input, TimeDistributed, Dense, LSTM
from tensorflow.keras.models import Model

data = np.zeros((20, 170, 3))

for i in range(data.shape[0]):
    for j in range(data.shape[2]):
        data[i, :, j] = np.random.normal((i%2 * 40 + 20), 3, data.shape[1])


inp = Input(shape=(data.shape[1], data.shape[2]))
    
encoder = TimeDistributed(Dense(200, activation='tanh'))(inp)
encoder = TimeDistributed(Dense(50, activation='tanh'))(encoder)
latent = TimeDistributed(Dense(10, activation='tanh'))(encoder)
decoder = TimeDistributed(Dense(50, activation='tanh'))(latent)
decoder = TimeDistributed(Dense(200, activation='tanh'))(decoder)
out = TimeDistributed(Dense(3))(decoder)
autoencoder = Model(inputs=inp, outputs=out)
autoencoder.compile(optimizer='adam', loss='mse')
encoder_model = Model(inputs=inp, outputs=latent)

autoencoder.fit(data, data, epochs=100, verbose=2)

predictions = autoencoder.predict([[data[11]]])
mse = np.mean(np.power(data[11] - predictions, 2), axis=2)

plt.figure(figsize=(16,6))
plt.scatter(range(data.shape[1]), mse)
plt.title('reconstruction error ')
plt.xlabel('time'); plt.ylabel('mse')

d = sch.distance.pdist(corr)
L = sch.linkage(d, method='ward')
ind = sch.fcluster(L, d.max(), 'distance')
dendrogram = sch.dendrogram(L, no_plot=True)
df = [df[i] for i in dendrogram['leaves']]
labels = [person_id[10:][i] for i in dendrogram['leaves']]
corr = np.corrcoef(df)
dendrogram = sch.dendrogram(L, labels=[person_id[10:][i] for i in dendrogram['leaves']])
