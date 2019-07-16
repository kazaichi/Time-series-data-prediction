# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
import tensorflow as tf
from make_chaos import create_duffing

import random as rn
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1337)
rn.seed(1337)
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
tf.set_random_seed(1337)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

#データの設定
x_lin, t_lin = create_duffing(1000)

length_of_sequences = len(t_lin)
maxlen = 150

data = []
target = []
for i in range(0, length_of_sequences - maxlen):
    data.append(x_lin[i: i + maxlen])
    target.append(x_lin[i + maxlen])

X = np.array(data).reshape(len(data), maxlen, 1)
Y = np.array(target).reshape(len(data), 1)

X_ = X[:len(t_lin) - 500 - 150]#学習用
Y_ = Y[:len(t_lin) - 500 - 150]#学習用
N_train = int(len(data) * 0.7)
N_validation = len(data) - N_train
X_train, X_validation, Y_train, Y_validation = \
    train_test_split(X_, Y_, test_size=N_validation, shuffle=False)

#予測対象の表示
plt.plot(x_lin[len(x_lin)-500:len(x_lin)])
plt.show()
print("a")

#学習モデルの作成
in_out_neurons = 1
hidden_neurons = 300

pat = 1
early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)
csv = CSVLogger('./Training_patience_' + str(pat) + '.csv')

model = Sequential()  
model.add(LSTM(hidden_neurons, batch_input_shape=(None, maxlen, in_out_neurons), return_sequences=False))  
model.add(Dense(in_out_neurons))  
model.add(Activation("linear"))  
model.compile(loss="mean_squared_error", optimizer="rmsprop")
fit = model.fit(X_train, Y_train, batch_size=512, epochs=10000, 
          validation_data=(X_validation, Y_validation), callbacks=[early_stopping, csv])

#予測
n_in = len(X[0][0])
original = x_lin[len(x_lin)-500:len(x_lin)-500+maxlen]
Z = original.reshape(1, 150, -1)
predicted = [None for i in range(maxlen)]

for i in range(length_of_sequences - maxlen + 1):
    z_ = Z[-1:]
    y_ = model.predict(z_)
    sequence_ = np.concatenate((z_.reshape(maxlen, n_in)[1:], y_),axis=0)
    sequence_ = sequence_.reshape(1, maxlen, n_in)
    Z = np.append(Z, sequence_, axis=0)
    predicted.append(y_.reshape(-1))

plt.rc('font', family='serif')
plt.figure()
plt.ylim([-4, 4])
plt.plot(x_lin[len(x_lin)-500:len(x_lin)], linestyle='dotted', color='#aaaaaa')
plt.plot(original, linestyle='dashed', color='black')
plt.plot(predicted[:500], color='black')
plt.show()

# loss
def plot_history_loss(fit):
    # Plot the loss in the history
    plt.plot(fit.history['loss'],label="loss for training")
    plt.plot(fit.history['val_loss'],label="loss for validation")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')

plt.figure()
plot_history_loss(fit)