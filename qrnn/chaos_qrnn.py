# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from keras import backend as K
from keras.layers import Input,Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
import tensorflow as tf
from qrnn import QRNN
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
X_yosoku = X[9851:len(X)]#予測用
Y_yosoku = Y[9851:len(Y)]#予測用

# データ設定
N_train = int(len(data) * 0.7)
N_validation = len(data) - N_train

X_train, X_validation, Y_train, Y_validation = \
    train_test_split(X_, Y_, test_size=N_validation, shuffle=False)

print('Build model...')
ws = 128 #window_size
input_layer = Input(shape=(maxlen, 1))
qrnn_output_layer = QRNN(64, window_size=ws, dropout=0)(input_layer)
prediction_result = Dense(1)(qrnn_output_layer)
model = Model(inputs=input_layer, outputs=prediction_result)
model.compile(loss="mean_squared_error", optimizer="adam")

plt.plot(x_lin[len(x_lin)-500:len(x_lin)])
plt.show()

pat = 1
early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)
csv = CSVLogger('./TrainingWS_' + str(ws) + 'patience_' + str(pat) + '.csv')
fit = model.fit(X_train, Y_train, batch_size=512, epochs=10000, 
          validation_data=(X_validation, Y_validation), callbacks=[early_stopping, csv])

n_in = len(X[0][0])
original = x_lin[len(x_lin)-500:len(x_lin)-500+maxlen]
Z = original.reshape(1, 150, -1)#X_yosoku
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

#重みの保存###################
#json_string = model.to_json()
#open('modelWS_' + str(ws) + 'patience_' + str(pat) + '.json', 'w').write(json_string)
#model.save_weights('weightsWS_' + str(ws) + 'patience_' + str(pat) + '.h5')
##############################
