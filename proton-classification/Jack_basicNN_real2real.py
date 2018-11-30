#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basicNN_real2real_multic.py
==========================
Author: Jack Taylor
Testing a basic neural network on nuclear scattering data.
Trains on real tests on real data.
"""
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import numpy as np
import scipy as sp

import sys
sys.path.insert(0, '../modules/')
from keras import metrics
#metrics = metrics.MulticlassMetrics()

#fix random seed
np.random.seed(7)
epochs = 100
validation_split = 0.25
batch_size = 10

#real data
p_0130 = sp.sparse.load_npz('/home/chen/data/real/20x20x20/run_0130_pDisc.npz')
C_0130 = sp.sparse.load_npz('/home/chen/data/real/20x20x20/run_0130_CDisc.npz')
noise_0130 = sp.sparse.load_npz('/home/chen/data/real/20x20x20/run_0130_junkDisc.npz')
p_0210 = sp.sparse.load_npz('/home/chen/data/real/20x20x20/run_0210_pDisc.npz')
C_0210 = sp.sparse.load_npz('/home/chen/data/real/20x20x20/run_0210_CDisc.npz')
noise_0210 = sp.sparse.load_npz('/home/chen/data/real/20x20x20/run_0210_junkDisc.npz')
print(type(noise_0210))
p_real = sp.sparse.vstack([p_0130, p_0210], format='csr')
C_real = sp.sparse.vstack([C_0130, C_0210], format='csr')
noise_real = sp.sparse.vstack([noise_0130, noise_0210], format='csr')

#labels
p_real_labels = np.zeros((p_real.shape[0],))
C_real_labels = np.ones((C_real.shape[0],))
noise_real_labels = np.full((noise_real.shape[0],), 2)

#merging training data
real_train_data = sp.sparse.vstack([p_real, C_real, noise_real], format='csr')
real_labels_train_categorical = np.hstack((p_real_labels, C_real_labels, noise_real_labels))
real_labels_train = np_utils.to_categorical(real_labels_train_categorical)

#(real_train_data, real_test_data,
#real_labels_train, real_labels_test) = train_test_split(full_real_data,
#                                                        full_real_labels,
#                                                        test_size=validation_split,
#                                                        random_state=42)

#testing data
test_0131 = sp.sparse.load_npz('/home/chen/data/tilt/20x20x20/run_0131.npz')
real_test_data = sp.sparse.vstack([test_0131], format='csr')
real_test_labels = np.zeros((real_test_data.shape[0],))

#merging training and testing data
full_real_data = sp.sparse.vstack([real_train_data, real_test_data], format='csr')
full_real_labels_categorical = np.hstack((real_labels_train, real_test_labels))
full_real_labels = np_utils.to_categorical(full_real_labels_categorical)

#define model
model = Sequential()
model.add(Dense(128, input_dim=full_real_data.shape[1], activation='relu'))
model.add(Dense(full_real_labels.shape[1], activation='softmax'))

#compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit the model with a validation set split
history = model.fit(real_train_data.todense(), real_labels_train,
                    validation_data=(real_test_data.todense(), real_test_labels),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[metrics])

print(history.history.keys())
# summarize history for accuracy
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Single Layer NN Accuracy - p vs. C + junk')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('../plots/results/tilt/basicNN_sim_pCjunk_acc.pdf')
# # summarize history for loss
# plt.figure(2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Single Layer NN Loss - p vs. C + junk')
# plt.ylabel('loss')25
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], l25oc='upper left')
# plt.savefig('../plots/results/tilt/basicNN_sim_pCjunk_loss.pdf')

textfile = open('../keras-results/NN/real2real/multic.txt', 'w')
textfile.write('acc \n')
textfile.write(str(history.history['acc']))
textfile.write('\n')
textfile.write('\nval_acc \n')
textfile.write(str(history.history['val_acc']))
textfile.write('\n')
textfile.write('\nloss \n')
textfile.write(str(history.history['loss']))
textfile.write('\n')
textfile.write('\nval_loss \n')
textfile.write(str(history.history['val_loss']))
textfile.write('\n')
textfile.write('\nconfusion matrices \n')
for cm in metrics.val_cms:
    textfile.write(str(cm))
    textfile.write('\n')

print("Maximum Validation Accuracy Reached: %.5f%%" % max(history.history['val_acc']))

#model_path = '../models/20x20x20/'

# # serialize model to YAML
# model_yaml = model.to_yaml()
# with open(model_path + "basicNN_NOISE.yaml", "w") as yaml_file:
#     yaml_file.write(model_yaml)
# # serialize weights to HDF5
# model.save_weights(model_path + "basicNN_NOISE.h5")
# print("Saved model to disk")
