#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basicNN_real2real_pCnoise.py
============================
Testing a basic neural network on nuclear scattering data.
Binary classification of protons and "everything else" (carbon and noise)
Trains on real tests on real.
"""
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import scipy as sp
from keras.utils import np_utils

import sys
sys.path.insert(0, '../modules/')
#import metrics
#metrics = metrics.BinaryMetrics()

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

p_real = sp.sparse.vstack([p_0130, p_0210], format='csr')
C_real = sp.sparse.vstack([C_0130, C_0210], format='csr')
noise_real = sp.sparse.vstack([noise_0130, noise_0210], format='csr')

#labels
p_real_labels = np.zeros((p_real.shape[0],))
C_real_labels = np.ones((C_real.shape[0],))
noise_real_labels = np.ones((noise_real.shape[0],))

#merging
full_real_data = sp.sparse.vstack([p_real, C_real, noise_real], format='csr')
full_real_labels_categorical = np.hstack((p_real_labels, C_real_labels, noise_real_labels))
full_real_labels = np_utils.to_categorical(full_real_labels_categorical)
print(full_real_labels)

(train_data, test_data,
labels_train, labels_test) = train_test_split(full_real_data,
                                              full_real_labels,
                                              test_size=validation_split,
                                              random_state=42)

#define model
model = Sequential()
model.add(Dense(128, input_dim=full_real_data.shape[1], activation='relu'))
model.add(Dense(full_real_labels.shape[1], activation='sigmoid'))

#compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit the model with a validation set split
history = model.fit(train_data.todense(), labels_train,
                    validation_data=(test_data.todense(), labels_test),
                    epochs=epochs,
                    batch_size=batch_size)

print(history.history.keys())
print("Maximum Validation Accuracy Reached: %.5f%%" % max(history.history['val_acc']))
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
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('../plots/results/tilt/basicNN_sim_pCjunk_loss.pdf')

textfile = open('/home/chen/data/keras-results/pCnoise.txt', 'w')
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


model_path = '/home/chen/data/models/pCnoise/'

# serialize model to YAML
model_yaml = model.to_yaml()
with open(model_path + "basicNN_NOISE.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights(model_path + "basicNN_NOISE.h5")
print("Saved model to disk")