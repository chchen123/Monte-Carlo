#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 09:57:00 2018

@author: chen
"""
from keras.models import Sequential
from keras.models import model_from_yaml
import scipy as sp
import csv

#import os

# load json and create model
yaml_file = open("/home/chen/data/models/pCnoise/basicNN_NOISE.yaml", 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("/home/chen/data/models/pCnoise/basicNN_NOISE.h5")
print("Loaded model from disk")

run_num = 131
run_npz = sp.sparse.load_npz('/home/chen/data/tilt/20x20x20/run_0'+str(run_num)+
                              '.npz')
real_csr = sp.sparse.vstack([run_npz], format='csr')


# evaluate loaded model on test data
model = Sequential()
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
result = loaded_model.predict(real_csr.todense(), batch_size=10, verbose=1, steps=None)

for i in range(len(result)):
    for j in range(len(result[i])):
        if result[i][j] == max(result[i]):
            result[i][j] = 1
        else:
            result[i][j] = 0
print(len(result))
proton_evt_list = []
for i in range(len(result)):
    if result[i][0] == 1:
        proton_evt_list.append(i)
  
csvfile = '/home/chen/data/keras-results/pCnoise/proton_events_0'+str(run_num)+'.txt'
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in proton_evt_list:
        writer.writerow([val])