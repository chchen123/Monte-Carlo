#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 10:06:20 2018
@author: chen
This script fits all the proton events in run 130, selected using a CNN model,
using Monte Carlo method.
"""

import numpy as np
import pytpc
import yaml
import h5py
import time
import logging
import logging.config
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)

#defining paths
run_num = 130
config_path = '/home/chen/ar46/config/config_e15503b_p.yml'
input_path = '/home/chen/ar46/clean_events/clean_run_0'+str(run_num)+'.h5'
output_path = '/home/chen/ar46/MonteCarlo/proton_run_0'+str(run_num)+'.h5'
proton_path = '/home/chen/event-classification/test'+'.txt'

run_ID = input_path[-11:-3]
DETECTOR_LENGTH = 1250.0
DRIFT_VEL = 5.2
CLOCK = 12.5 

with open(config_path, 'r') as f:
    config = yaml.load(f)

inFile = h5py.File(input_path, 'r')
dataset_name = '/clean'
evt_inFile = inFile[dataset_name]

mcfitter = pytpc.fitting.MCFitter(config)
chi_values = np.empty(shape=(0,0))
proton_evts = np.empty(shape=(0,0))
full_evts = np.empty(shape=(0,0))
deleted_evtid = []

sum_values = np.empty(shape=(0,0))
time_values = np.empty(shape=(0,0))

output_array = np.empty(shape=(0,0))
#read and append the event IDs for proton events
with open(proton_path, "r") as output:
    for i in output:
        full_evts = np.append(full_evts, int(i[0]))
        
for i in range(len(full_evts)):
    if full_evts[i] == 1:
        proton_evts = np.append(proton_evts,int(i))
print("number of p events:",len(proton_evts))
        
        
        
for evt_index in range(len(evt_inFile)):
    try:
        #testing if each event exists
        xyzs_h5 = evt_inFile[str(evt_index)]
        xyzs = np.array(xyzs_h5)
    except Exception:
        #if a certain event does not exist, leave out the empty event index so that proton event ID 
        #fits the total event ID 
        for i in range(len(proton_evts)):
            if proton_evts[i] >= evt_index:
                proton_evts[i] += 1
        continue
        
for evt_index in range(len(evt_inFile)):
#for evt_index in range(5000,15000):
    try:
        #testing if each event exists
        xyzs_h5 = evt_inFile[str(evt_index)]
        xyzs = np.array(xyzs_h5)
    except Exception:
        continue
        
    if evt_index in proton_evts:
        
        print(evt_index)
        del_list = []  
        for i in range(len(xyzs)):
            #disregard the points that have time bucket index>500
            if (xyzs[i][2])*CLOCK/DRIFT_VEL > 500.0:
                del_list.append(i)
            #disregard the points that have less than two neighbors
            elif (xyzs[i][5] < 2.0): 
                del_list.append(i) 
            #delete points that are more than 40mm away from the unfolded spiral
            elif xyzs[i][6] > 40.0:
                del_list.append(i)
        xyzs = np.delete(xyzs, del_list, axis=0)
        xy = xyzs[:, 0:2]
    
        #find the center of curvature of each event's track
        try:
            xy_C = np.ascontiguousarray(xy, dtype=np.double)
            cx, cy = pytpc.cleaning.hough_circle(xy_C)
        except Exception:
            logger.exception('Cannot find the center of curvature for event with index %d', evt_index)
            continue
                
        #preprocess each event
        try:
            uvw, (cu, cv) = mcfitter.preprocess(xyzs[:,0:5], center=(cx, cy), rotate_pads=False)
        except Exception:
            logger.exception('Failed to preprocess event with index %d', evt_index)
            continue
                
        #fit each event with naive Monte Carlo method
        try:
            t0 = time.time()
            mcres, minChis, all_params, good_param_idx = mcfitter.process_event(uvw, cu, cv, return_details=True)
            t1 = time.time()
            
            if np.isnan(mcres['enChi2']) == True: # disregard the NaN results
                raise ValueError('event is not physical')
            else: 
                sum_values = np.append(sum_values,mcres['enChi2']+mcres['posChi2']+mcres['vertChi2'])
                print("chi^2: " + str(sum(sum_values)/float(len(sum_values))))
                time_values = np.append(time_values, t1-t0)
                print("time: " + str(sum(time_values)/float(len(time_values)))) 
        except Exception:
            logger.exception('Monte Carlo fitting failed for event with index %d', evt_index)
            continue        

np.savetxt("MonteCarlo130Proton_cleaned.txt",sum_values)