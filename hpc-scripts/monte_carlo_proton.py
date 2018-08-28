#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 10:06:20 2018
@author: chen
This script fits all the proton events, selected using Jack Taylor's NN model, in a designated run.
"""

import numpy as np
import pytpc
import yaml
import h5py
import argparse
import logging
import logging.config

logger = logging.getLogger(__name__)

#defining paths
run_num = 122
config_path = '/home/chen/ar46/config/config_e15503b_p.yml'
input_path = '/home/chen/ar46/clean_events/clean_run_0'+str(run_num)+'.h5'
output_path = '/home/chen/data/proton_0'+str(run_num)+'.h5'
proton_path = '/home/chen/data/keras-results/pCnoise/proton_events_0'+str(run_num)+'.txt'

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

#read and append the event IDs for proton events
with open(proton_path, "r") as output:
    for i in output:
        i = i[:-1]
        proton_evts = np.append(proton_evts, int(i))
        
with h5py.File(output_path, 'w') as outFile:
    gp = outFile.require_group('monte carlo')
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
        
        if evt_index in proton_evts:
            #check if the z-coordinate is located within the length detector
            try:
                for point in xyzs:
                    if (point[2] > DETECTOR_LENGTH):
                        raise ValueError('event is not physical') #disregard the non-physical events
            except ValueError:
                logger.exception('Event index %d deleted: non-physical evet', evt_index)
                continue
            
            #delete events that have less than 50 data points
            try:
                if len(xyzs) < 50:
                    raise ValueError('event has too few data points')
            except ValueError:
                logger.exception('Event index %d deleted: non-physical evet', evt_index)
                continue
            
            try:
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
            except Exception:
                logger.exception('Failed to read event with index %d from input', evt_index)
                continue
            
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
                mcres, minChis, all_params, good_param_idx = mcfitter.process_event(uvw, cu, cv, return_details=True)
                if np.isnan(mcres['posChi2']) != True:
                    chi_values = np.append(chi_values, [mcres['posChi2']+mcres['enChi2']+mcres['vertChi2']])
            except Exception:
                logger.exception('Monte Carlo fitting failed for event with index %d', evt_index)
                continue        
            
            #write the results for each event onto the .h5 file
            try:
                dset = gp.create_dataset('{:d}'.format(evt_index), data=chi_values, compression='gzip')
            except Exception:
                logger.exception('Writing to HDF5 failed for event with index %d', evt_index)
                continue
            
            chi_values = np.empty(shape=(0,0))

print('MC fitting complete for '+run_ID)