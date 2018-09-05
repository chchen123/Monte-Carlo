#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 15:05:53 2018

@author: chen
This Python script reads labeled proton data and fit them using Monte Carlo method.
Returns a .h5 file containing the chi^2 fitting result for each proton event.
"""
import pandas as pd
import pytpc
import h5py
import numpy as np
import yaml

run = '0210'

data_path = '/home/chen/ar46/clean_events/clean_run_'+run+'.h5'
output_path = '/home/chen/ar46/proton_chi_values/run_'+run+'.h5'
config_path = '/home/chen/ar46/config/config_e15503b_p.yml'

labels = pd.read_csv('/home/chen/data/real/' + "run_" + run + "_labels.csv", sep=',')
p_indices = labels.loc[(labels['label'] == 'p')]['evt_id'].values
print(p_indices)
DRIFT_VEL = 5.2
CLOCK = 12.5 

with open(config_path, 'r') as f:
    config = yaml.load(f)
mcfitter = pytpc.fitting.MCFitter(config)

chi_values = np.empty(shape=(0,0))


inFile = h5py.File(data_path, 'r')
dataset_name = '/clean'
evt_inFile = inFile[dataset_name]

with h5py.File(output_path, 'w') as outFile:
    gp = outFile.require_group('monte carlo')
    for event_index in range(len(evt_inFile)):
        if event_index in p_indices:
            try:
                xyzs_h5 = evt_inFile[str(event_index)]
                xyzs = np.array(xyzs_h5)
            except Exception: #occurs when certain events do not exist
                continue            
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
                continue
                    
            #preprocess each event
            try:
                uvw, (cu, cv) = mcfitter.preprocess(xyzs[:,0:5], center=(cx, cy), rotate_pads=False)
            except Exception:
                continue
                    
            #fit each event with naive Monte Carlo method
            try:
                mcres, minChis, all_params, good_param_idx = mcfitter.process_event(uvw, cu, cv, return_details=True)
                if np.isnan(mcres['enChi2']) == True: # disregard the NaN results
                    raise ValueError('event is not physical')
                elif np.isnan(mcres['posChi2']) == True:
                    raise ValueError('event is not physical')
                elif np.isnan(mcres['vertChi2']) == True:
                    raise ValueError('event is not physical')
                else: 
                    chi_values = np.append(chi_values, [mcres['posChi2']+mcres['enChi2']+mcres['vertChi2']])
            except Exception:
                continue        
                    
            #write the results for each event onto the .h5 file
            try:
                dset = gp.create_dataset('{:d}'.format(event_index), data=chi_values, compression='gzip')
            except Exception:
                continue
                        
            chi_values = np.empty(shape=(0,0))
                  
