#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 10:36:19 2018
@author: chen
This Python script fits each event of a designated run using the naive Monte Carlo method.
Expected to be called within a Bash script.

returns the Monte Carlo results of total Chi^2 values.
"""

import numpy as np
import pytpc
import yaml
import h5py

#define user inputs for the Bash scripts
data_path = '/home/chen/ar46/clean_events/clean_run_0130.h5'
output_path = '/home/chen/ar46/MonteCarlo/chi_values/run_0130.h5'
config_path = '/home/chen/ar46/config/config_e15503b_p.yml'


run_ID = data_path[-11:-3]
DETECTOR_LENGTH = 1250.0
DRIFT_VEL = 5.2
CLOCK = 12.5 

#load configurations
with open(config_path, 'r') as f:
    config = yaml.load(f)

#load (cleaned) real events
inFile = h5py.File(data_path, 'r')
dataset_name = '/clean'
evt_inFile = inFile[dataset_name]

mcfitter = pytpc.fitting.MCFitter(config)
chi_values = np.empty(shape=(0,0))

chi_position = np.empty(shape=(0,0))
chi_energy = np.empty(shape=(0,0))
chi_vert = np.empty(shape=(0,0))

#writing output files
with h5py.File(output_path, 'w') as outFile:
    gp0 = outFile.require_group('total')
    gp1 = outFile.require_group('position')
    gp2 = outFile.require_group('energy')
    gp3 = outFile.require_group('vertex')
    
    for evt_index in range(1002):
        #read events
        try:
            xyzs_h5 = evt_inFile[str(evt_index)]
            xyzs = np.array(xyzs_h5)
        except Exception: #occurs when certain events do not exist
            print('Failed to read event with index %d from input '+str(evt_index))
            continue
        
        #check if the z-coordinate is located within the length detector 
        try:
            for point in xyzs:
                if (point[2] > DETECTOR_LENGTH):
                    raise ValueError('event is not physical') #disregard the non-physical events
        except ValueError:
                print('Event index %d deleted: non-physical evet '+str(evt_index))
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
        
        #delete events that have less than 50 data points
        try:
            if len(xyzs) < 50:
                raise ValueError('event has too few data points')
        except ValueError:
            print('Event index %d deleted: non-physical evet '+str(evt_index))
            continue
        
        #find the center of curvature of each event's track
        try:
            xy = xyzs[:, 0:2]
            xy_C = np.ascontiguousarray(xy, dtype=np.double)
            cx, cy = pytpc.cleaning.hough_circle(xy_C)
        except Exception:
            print('Cannot find the center of curvature for event with index '+str(evt_index))
            continue
        
        #preprocess each event
        try:
            uvw, (cu, cv) = mcfitter.preprocess(xyzs[:,0:5], center=(cx, cy), rotate_pads=False)
        except Exception:
            print('Failed to preprocess event with index '+str(evt_index))
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
                
            chi_values = np.append(chi_values, [mcres['posChi2']+mcres['enChi2']+mcres['vertChi2']])
            chi_position = np.append(chi_position, [mcres['posChi2']])
            chi_energy = np.append(chi_energy, [mcres['enChi2']])
            chi_vert = np.append(chi_vert, [mcres['vertChi2']])

        except Exception:
            print('monte carlo fitting failed for event with index '+str(evt_index))
            continue        
        
        #write the results for each event onto the .h5 file
        try:
            dset0 = gp0.create_dataset('{:d}'.format(evt_index), data=chi_values, compression='gzip')
            dset1 = gp1.create_dataset('{:d}'.format(evt_index), data=chi_position, compression='gzip')
            dset2 = gp2.create_dataset('{:d}'.format(evt_index), data=chi_energy, compression='gzip')
            dset3 = gp3.create_dataset('{:d}'.format(evt_index), data=chi_vert, compression='gzip')
        except Exception:
            print('Writing to HDF5 failed for event with index '+str(evt_index))
            continue
        
        chi_values = np.empty(shape=(0,0))
        chi_position = np.empty(shape=(0,0))
        chi_energy = np.empty(shape=(0,0))
        chi_vert = np.empty(shape=(0,0))

print('MC fitting complete for '+run_ID)