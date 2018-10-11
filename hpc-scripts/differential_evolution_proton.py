#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 13:25:16 2018

@author: chen
"""

import numpy as np
import pytpc
from pytpc.fitting.mcopt_wrapper import Minimizer
import yaml
import h5py
import scipy
from math import pi
import pandas as pd
import time

#define user inputs for the Bash scripts
data_path = '/home/chen/ar46/clean_events/clean_run_0130.h5'
output_path = '/home/chen/ar46/DiffEvolution/run_0130_proton.h5'
config_path = '/home/chen/ar46/config/config_e15503b_p.yml'

labels = pd.read_csv('/home/chen/data/real/' + "run_0130_labels.csv", sep=',')
p_indices = labels.loc[(labels['label'] == 'p')]['evt_id'].values

bounds = [(-1,1), (-1, 1), (0, 1), (0,5), (-2 * pi, 2 * pi), (-2 * pi, 2 * pi)]

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

num_iters = config['num_iters']
num_pts = config['num_pts']
red_factor = config['red_factor']

chi_position = np.empty(shape=(0,0))
chi_energy = np.empty(shape=(0,0))
chi_vert = np.empty(shape=(0,0))

sum_values = np.empty(shape=(0,0))
time_values = np.empty(shape=(0,0))
#writing output files
with h5py.File(output_path, 'w') as outFile:
    gp0 = outFile.require_group('total')
    gp1 = outFile.require_group('position')
    gp2 = outFile.require_group('energy')
    gp3 = outFile.require_group('vertex')
    for evt_index in range(1002):
        if evt_index in p_indices:
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
                uvw_values = uvw.values
                uvw_sorted = uvw.sort_values(by='w', ascending=True)
                prefit_data = uvw_sorted.iloc[-len(uvw_sorted) // 4:].copy()
                prefit_res = mcfitter.linear_prefit(prefit_data, cu, cv)
                ctr0 = mcfitter.guess_parameters(prefit_res)
                exp_pos = uvw_sorted[['u', 'v', 'w']].values.copy() / 1000
                exp_hits = np.zeros(10240)
                for a, p in uvw[['a', 'pad']].values:
                    exp_hits[int(p)] = a
                minimizer = Minimizer(mcfitter.tracker, mcfitter.evtgen, num_iters, num_pts, red_factor)
            except Exception:
                print('Failed to preprocess event with index '+str(evt_index))
                continue
            
            #define the objective function
            def chi2(y,add_chi2=False):
                global chi_position
                global chi_energy
                global chi_vert
                ctr = np.zeros([1,6])
                ctr[0] = y
                chi_result = minimizer.run_tracks(ctr, exp_pos, exp_hits)
                if add_chi2 == True: 
                    #add the individual component of chi^2 values once the total chi^2 is minimized
                    chi_position = np.append(chi_position, chi_result[0][0])
                    chi_energy = np.append(chi_energy, chi_result[0][1])
                    chi_vert = np.append(chi_vert, chi_result[0][2])
                return (chi_result[0][0]+chi_result[0][1]+chi_result[0][2])
    
            #fit each event with differential evolution method
            try:
                t0 = time.time()
                results = scipy.optimize.differential_evolution(chi2, bounds,\
                                                                maxiter=1000, strategy='rand1bin', recombination=0.9, popsize=10, mutation=0.8)
                t1 = time.time()
                if np.isnan(results.fun) == True: # disregard the NaN results
                    raise ValueError('event is not physical')
                chi_values = np.append(chi_values, results.fun)

                sum_values = np.append(sum_values,chi_values)
#                print(sum_values)
                print("chi^2: " + str(sum(sum_values)/float(len(sum_values))))
                time_values = np.append(time_values, t1-t0)
#                print(time_values)
                print("time: " + str(sum(time_values)/float(len(sum_values))))
                chi2(results.x,add_chi2=True)
            except Exception:
                print('Differential evolution fitting failed for event with index '+str(evt_index))
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
