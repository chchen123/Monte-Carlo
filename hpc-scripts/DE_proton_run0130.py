#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 13:25:16 2018

This script fits all the proton events in run 130, selected using a CNN model,
using Differential Evolution method.
"""

import numpy as np
import pytpc
from pytpc.fitting.mcopt_wrapper import Minimizer
import yaml
import h5py
import scipy
from math import pi
import logging
import logging.config
logger = logging.getLogger(__name__)
import time

#define user inputs for the Bash scripts
data_path = '/home/chen/ar46/clean_events/clean_run_0130.h5'
output_path = '/home/chen/ar46/DiffEvolution/run_0130_proton_batch3.h5'
config_path = '/home/chen/ar46/config/config_e15503b_p.yml'
proton_path = '/home/chen/event-classification/test'+'.txt'

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

proton_evts = np.empty(shape=(0,0))
full_evts = np.empty(shape=(0,0))
deleted_evtid = []

#read and append the event IDs for proton events
with open(proton_path, "r") as output:
    for i in output:
        full_evts = np.append(full_evts, int(i[0]))
        
for i in range(len(full_evts)):
    if full_evts[i] == 1:
        proton_evts = np.append(proton_evts,int(i))

for evt_index in range(len(evt_inFile)):
    try:
        #testing if each event exists
        xyzs_h5 = evt_inFile[str(evt_index)]
        xyzs = np.array(xyzs_h5)
#       xyzs = np.insert(xyzs,7,evt_index,axis=1)
    except Exception:
        #if a certain event does not exist, leave out the empty event index so that proton event ID 
        #fits the total event ID 
        for i in range(len(proton_evts)):
            if proton_evts[i] >= evt_index:
                proton_evts[i] += 1
        continue

for evt_index in range(len(evt_inFile)):
    try:
        xyzs_h5 = evt_inFile[str(evt_index)]
        xyzs = np.array(xyzs_h5)
    except Exception: #occurs when certain events do not exist
        print('Failed to read event with index %d from input '+str(evt_index))
        continue
    
    if evt_index in proton_evts:
        
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

            return (chi_result[0][0]+chi_result[0][1]+chi_result[0][2])
        
        #fit each event with differential evolution method
        try:
            t0 = time.time()
            results = scipy.optimize.differential_evolution(chi2, bounds,\
                                                            maxiter=1000, strategy='rand1bin', recombination=0.9, popsize=10, mutation=(0.5,1.0))
            t1 = time.time()
            if np.isnan(results.fun) == True: # disregard the NaN results
                raise ValueError('event is not physical')
            time_values = np.append(time_values, t1-t0)
            sum_values = np.append(sum_values,results.fun)
            print("chi^2: " + str(sum(sum_values)/float(len(sum_values))))
            chi2(results.x,add_chi2=True)
        except Exception:
            print('Differential evolution fitting failed for event with index '+str(evt_index))
            continue        

np.savetxt("DE_run_0130_uncleaned.txt",sum_values)
np.savetxt("DE_run_0130_time_uncleaned.txt",time_values)