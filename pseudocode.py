#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 10:36:19 2018

@author: chen
"""

#pseudo codes
#imports
#/mnt/research/attpc/data/e15503a/hdf5_cleaned has runs from 0066 to 0137, run numbers are not continuous
#config reqfiles for Argon 40: /home/ATTPC/ar40_reqfiles has runs from 54 to 137
# 54-63, 66-91, 92-104, 105-137, and config_e15503a.yml
#for loop: for i in (range from 0066 to 0137)
#full = h5py.File('/home/chen/Real Data/clean_run_i.h5', 'r')
#if i is in xx-xx: with open('/home/ATTPC/ar40_reqfiles/config_e15503a_runs_xx-xx.yml', 'r') as f:
#    config = yaml.load(f)
# number of events = len(full)



#################################################################################
#dictionary = {}

#for each run (i):
#   dictionary['run_xxxx'] = []
#   full = h5py.File('/home/chen/Real Data/clean_run_xxxx.h5', 'r')
#   if i is in xx-xx: 
#       with open('/home/ATTPC/ar40_reqfiles/config_e15503a_runs_xx-xx.yml', 'r') as f:
#    config = yaml.load(f)

#    for i in range(len(full)):
#        evt_ID = i +1 
#        dataset_name = '/clean'
#        evt_full = full[dataset_name]
#        xyzs_h5 = evt_full[str(evt_ID)]
    
#        xyzs = np.array(xyzs_h5)
        
#        xy = xyzs[:, 0:2]
#        xy_C = np.ascontiguousarray(xy, dtype=np.double)
#        cx, cy = pytpc.cleaning.hough_circle(xy_C)
    
#        uvw, (cu, cv) = mcfitter.preprocess(xyzs[:,0:5], center=(cx, cy), rotate_pads=False) # get calibrated set of data
#        uvw_values = uvw.values #transform pd file to arrays
#        mcres, minChis, all_params, good_param_idx = mcfitter.process_event(uvw, cu, cv, return_details=True)
#        if np.isnan(mcres['posChi2']) != True:
#            dictionary['run_xxxx'].append(mcres['posChi2']+mcres['enChi2']+mcres['vertChi2'])


    