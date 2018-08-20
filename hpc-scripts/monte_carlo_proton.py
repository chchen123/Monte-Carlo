#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 10:06:20 2018
@author: chen
This script fits all the proton events, selected using Jack Taylor's NN model, in a designated run.
(not finished)
"""

import numpy as np
import pytpc
import yaml
import h5py
import argparse
import logging
import logging.config

logger = logging.getLogger(__name__)

#parser = argparse.ArgumentParser(description='A script for the Monte Carlo Fitting')
#parser.add_argument('config', help='Path to a config file')
#parser.add_argument('input', help='path to an input event file')
#parser.add_argument('output', help='the output HDF5 file')
#parser.add_argument('proton', help='path to file containing proton-like evts')
#args = parser.parse_args()

config_path = '/home/chen/ar46/config/config_e15503b_p.yml'
input_path = '/home/chen/ar46/clean_events/clean_run_0131.h5'
output_path = '/home/chen/data/proton_0131.h5'
proton_path = '/home/chen/data/keras-results/pCnoise/proton_events_0131.txt'



run_ID = input_path[-11:-3]

with open(config_path, 'r') as f:
    config = yaml.load(f)

inFile = h5py.File(input_path, 'r')
dataset_name = '/clean'
evt_inFile = inFile[dataset_name]

mcfitter = pytpc.fitting.MCFitter(config)
chi_values = np.empty(shape=(0,0))
proton_evts = np.empty(shape=(0,0))

#def event_iterator(input_evtid_set, output_evtid_set):
#    unprocessed_events = input_evtid_set - output_evtid_set
#    num_input_evts = len(input_evtid_set)
#    num_events_remaining = len(unprocessed_events)
#    num_events_finished = len(output_evtid_set)
#    if num_events_remaining == 0:
#        logger.warning('All events have already been processed.')
#        raise StopIteration()
#    elif num_events_finished > 0:
#        logger.info('Already processed %d events. Continuing from where we left off.', num_events_finished)
#
#    for i in unprocessed_events:
#        if i % 100 == 0:
#            logger.info('Processed %d / %d events', i, num_input_evts)
#        yield i
#    else:
#        raise StopIteration()
#        
with open(proton_path, "r") as output:
    for i in output:
        i = i[:-1]
        proton_evts = np.append(proton_evts, int(i))
        
with h5py.File(output_path, 'w') as outFile:
    gp = outFile.require_group('monte carlo')
#    input_evtid_set = {int(k) for k in evtids}
#    print(input_evtid_set)
#    num_input_evts = len(input_evtid_set)
#    logger.info('Input file contains %d events', num_input_evts)
#    output_evtid_set = {int(k) for k in gp}

#    full_evt_ID = []
#    for i in range(len(evt_inFile)):
#        full_evt_ID.append(i)
#    print(len(full_evt_ID))
    for evt_index in range(len(evt_inFile)):
        try:
            xyzs_h5 = evt_inFile[str(evt_index)]
        except Exception:
            for i in range(len(proton_evts)):
                if proton_evts[i] >= evt_index:
                    proton_evts[i] += 1
        if evt_index in proton_evts:
            print(evt_index)
            try:
                del_list = []
                xyzs_h5 = evt_inFile[str(evt_index)]
                xyzs = np.array(xyzs_h5)
                for i in range(len(xyzs)):
                    if (xyzs[i,6]) > 75.0:
                        del_list.append(i)
                xyzs = np.delete(xyzs,del_list,axis=0)
                xy = xyzs[:, 0:2]
            except Exception:
                logger.exception('Failed to read event with index %d from input', evt_index)
                continue
    
            try:
                xy_C = np.ascontiguousarray(xy, dtype=np.double)
                cx, cy = pytpc.cleaning.hough_circle(xy_C)
            except Exception:
                logger.exception('Cannot find the center of curvature for event with index %d', evt_index)
                continue
            
            try:
                uvw, (cu, cv) = mcfitter.preprocess(xyzs[:,0:5], center=(cx, cy), rotate_pads=False)
            except Exception:
                logger.exception('Failed to preprocess event with index %d', evt_index)
                continue
            
            try:
                mcres, minChis, all_params, good_param_idx = mcfitter.process_event(uvw, cu, cv, return_details=True)
                if np.isnan(mcres['posChi2']) != True:
                    chi_values = np.append(chi_values, [mcres['posChi2']+mcres['enChi2']+mcres['vertChi2']])
            except Exception:
                logger.exception('Monte Carlo fitting failed for event with index %d', evt_index)
                continue        
                
            try:
                dset = gp.create_dataset('{:d}'.format(evt_index), data=chi_values, compression='gzip')
            except Exception:
                logger.exception('Writing to HDF5 failed for event with index %d', evt_index)
                continue
            
            chi_values = np.empty(shape=(0,0))

print('MC fitting complete for '+run_ID)
