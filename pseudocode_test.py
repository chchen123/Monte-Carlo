#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 14:04:09 2018

@author: chen
"""

import numpy as np
import pytpc
import yaml
from math import pi
import h5py
import logging
import logging.config

logger = logging.getLogger(__name__)

config_path = '/mnt/home/mkuchera/ATTPC/ar40_reqfiles/config_e15503a_runs_105-137.yml'
input_path = '/mnt/research/attpc/data/e15503a/hdf5_cleaned/clean_run_0102.h5'
output_path = '/mnt/research/attpc/data/e15503a/mc_test/run_0102.h5'

run_ID = input_path[-11:-3]

with open(config_path, 'r') as f:
    config = yaml.load(f)

inFile = h5py.File(input_path, 'r')
dataset_name = '/clean'
evt_inFile = inFile[dataset_name]

mcfitter = pytpc.fitting.MCFitter(config)
chi_values = np.empty(shape=(0,0))
evtids = np.empty(shape=(0,0))

def event_iterator(input_evtid_set, output_evtid_set):
    unprocessed_events = input_evtid_set - output_evtid_set
    num_input_evts = len(input_evtid_set)
    num_events_remaining = len(unprocessed_events)
    num_events_finished = len(output_evtid_set)
    if num_events_remaining == 0:
        logger.warning('All events have already been processed.')
        raise StopIteration()
    elif num_events_finished > 0:
        logger.info('Already processed %d events. Continuing from where we left off.', num_events_finished)

    for i in unprocessed_events:
        if i % 100 == 0:
            logger.info('Processed %d / %d events', i, num_input_evts)
        yield i
    else:
        raise StopIteration()

for i in range(len(evt_inFile)):
    if i < 20:
        evtids = np.append(evtids, [i])


with h5py.File(output_path, 'w') as outFile:
    gp = outFile.require_group('monte carlo')
    input_evtid_set = {int(k) for k in evtids}
    num_input_evts = len(input_evtid_set)
    logger.info('Input file contains %d events', num_input_evts)
    output_evtid_set = {int(k) for k in gp}
    
    for evt_index in event_iterator(input_evtid_set, output_evtid_set):
        try:
            xyzs_h5 = evt_inFile[str(evt_index)]
            xyzs = np.array(xyzs_h5)
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