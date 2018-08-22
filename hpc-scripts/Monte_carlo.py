#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 10:36:19 2018
@author: chen
This Python script fits each event of a designated run using the naive Monte Carlo method.
Expected to be called within a Bash script.
"""

import numpy as np
import pytpc
import yaml
from math import pi
import h5py
import argparse
import logging
import logging.config

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='A script for the Monte Carlo Fitting')
parser.add_argument('config', help='Path to a config file')
parser.add_argument('input', help='path to an input event file')
parser.add_argument('output', help='the output HDF5 file')
args = parser.parse_args()

run_ID = args.input[-11:-3]
DETECTOR_LENGTH = 1250.0

with open(args.config, 'r') as f:
    config = yaml.load(f)

inFile = h5py.File(args.input, 'r')
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
    evtids = np.append(evtids, [i])
        
with h5py.File(args.output, 'w') as outFile:
    gp = outFile.require_group('monte carlo')
    input_evtid_set = {int(k) for k in evtids}
    num_input_evts = len(input_evtid_set)
    logger.info('Input file contains %d events', num_input_evts)
    output_evtid_set = {int(k) for k in gp}
    
    for evt_index in event_iterator(input_evtid_set, output_evtid_set):
        try:
            new_xyzs = []
            xyzs_h5 = evt_inFile[str(evt_index)]
            xyzs = np.array(xyzs_h5)
        except Exception:
            logger.exception('Failed to read event with index %d from input', evt_index)
            continue
        
        try:
            for point in xyzs:
                if (point[2] > DETECTOR_LENGTH):
                    raise ValueError('event is not physical')
        except Exception:
                logger.exception('Event index %d deleted: non-physical evet', evt_index)
                continue
        try:
            xy = xyzs[:, 0:2]
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