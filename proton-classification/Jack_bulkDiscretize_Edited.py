#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bulkDiscretize_p.py
===================
Author: Jack Taylor
Bulk discretization of hdf5 formatted point cloud data.
"""
import sys
sys.path.insert(0, '../modules/')
import dataDiscretization as dd
import scipy as sp

input_dir = '/home/chen/ar46/clean_events/clean_run_0131.h5'

#Whether or not we want to sum charge and add noise during discretization
CHARGE = True
NOISE = False #Chen: changed noise to False

data = dd.bulkDiscretize(input_dir, 20, 20, 20, CHARGE, NOISE)
sp.sparse.save_npz('/home/chen/data/tilt/20x20x20/run_0131.npz', data)

print(data.shape)
