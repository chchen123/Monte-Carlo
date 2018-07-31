#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bulkDiscretize_RealData.py
==========================
Script to discretize labeled events from real AT-TPC h5 datafiles. Labels should
be stored in .csv files and follow the same labeling scheme as those provided.
Here we discretize and save protons, carbon, and junk events separately for each
run.
"""
import sys
sys.path.insert(0, '../modules/')
import dataDiscretization as dd
import scipy as sp
import pandas as pd

import pytpc

data_path = '/home/chen/data/real/'
disc_path = data_path + '50x50x50/'
runs = ['0130', '0210']

x_disc = 50
y_disc = 50
z_disc = 50

for run in runs:
    data = pytpc.HDFDataFile(data_path + "run_" + run + ".h5", 'r')
    labels = pd.read_csv(data_path + "run_" + run + "_labels.csv", sep=',')
    print("Successfully loaded data and labels for run " + str(run) + ".")

    #discretize proton events
    p_indices = labels.loc[(labels['label'] == 'p')]['evt_id'].values
    p_discEvts = []

    for evt_id in p_indices:
        curEvt = data[evt_id]
        curxyz = curEvt.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False, baseline_correction=False, cg_times=False)
        p_discEvts.append(dd.discretizeGridCharge(curxyz, x_disc, y_disc, z_disc))
        print("Discretized run_" + run + " proton event " + str(evt_id))

    p_data = sp.sparse.vstack(p_discEvts, format='csr')
    sp.sparse.save_npz(disc_path + 'run_' + run + '_pDisc.npz', p_data)

    #discretize carbon events
    C_indices = labels.loc[(labels['label'] == 'c')]['evt_id'].values
    C_discEvts = []

    for evt_id in C_indices:
        curEvt = data[evt_id]
        curxyz = curEvt.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False, baseline_correction=False, cg_times=False)
        C_discEvts.append(dd.discretizeGridCharge(curxyz, x_disc, y_disc, z_disc))
        print("Discretized run_" + run + " Carbon event " + str(evt_id))

    C_data = sp.sparse.vstack(C_discEvts, format='csr')
    sp.sparse.save_npz(disc_path + 'run_' + run + '_CDisc.npz', C_data)

    #discretize junk events
    junk_indices = labels.loc[(labels['label'] == 'j')]['evt_id'].values
    junk_discEvts = []

    for evt_id in junk_indices:
        curEvt = data[evt_id]
        curxyz = curEvt.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False, baseline_correction=False, cg_times=False)
        junk_discEvts.append(dd.discretizeGridCharge(curxyz, x_disc, y_disc, z_disc))
        print("Discretized run_" + run + " junk event " + str(evt_id))

    junk_data = sp.sparse.vstack(junk_discEvts, format='csr')
    sp.sparse.save_npz(disc_path + 'run_' + run + '_junkDisc.npz', junk_data)

print("Discretization complete.")