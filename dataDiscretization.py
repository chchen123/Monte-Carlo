#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataDiscretization.py
=====================
This module contains functions for discretizing 3D point cloud data produced by
the Active-Target Time Projection Chamber.
"""
import math
import numpy as np
import scipy as sp

import pytpc

from pytpc.hdfdata import HDFDataFile
import yaml
import h5py

DETECTOR_LENGTH = 1250.0
DETECTOR_RADIUS = 275.0

def discretizeCylindrical(xyz, z_disc, radial_disc, angular_disc):
    """(Inefficiently) Discretizes AT-TPC point cloud data using a cylindrical
    geometry. We found that this strategy forces an inappropriate geometry on
    our data.
    Parameters
    ----------
    xyz          : point cloud data with shape (n,5) where n is the number of traces
    z_disc       : number of slices in z
    radial_disc  : number of radial slices/concentric circles
    angular_disc : number of angular wedge slices
    Returns
    -------
    The discretized data in an array of shape (1, z_disc*radial_disc*angular_disc)
    """
    #calculate dimensional increments
    z_inc = DETECTOR_LENGTH/z_disc
    radial_inc = DETECTOR_RADIUS/radial_disc
    angular_inc = (2*math.pi)/angular_disc

    #create slice boundary arrays
    z_slices = np.arange(DETECTOR_LENGTH, 0.0-z_inc, -z_inc)
    radial_slices = np.arange(DETECTOR_RADIUS, 0.0-radial_inc, -radial_inc)
    angular_slices = np.arange(-math.pi, math.pi+angular_inc, angular_inc)


    discretized_data = np.zeros((1,z_disc*radial_disc*angular_disc))
    discretized_xyz = np.zeros([xyz.shape[0],xyz.shape[1]])
    bucket_num = 0
    num_pts = 0

    for i in range(len(z_slices)-1):
        for j in range(len(radial_slices)-1):
            for k in range(len(angular_slices)-1):
                for point in xyz:
                    if ((z_slices[i] > point[2] > z_slices[i+1]) and
                    (radial_slices[j] > math.sqrt(point[0]**2+ point[1]**2) > radial_slices[j+1]) and
                    (angular_slices[k] < math.atan2(point[1], point[0])  < angular_slices[k+1])):

                        discretized_data[0][bucket_num] = 1
                        num_pts += 1

            bucket_num += 1

    return discretized_data


def discretizeGrid(xyz, x_disc, y_disc, z_disc):
    """Discretizes AT-TPC point cloud data using a grid geometry based on
    whether or not a point exists in a given rectangular bucket.
    Parameters
    ----------
    xyz    : point cloud data with shape (n,5) where n is the number of traces
    x_disc : number of slices in x
    y disc : number of slices in y
    z_disc : number of slices in z
    Returns
    -------
    The discretized data in a csr sparse matrix of shape (1, x_disc*y_disc*z_disc)
    """

    #calculate desired discretization resolution
    discElements = x_disc*y_disc*z_disc

    #calculate dimensional increments
    x_inc = (2*DETECTOR_RADIUS)/x_disc
    y_inc = (2*DETECTOR_RADIUS)/y_disc
    z_inc = DETECTOR_LENGTH/z_disc

    buckets = []

    for point in xyz:
        x_bucket = math.floor(((point[0]+DETECTOR_RADIUS)/(2*DETECTOR_RADIUS))*x_disc)
        y_bucket = math.floor(((point[1]+DETECTOR_RADIUS)/(2*DETECTOR_RADIUS))*y_disc)
        z_bucket = math.floor((point[2]/DETECTOR_LENGTH)*z_disc)

        bucket_num = z_bucket*x_disc*y_disc + x_bucket + y_bucket*x_disc
        buckets.append(bucket_num)

    cols = np.unique(buckets)
    rows = np.zeros(len(cols))
    data = np.ones(len(cols))

    discretized_data = sp.sparse.csr_matrix((data, (rows, cols)), shape=(1, discElements))
    return discretized_data


def discretizeGridCharge(xyz, x_disc, y_disc, z_disc):
    """Discretizes AT-TPC point cloud data using a grid geometry by totalling
    charge of hits in a given rectangular bucket.
    Parameters
    ----------
    xyz    : point cloud data with shape (n,5) where n is the number of traces
    x_disc : number of slices in x
    y disc : number of slices in y
    z_disc : number of slices in z
    Returns
    -------
    The discretized data in a csr sparse matrix of shape (1, x_disc*y_disc*z_disc)
    """

    #calculate desired discreuniform_param_generatortization resolution
    discElements = x_disc*y_disc*z_disc

    #calculate dimensional increments
    x_inc = (2*DETECTOR_RADIUS)/x_disc

    y_inc = (2*DETECTOR_RADIUS)/y_disc
    z_inc = DETECTOR_LENGTH/z_disc

    buckets = []
    charges = []

    for point in xyz:
        #check that z-coordinate of point is in appropriate range
        if (point[2] > DETECTOR_LENGTH):
            continue

        x_bucket = math.floor(((point[0]+DETECTOR_RADIUS)/(2*DETECTOR_RADIUS))*x_disc)
        y_bucket = math.floor(((point[1]+DETECTOR_RADIUS)/(2*DETECTOR_RADIUS))*y_disc)
        z_bucket = math.floor((point[2]/DETECTOR_LENGTH)*z_disc)

        bucket_num = z_bucket*x_disc*y_disc + x_bucket + y_bucket*x_disc

        buckets.append(bucket_num)
        #scaling by factor of 1000
        charges.append(point[3]/1000)


    cols = buckets
    rows = np.zeros(len(cols))
    data = charges

    #automatically sums charge values for data occuring at the (row, col)
    discretized_data_sparse_CHARGE  = sp.sparse.csr_matrix((data, (rows, cols)), shape=(1, discElements))
    return discretized_data_sparse_CHARGE


def addNoise(cleanxyz):
    """Adds random noise to an undiscretized Event object.
    Parameters
    ----------
    cleanxyz : point cloud data with shape (n,5)
    Returns
    -------
    Point cloud data augmented with random noise
    """
    num_noisepts = np.random.randint(20, 300,)

    #generate x and y based on random pad numbers
    paddresses = np.random.randint(0,10240, (num_noisepts, 1))
    pads = pytpc.generate_pad_plane()
    pcenters = pads.mean(1)
    xys = pcenters[paddresses].reshape(num_noisepts, 2)

    #z and charge values are generated randomly in realistic ranges
    zs = np.random.uniform(0, DETECTOR_LENGTH, (num_noisepts, 1))
    charges = np.random.uniform(1, 4000, (num_noisepts, 1))

    noise_mat = np.hstack((xys, zs, charges))
    return np.vstack((cleanxyz, noise_mat))


def createNoiseEvents(num_evts, x_disc, y_disc, z_disc, charge):
    """Creates discretized random noise events.
    Parameters
    ----------
    num_evts : the number of desired noise events
    x_disc  : number of slices in x
    y disc  : number of slices in y
    z_disc  : number of slices in z
    Returns
    -------
    A numpy array of shape (num_evts, x_disc*y_disc*z_disc).
    """
    discElements = x_disc*y_disc*z_disc
    discEvts = []

    for evt_id in range(num_evts):
        empty_evt = np.empty([1,4])
        noise_evt = addNoise(empty_evt)

        if (charge == True):
            discEvts.append(discretizeGridCharge(noise_evt, x_disc, y_disc, z_disc))
        else:
            discEvts.append(discretizeGrid(noise_evt, x_disc, y_disc, z_disc))
        if (evt_id%1000 == 0):
            print("Discretized event " + str(evt_id))

    discretized_data = sp.sparse.vstack(discEvts, format='csr')
    print("Data discretization complete.")
    return discretized_data


def bulkDiscretize(hdfPath, x_disc, y_disc, z_disc, charge, noise):
    """Discretizes all events in an HDF5 file using a grid geometry.
    Parameters
    ----------
    hdfPath : the system path to the hdf5 file to be
    x_disc  : number of slices in x
    y disc  : number of slices in y
    z_disc  : number of slices in z
    charge  : boolean variable denoting whether or not charge will be included
              in the discretization
    noise   : boolean variable to add noise to (simulated) data
    Returns
    -------
    A numpy array of shape (n, x_disc*y_disc*z_disc) where n is the number of
    events in the provided hdf5 file.
    """

    discElements = x_disc*y_disc*z_disc
    discEvts = []

    with h5py.File(hdfPath, 'r') as f:
        dataset_name = '/clean'
        evt_full = f[dataset_name]
        n_evts = len(evt_full)

        for evt_id in range(n_evts):
            try:
                xyzs_h5 = evt_full[str(evt_id)]
                curxyz = np.array(xyzs_h5)
                if (noise == True):
                    curxyz = addNoise(curxyz)
    
                if (charge == True):
                    discEvts.append(discretizeGridCharge(curxyz, x_disc, y_disc, z_disc))
                else:
                    discEvts.append(discretizeGrid(curxyz, x_disc, y_disc, z_disc))
                if (evt_id%1000 == 0):
                    print("Discretized event " + str(evt_id))
            except Exception:
                print('event does not exist', str(evt_id))
                continue
#            evt_id += 1

    discretized_data = sp.sparse.vstack(discEvts, format='csr')
    print("Data discretization complete.")
    return discretized_data