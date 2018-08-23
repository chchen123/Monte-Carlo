## Optimization Methods for Track Fitting in the Active-Target Time Projection Chamber

This repository contains the code made in 2018-2019 for analyzing global and local optimization methods for the track fitting process. 

## Packages

Following Python packages are required for using the code:
- pytpc
- numpy
- SciPy
- matplotlib
- Keras
- tensorflow

The code is written in Python 3.6.

## Global Track-Fitting Methods

- naive Monte Carlo method
- differential evolution
- basin hopping

## Folders

- **hpc-scripts**

  - Contains shell scripts for hpc job submission 
  - Contains Python files for MC fitting 
- **jupyter notebooks**

  - Contains the code used to analyze different real and simulated proton events
  - Contains the plots of track fitting using different global and local optimization methods
- **proton-classification**

  - Contains codes for using Keras model to classify proton events
