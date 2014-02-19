# coding: utf-8

""" Bootstrap the Gaia-ESO Survey benchmark sample to investigate the volatility
in optimal weights """


from __future__ import division, print_function

__author__ = "Andy Casey <arc@cam.ast.ac.uk>"

# Standard libraries
import logging
import os
from glob import glob
from random import shuffle
from time import time

# Third party libraries
import emcee
import matplotlib.pyplot as plt
import numpy as np
import pyfits
import triangle

# Module imports
import weights

# Initialise logging
logging.basicConfig(filename="bootstrap.log", level=logging.INFO)
logger = logging.getLogger(__name__)


# CODE 0  [DONE]
#========
# Convolve posterior weighted distributions with measurements from different
# nodes to obtain a temperature and surface gravity distribution, and obtain
# an uncertainty in temperature and surface gravity.


# TEST 0  
#========
# Convolve weighted posterior distributions to obtain stellar parameters and
# their uncertainties [DONE]

# - How do our calculated uncertainties compare to all the individual node
#   uncertainties? 

# - How do our stellar parameters compare to the benchmark stars, and to the
#   individual nodes?



# TEST 1
#========
# Four fake nodes with fake benchmark data:
# 1: low accuracy, realistic errors
# 2: high accuracy, unrealistically small errors
# 3: high accuracy, realistic errors
# 4: low accuracy, unrealistically small errors

# Randomly throw away 22% of the results
# Run 1,000 simulations. Which faux node performed the "best"?


# TEST 1.5
#==========
# Create weights using 2 random nodes, then 3 random nodes, then 4 random nodes,
# etc and investigate the optimally calculated stellar parameters.

# - Do we get better or worse with increasing nodes?



# TEST 2
#========
# Create 10,000 samples of 23 benchmark stars
# Keep population statistics on temperature, logg, etc
# Determine optimal weighting for all samples

# For a specific subset of samples (e.g. where Metal-Poor stars dominated)
# how did the weightings change?

# - How did they change for MRG, MPG, MRD, MPD?

# - Is there some optimal mapping we can do to interpolate between distributions
#   depending on the actual measured parameters by all the nodes? In essence, is
#   there some way to go: "ah, the teff and logg are X and Y, so in this region 
#   the optimal weights are..."



# TEST 3
#========
# Create 23 (num of benchmark stars) tests where:
# - Each benchmark star is removed, and we use all others
#   to come up with optimal weighting

# - Calculate best parameters for the removed star with
#   the weights from all other stars. 

# - Should all benchmark stars be kept in?
# - Do our partial samples with weights predict good parameters?



# TEST 4
#========
# Apply optimal weights to all stars in M67.

# - How do they change?

# - Are our new values better than anyone else'? Are they worse?

# - Do they follow an isochrone for M67?


# TEST 5
#========
# Repeat test 4 with all clusters available.


