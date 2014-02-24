# coding: utf-8
 
""" Lyons et al 1988 """
 
from __future__ import division, print_function
 
__author__ = "Andy Casey <arc@cam.ast.ac.uk>"
 
# Standard libraries
import logging
import os
from glob import glob
from time import time
 
# Third party libraries
import emcee
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import pyfits
import triangle
 

def get_weights(covariance_matrix):
	"""Determine weights for Best Linear Unbiased
	Estimate of parameters"""

	n = covariance_matrix.shape[0]

	def func(weights, covariance_matrix):
		S = 0
		n, m = covariance_matrix.shape
		for i in xrange(n):
			for j in xrange(m):
				S += weights[i] * weights[j] * covariance_matrix[i, j]
		return S

	return scipy.optimize.fmin_slsqp(func, x0=np.array([1.0/n] * n),
		args=(covariance_matrix, ), eqcons=[lambda x, c_m: np.sum(x) - 1],
		disp=False)

 
def blue(measurements, uncertainties, N=1000, full_output=False):
	""" Calculate the Best Linear Unbiased Estimate of a parameter
	from all measurements and uncertainties."""

	isfinite = np.multiply(*map(np.isfinite, [measurements, uncertainties]))
	num_finite_measurements = sum(isfinite)

	if num_finite_measurements == 0:
		raise ValueError("no finite measurements")

	# Discard non-finite measurements
	measurements = measurements[isfinite]
	uncertainties = uncertainties[isfinite]
	 
	covariance_matrix = np.zeros((num_finite_measurements, num_finite_measurements))
	 
	for i in xrange(num_finite_measurements):
		for j in xrange(num_finite_measurements):
			
			if i == j:
				covariance_matrix[i, i] = uncertainties[i]**2
			 
			else:
				# Simulate N number of MCMC simulations and
				# approximate the distributions
				distribution_i = np.random.normal(
					loc=measurements[i], scale=uncertainties[i],
					size=N)
				distribution_j = np.random.normal(
					loc=measurements[j], scale=uncertainties[j],
					size=N)

				mean_i, mean_j = map(np.mean, [distribution_i, distribution_j])
				covariance = np.sum((distribution_i - mean_i) * (distribution_j - mean_j))/N
	 			covariance_matrix[j, i] = covariance
	 			covariance_matrix[i, j] = covariance

	# Calculate the weights, the BLUE, and the variance
	weights = get_weights(covariance_matrix)
	m_average = sum(weights * measurements)
	m_variance = 0
	for i in xrange(num_finite_measurements):
		for j in xrange(num_finite_measurements):
			m_variance += weights[i] * weights[j] * covariance_matrix[i, j]
	m_variance = m_variance**0.5

	if full_output:
		return m_average, m_variance, weights

	return m_average, m_variance
