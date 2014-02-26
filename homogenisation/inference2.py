# coding: utf-8
 
""" Infer dem things """
 
from __future__ import division, print_function
 
__author__ = "Andy Casey <arc@cam.ast.ac.uk>"

# Standard libraries
import logging
from multiprocessing import cpu_count
from time import time

# Third party libraries
import emcee
import numpy as np
import scipy.stats

# Initialise logging
logging.basicConfig(filename="inference.log", level=logging.DEBUG)
logger = logging.getLogger(__name__)

__all__ = ["straight_line"]


class LinearModelWithOutliers(object):
    """ A linear model that includes modelling of outliers """

    def __init__(self, x, y, y_uncertainties):
        self.x, self.y, self.y_uncertainties = x, y, y_uncertainties


    def _unpack_theta(self, theta):
    	""" Unpack theta into a more legible representation of parameters """

        N = (len(theta) - 2)/3
        m, b = theta[:2]
        P, Y, V = [theta[2 + i*N:2 + (i + 1)*N] for i in xrange(3)]

        return m, b, P, Y, V


    def log_prior(self, theta):
        """ Returns the logarithmic prior """

        m, b, P, Y, V = self._unpack_theta(theta)
        
        if any(map(any, [P > 1, 0 > P, 0 >= V])):
            return -np.inf 

        log_prior = 1.0/Y

        return sum(log_prior)
        


    def log_likelihood(self, theta):
        """ Likelihood function for theta """

        m, b, P, Y, V = self._unpack_theta(theta)

        model = m * self.x + b
        foreground = (1.0 - P) * np.exp(-0.5 * (self.y - model)**2/self.y_uncertainties**2) / np.sqrt(2.0 * np.pi *  self.y_uncertainties**2)
        background = P * np.exp(-0.5 * (self.y - Y)**2/(V + self.y_uncertainties)**2) / np.sqrt(2.0 * np.pi * (V + self.y_uncertainties**2))
        
        return np.sum(np.log(foreground + background))


    def log_posterior(self, theta):

        log_prior = self.log_prior(theta)
        if not np.isfinite(log_prior):
            return log_prior

        log_likelihood = self.log_likelihood(theta)
        
        return log_likelihood


    def __call__(self, theta):
        return self.log_posterior(theta)



def straight_line(x, y, x_uncertainties=None, y_uncertainties=None,
    walkers=250, burn_in=500, samples=500, model_outliers=False,
    threads=None):
    """ Model a straight line """

    # Check data shapes
    x, y = map(np.array, (x, y))

    if x.shape != y.shape:
        raise ValueError("x and y data shapes must be equal")

    if len(x.shape) != 1:
        raise ValueError("x data must be a 1-row vector")

    if len(y.shape) != 1:
        raise ValueError("y data must be a 1-row vector")

    if y_uncertainties is not None:
        y_uncertainties = np.array(y_uncertainties)
        if y_uncertainties.shape != y.shape:
            raise ValueError("shape of y data and y uncertainties must be equal")

    # Sry Gooby
    if x_uncertainties is not None or y_uncertainties is None or \
    not model_outliers:
        raise NotImplementedError("sry, not general enough yet")

    # Use maximum available threads if none specified
    threads = threads if threads is not None else cpu_count()
    
    N = len(x)
    dimensions = 3 * N + 2

    model = LinearModelWithOutliers(x, y, y_uncertainties=y_uncertainties)

    p0 = []
    for i in xrange(walkers):
    	# 2.24, 35.85
        pi = [np.random.normal(1, 1), np.random.normal(0, 30)]
        pi.extend(np.random.uniform(0, 1, size=N)) # P
        pi.extend(np.random.normal(np.mean(y), 0.1, size=N)) # V
        pi.extend(np.exp(np.random.normal(np.std(y), 0.1, size=N))) # Y

        p0.append(pi)

    p_init = [2.25, 30]
    p_init.extend([0.5]*N)
    p_init.extend([np.mean(y)]*N)
    p_init.extend([np.std(y)]*N)

    p0 = emcee.utils.sample_ball(p_init, 0.1 * np.ones(dimensions), walkers)

    sampler = emcee.EnsembleSampler(walkers, dimensions, model, threads=threads)

    # Burn in baby!
    t_init = time()
    lnprob0, state0 = None, None
    mean_acceptance_fractions = np.zeros(burn_in + samples)
    for i, (pos, lnprob, state) in enumerate(sampler.sample(p0,
        lnprob0=lnprob0, rstate0=state0, iterations=burn_in)):
        
        t_elapsed = time() - t_init
        t_to_burn_in_completion = (t_elapsed/(i + 1)) * (burn_in - i) if i + 1 != burn_in else 0
        mean_acceptance_fractions[i] = np.mean(sampler.acceptance_fraction)

        print("At burn in iteration {0} of {1}. Time to burn-in completion is ~{2:.0f} secs. Mean acceptance fraction: {3:.3f}".format(
            i + 1, burn_in, t_to_burn_in_completion, mean_acceptance_fractions[i]))

    sampler.reset()

    # Sample the sucker!
    t_init = time()
    for j, (pos, lnprob, state) in enumerate(sampler.sample(p0,
        lnprob0=lnprob0, rstate0=state0, iterations=samples)):
        
        t_elapsed = time() - t_init
        t_to_sample_completion = (t_elapsed/(j + 1)) * (samples - j) if j + 1 != samples else 0
        mean_acceptance_fractions[i + j] = np.mean(sampler.acceptance_fraction)

        print("At sample iteration {0} of {1}. Time to sample completion is ~{2:.0f} secs. Mean acceptance fraction: {3:.3f}".format(
            j + 1, burn_in, t_to_sample_completion, mean_acceptance_fractions[i + j]))

    # Return the most likely parameters over the marginalised distribution
    # Here we will just take the mean
    m, b = np.mean(sampler.flatchain[:, 0]), np.mean(sampler.flatchain[:, 1])

    # Should we actually return quantiles for each parameter?

    return (m, b, sampler, mean_acceptance_fractions)



def main():
    ids, x, y, xerrs, yerrs, pxy = np.loadtxt("hogg_data.dat", unpack=True)

    return straight_line(x, y, y_uncertainties=yerrs, model_outliers=True)

if __name__ == "__main__":
    result = main()