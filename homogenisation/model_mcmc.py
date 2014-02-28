# coding: utf-8

""" Infer dem things """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@cam.ast.ac.uk>"

# Standard libraries
import logging
from multiprocessing import cpu_count
from time import time

import matplotlib.pyplot as plt

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


    def log_prior(self, theta):
        """ Returns the logarithmic prior """

        m, b, P, Y, V = theta

        if any([P > 1.0, 0.0 > P, 0.0 >= V]):
            return -np.inf

        log_prior = sum([
            scipy.stats.norm.logpdf(m, 2.25, 0.1),
            scipy.stats.uniform.logpdf(b, 30, 40),
            scipy.stats.uniform.logpdf(P, 0, 1.0)
        ])

        return log_prior


    def log_likelihood(self, theta):
        """ Likelihood function for theta """

        m, b, P, Y, V = theta

        model = m * self.x + b
        foreground = (1 - P) * scipy.stats.norm.pdf(self.y, model, self.y_uncertainties)
        background = P * scipy.stats.norm.pdf(self.y, Y, V)

        likelihood = foreground + background

        return sum(np.log(likelihood))


    def __call__(self, theta):

        blob = list(theta)

        log_prior = self.log_prior(theta)
        log_likelihood = self.log_likelihood(theta)

        blob += [log_prior, log_likelihood]

        if any(~np.isfinite([log_prior, log_likelihood])):
            return (-np.inf, blob)

        log_posterior = log_prior + log_likelihood
        return (log_posterior, blob)



def straight_line(x, y, x_uncertainties=None, y_uncertainties=None,
    walkers=250, burn_in=500, samples=1500, model_outliers=False,
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
    dimensions = 5

    model = LinearModelWithOutliers(x, y, y_uncertainties=y_uncertainties)

    p0 = []
    for i in xrange(walkers):

    	# 2.24, 35.85
        pi = [np.random.normal(2.25, 0.1), np.random.normal(34, 0.1)]

        pi.append(np.random.uniform(0, 1)) # P
        pi.append(np.random.uniform(300,500))#normal(np.mean(y), 30)) # Y
        pi.append(np.random.uniform(0, 120))
        #pi.append(np.random.normal(np.std(y), 3)) # V
        p0.append(pi)

    p0 = np.array(p0)

    #p_init = [2.25, 30, 0.5, np.mean(y), np.std(y)]
    #p0 = emcee.utils.sample_ball(p_init, 0.1 * np.ones(dimensions), walkers)

    sampler = emcee.EnsembleSampler(walkers, dimensions, model, threads=threads)

    # Burn in baby!
    t_init = time()
    lnprob0, state0 = None, None
    mean_acceptance_fractions = np.zeros(burn_in + samples)
    for i, (pos, lnprob, state, blobs) in enumerate(sampler.sample(p0,
        lnprob0=lnprob0, rstate0=state0, iterations=burn_in)):

        t_elapsed = time() - t_init
        t_to_burn_in_completion = (t_elapsed/(i + 1)) * (burn_in - i) if i + 1 != burn_in else 0
        mean_acceptance_fractions[i] = np.mean(sampler.acceptance_fraction)

        print("At sample {0} of {1}. Time to burn-in completion is ~{2:.0f} secs. Mean acceptance fraction: {3:.3f}".format(
            i + 1, burn_in, t_to_burn_in_completion, mean_acceptance_fractions[i]))

    sampler.reset()

    # Sample the sucker!
    t_init = time()
    for j, (pos, lnprob, state, blobs) in enumerate(sampler.sample(p0,
        lnprob0=lnprob0, rstate0=state0, iterations=samples)):

        t_elapsed = time() - t_init
        t_to_sample_completion = (t_elapsed/(j + 1)) * (samples - j) if j + 1 != samples else 0
        mean_acceptance_fractions[i + j] = np.mean(sampler.acceptance_fraction)

        print("At sample iteration {0} of {1}. Time to sample completion is ~{2:.0f} secs. Mean acceptance fraction: {3:.3f}".format(
            j + 1, burn_in, t_to_sample_completion, mean_acceptance_fractions[i + j]))

    # Marginalise!
    blobs = np.array(sampler.blobs).reshape((-1, dimensions + 2))
    #me_m = np.median()

    # Return the most likely parameters over the marginalised distribution
    # Here we will just take the mean
    m, b = np.mean(sampler.flatchain[:, 0]), np.mean(sampler.flatchain[:, 1])

    # Should we actually return quantiles for each parameter?

    return (m, b, sampler, mean_acceptance_fractions, model)



def main():
    ids, x, y, xerrs, yerrs, pxy = np.loadtxt("hogg_data.dat", unpack=True)

    return straight_line(x, y, y_uncertainties=yerrs, model_outliers=True)

if __name__ == "__main__":

    result = main()
    print("m, b", result[:2])

    sampler = result[2]
    model = result[-1]
    chain = sampler.chain[:, -500:, :].reshape((-1, 5))

    import triangle
    labels = ["$m$", "$b$", "$P_B$", "$Y_B$", "$V_B$"]
    fig = triangle.corner(chain,
        labels=labels)
    fig.savefig("emcee.png")
    print("saved to emcee.png")

    plt.close("all")

    data = np.array(sampler.blobs)
    dimensions = data.shape[2]
    walkers = data.shape[1]

    xi = np.arange(data.shape[0])
    new_labels = ["$m$", "$b$", "$P_B$", "$Y_B$", "$V_B$", "$\log{Pr}$", "$\log{L}$"]
    fig = plt.figure()
    for i in xrange(dimensions):
        ax = fig.add_subplot(dimensions, 1, i + 1)
        for j in xrange(walkers):
            d = data[:, j, i]
            ax.plot(xi, d, "k", alpha=0.1)
            #if i >= 5:
            #    ax.set_yscale("log")

        ax.xaxis.set_visible(False)
        ax.set_ylabel(new_labels[i])

    ax.xaxis.set_visible(True)
    ax.set_xlabel("Iteration")

    fig.savefig("iterations.png")
    print("saved to iterations.png")

    #plt.show()
