# coding: utf-8

""" Homogenisation of Gaia-ESO Survey data """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@cam.ast.ac.uk>"

# Standard libraries
import logging
import os
from glob import glob
from multiprocessing import cpu_count
from time import time

# Third party libraries
import emcee
import matplotlib.pyplot as plt
import numpy as np
import pyfits
import triangle

# Module imports
from gesio import load_benchmarks, load_node_results, prepare_data


# Initialise logging
logging.basicConfig(filename="homogenisation.log", level=logging.INFO)
logger = logging.getLogger(__name__)






def chi_sq(model, observed, uncertainty, ignore_non_finite_values=True):
    """ Calculate the $\chi^2$ difference between the model and an
    observation with uncertainties """

    assert len(model) == len(observed)
    assert len(observed) == len(uncertainty)

    if ignore_non_finite_values:
        isfinite = np.isfinite(observed) & np.isfinite(uncertainty)

    else:
        isfinite = np.array([True] * len(observed))

    chi_sq_i = ((observed - model)**2)/(uncertainty**2)
    chi_sq = np.sum(chi_sq_i[isfinite])
    return chi_sq




def lnprobfn(theta, benchmarks, node_data, stellar_parameters):
    """ Log-likelihood probability function """

    # theta contains:
    # * N node weights [uniform 0-1]
    # * M stellar parameters [uniform in range of each stellar parameter]
    # * M stellar parameter scale factors [uniform 0-100% of the stellar parameter ranges]

    num_nodes = node_data.shape[2] # N
    num_benchmarks = len(benchmarks) # B
    num_parameters = len(stellar_parameters) #M

    assert num_nodes > 0
    assert num_benchmarks > 0
    assert len(stellar_parameters) > 0

    # Split up theta into something more readable
    alpha = theta[:num_nodes - 1]
    
    # There's one final alpha 
    alpha = np.append(alpha, 1 - np.sum(alpha))

    if np.any(0 > alpha) or np.any(alpha > 1):
        return -np.inf

    gaussian = lambda x, mu, sigma: np.exp((-(x - mu)**2)/(2*sigma**2))

    total_chi_sq = 0
    for j, benchmark in enumerate(benchmarks):

        for i, stellar_parameter in enumerate(stellar_parameters):

            isfinite_parameter = np.isfinite(node_data[2*i, j, :])
            isfinite_variance = np.isfinite(node_data[2*i + 1, j, :])

            weighted_parameter = np.sum(alpha[isfinite_parameter] * node_data[2*i, j, isfinite_parameter])
            weighted_variance = np.sum(alpha[isfinite_variance] * node_data[2*i + 1, j, isfinite_variance])

            chi_sq_i = (benchmark[stellar_parameter] - weighted_parameter)**2/weighted_variance**2


            if np.isfinite(chi_sq_i):
                total_chi_sq += chi_sq_i

    if total_chi_sq == 0:
        return -np.inf

    print("Total", total_chi_sq, "from alpha", alpha)
    return -0.5 * total_chi_sq




def calculate_weights(benchmarks, data, stellar_parameters,
    walkers=200, burn=500, sample=1000, threads=None):
    """ Calculates optimal relative weights for each Gaia-ESO node """

    # Don't provide any error stellar parameters, we will figure it out
    assert not any([stellar_parameter.startswith("e_") for stellar_parameter in stellar_parameters])

    lnprob0, state0 = None, None
    num_nodes, num_parameters = len(node_results_filenames), len(stellar_parameters)

    logger.info("Initialising priors..")
    
    # How many variables do we have?
    # * N node weights [uniform 0-1]
    # * M stellar parameters [uniform in range of each stellar parameter]
    # * M stellar parameter scale factors [uniform 0-100% of the stellar parameter ranges]
    
    p0 = []
    # Calculate the range in stellar parameters just once
    ranges = {}
    for i, stellar_parameter in enumerate(stellar_parameters):
        ranges[stellar_parameter] = [
            np.min(benchmarks[stellar_parameter]),
            np.max(benchmarks[stellar_parameter])
        ]

    for i in xrange(walkers):
        # N node weights [uniform 0 to 1] 
        #pi = list(np.random.uniform(low=0, high=1, size=num_nodes - 1))
        pi = list(np.random.uniform(low=-0.5, high=0.5, size=num_nodes - 1))
        # M stellar parameter offsets [uniform -1, 1]
        #pi.extend(np.random.uniform(low=-1, high=1, size=num_parameters))

        # M stellar parameters [uniform in range of each]
        #for j, stellar_parameter in enumerate(stellar_parameters):
        #    pi.append(np.random.uniform(*ranges[stellar_parameter]))

        # M stellar parameter scale factors 
        # [uniform in 0-100% of the stellar parameter ranges]
        #for j, stellar_parameter in enumerate(stellar_parameters):
        #    pi.append(np.random.uniform(low=0,
        #        high=np.ptp(ranges[stellar_parameter])))
        
        p0.append(sum([], pi))

    num_dimensions = len(p0[0])

    logger.debug("Priors for first walker: {0}".format(p0[0]))

    threads = threads if threads is not None else cpu_count()
    
    logger.info("Initialising sampler with {0} threads..".format(threads))
    sampler = emcee.EnsembleSampler(walkers, num_dimensions, lnprobfn,
        args=(benchmarks, data, stellar_parameters), threads=threads)

    # Burn in and sample
    t_init = time()
    total_steps = burn + sample
    mean_acceptance_fractions = np.zeros(total_steps)
    for i, (pos, lnprob, state) in enumerate(sampler.sample(p0,
        lnprob0=lnprob0, rstate0=state0, iterations=total_steps)):
        
        try:
            t_elapsed = time() - t_init
            t_to_completion = (t_elapsed/(i + 1)) * (total_steps - i) if i + 1 != total_steps else 0
            mean_acceptance_fractions[i] = np.mean(sampler.acceptance_fraction)

            logger.info("At iteration {0} of {1}. Time to completion is ~{2:.0f} secs. Mean acceptance fraction: {3:.3f}".format(
                i + 1, total_steps, t_to_completion, mean_acceptance_fractions[i]))

        except KeyboardInterrupt:
            break

    # Show posterior values
    samples = sampler.chain[:, -sample:, :].reshape((-1, num_dimensions))
    posteriors = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    optimal_weights = np.zeros((num_dimensions, 3))

    logger.info("Posterior quantiles:")
    for i, (posterior_value, positive_stddev, negative_stddev) in enumerate(posteriors):
        optimal_weights[i, :] = [posterior_value, positive_stddev, negative_stddev]
        logger.info("$w_{{{0:2.00f}}}$ = {1:7.3f} +/- ({2:+8.3f}, {3:+8.3f})".format(
            i, posterior_value, positive_stddev, negative_stddev))

    # Calculate benchmark properties using optimal weights
    units = {
        "teff": "K",
        "logg": "dex",
        "feh": "dex",
    }
    num_benchmarks = len(benchmarks)

    """
    for i, stellar_parameter in enumerate(stellar_parameters):

        logger.info("Calculating optimally weighted {0} for benchmark stars..".format(stellar_parameter))
        unit = " " + units[stellar_parameter.lower()] if stellar_parameter.lower() in units.keys() else ""

        # Calculate optimally weighted properties
        optimally_weighted_values = np.array([calculate_weighted_value(optimal_weights[:num_nodes, 0], data[2*i, j, :], data[2*i + 1, j, :]) for j in xrange(num_benchmarks)])
        optimally_weighted_values, optimally_weighted_errors = optimally_weighted_values[:, 0], optimally_weighted_values[:, 1]

        # Print them out
        for benchmark, optimally_weighted_value, optimally_weighted_error in \
            zip(benchmarks, optimally_weighted_values, optimally_weighted_errors):
            logger.info("Benchmark {0} for {1:13s}: {2:5.2f}{5}, optimally weighted value: {3:5.2f}{5} (Delta: {4:+7.2f}{5})".format(
                stellar_parameter, benchmark["Object"], benchmark[stellar_parameter], optimally_weighted_value, 
                optimally_weighted_value - benchmark[stellar_parameter], unit))

        # Provide some general information
        logger.info("Mean offset between benchmark and optimally weighted {0}: {1:5.2f}{2}".format(
            stellar_parameter, np.mean(benchmarks[stellar_parameter] - optimally_weighted_values), unit))
        logger.info("Mean standard deviation between benchmark and optimally weighted {0}: {1:5.2f}{2}".format(
            stellar_parameter, np.std(benchmarks[stellar_parameter] - optimally_weighted_values), unit))
    """

    return (optimal_weights, sampler, mean_acceptance_fractions)


def plot_acceptance_fractions(mean_acceptance_fractions, burn=None):
    """ Plots the mean acceptance fraction for all iterations """

    fig = plt.figure()
    axes = fig.add_subplot(111)
    
    axes.plot(np.arange(len(mean_acceptance_fractions)) + 1, mean_acceptance_fractions, "k", lw=2)
    
    if burn is not None:
        axes.plot([burn, burn], [0, 1], ":", c="#666666")
    
    axes.set_xlim(1, len(mean_acceptance_fractions) + 1)
    normal_limits = [0.25, 0.50]

    if not all((normal_limits[1] >= mean_acceptance_fractions) & (mean_acceptance_fractions >= normal_limits[0])):
        normal_limits = [
            0.95 * np.min(mean_acceptance_fractions),
            1.05 * np.max(mean_acceptance_fractions)
        ]
    axes.set_ylim(normal_limits)
    axes.set_xlabel("Iteration")
    axes.set_ylabel("Mean acceptance fraction")

    return fig


def convolve_posterior_weights(weights_posterior, benchmarks, data, stellar_parameters):
    """Convolves the posterior distribution of optimal weights with the data
    from each node, and returns distributions of the optimally weighted stellar
    parameters."""

    # weights can be a NxM array where N is number of samples and M = len(benchmarks) = len(data)

    assert data.shape[0] == len(stellar_parameters)*2
    assert data.shape[1] == len(benchmarks)
    #assert data.shape[2] == weights_posterior.shape[1]

    # Use *all* the RAM
    stellar_parameter_distributions = np.zeros(map(len, [weights_posterior, benchmarks, stellar_parameters]))

    for i, benchmark in enumerate(benchmarks):
        for j, stellar_parameter in enumerate(stellar_parameters):
            weighted_values, weighted_variances = calculate_weighted_value(
                weights_posterior[:, :-2 * len(stellar_parameters)], data[2*j, i, :], data[2*j+1, i, :])
            stellar_parameter_distributions[:, i, j] = weighted_values

    return stellar_parameter_distributions



# Do the things
if __name__ == "__main__":

    # Data files
    stellar_parameters = ["TEFF", "LOGG", "MH"]
    benchmarks_filename = "data/benchmarks.txt"

    node_results_filenames = glob("data/iDR2.1/GES_iDR2_WG11_*.fits")

    # Loads them data
    benchmarks, node_data = prepare_data(benchmarks_filename, node_results_filenames,
        stellar_parameters)

    # Use only the sun
    #index = np.where(benchmarks["Object"] == "Sun")

    #solar = benchmarks[index]
    #num_nodes = len(node_results_filenames)
    #node_solar_data = node_data[:, index, :]


    walkers, burn, sample = 500, 250, 250
    optimal_weights, sampler, mean_acceptance_fractions = calculate_weights(
        benchmarks, node_data, stellar_parameters, walkers=walkers, burn=burn,
        sample=sample)

    # Get samples after burn-in 
    samples = sampler.chain[:, burn:, :].reshape((-1, sampler.chain.shape[-1]))

    # Convolve with the data to obtain optimal stellar parameter distributions
    #distribution = convolve_posterior_weights(samples, benchmarks, data, stellar_parameters)

    # Visualise
    # - Acceptance
    fig_acceptance = plot_acceptance_fractions(mean_acceptance_fractions, burn=burn)
    fig_acceptance.savefig("acceptance.png")

    # - All sampled parameters, their $\chi^2$ and $L$ values

    # - Triangle
    fig_weights = triangle.corner(samples,
        verbose=True)
    fig_weights.savefig("weights.png")

    # - Visualise optimally weighted stellar parameter distributions for each star
    parameter_extents = {"TEFF": 200, "LOGG": 0.5, "MH": 0.2}
    for i, benchmark in enumerate(benchmarks):
        plt.close("all")

        extents = []
        truths = [benchmark[stellar_parameter] for stellar_parameter in stellar_parameters]
        for truth, stellar_parameter in zip(truths, stellar_parameters):
            extents.append((
                truth - parameter_extents[stellar_parameter],
                truth + parameter_extents[stellar_parameter]
                ))

        #fig_parameters = triangle.corner(distribution[:, i, :],
        #    labels=stellar_parameters, truths=truths)#, extents=extents)
        #fig_parameters.savefig("benchmark_{0}.png".format(benchmark["Object"]))
