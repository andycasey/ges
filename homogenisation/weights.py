# coding: utf-8

""" Optimal relative weighting of Gaia-ESO Survey data """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@cam.ast.ac.uk>"

# Standard libraries
import logging
import os
from glob import glob
from multiprocessing import cpu_count
from random import shuffle
from time import time

# Third party libraries
import emcee
import matplotlib.pyplot as plt
import numpy as np
import pyfits
import triangle

# Initialise logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_benchmarks(filename):
    """ Loads the Gaia benchmarks and expected stellar parameters as
    a record array """

    data = np.loadtxt(filename, dtype=str)
    max_filenames_len, max_cname_len, max_object_len = (max(map(len, data[:, i])) for i in xrange(3))

    benchmarks = np.core.records.fromarrays(data.T,
        names=["FILENAME", "CNAME", "Object", "TEFF", "LOGG", "MH"],
        formats=[
            "|S{0:.0f}".format(max_filenames_len),
            "|S{0:.0f}".format(max_cname_len),
            "|S{0:.0f}".format(max_object_len),
            "f8", "f8", "f8"])

    return benchmarks


def load_node_results(filename):
    """ Loads the spectroscopic results from a Gaia-ESO Survey node
    as an array """

    image = pyfits.open(filename)
    return image[1].data


def calculate_weighted_value(weights, node_values, node_errors,
    ignore_non_finite_values=True):
    """ Calculate a weighted stellar parameter based on results from 
    multiple Gaia-ESO survey nodes """

    assert len(weights) == len(node_values)
    assert len(node_values) == len(node_errors)

    if ignore_non_finite_values:
        isfinite = np.isfinite(node_values) & np.isfinite(node_errors)

    else:
        isfinite = np.array([True] * len(node_values))

    # Should we be allowing a range of acceptable values through kwargs?
    weighted_value = np.sum((weights * node_values)[isfinite])/sum(isfinite)
    weighted_value_variance = (1/np.sum(1/(node_errors[isfinite]**2))) \
        * (1/(sum(isfinite) - 1)) * np.sum(((node_values[isfinite] - weighted_value)**2)/(node_errors[isfinite])**2)

    weighted_value_stddev = weighted_value_variance**0.5

    return (weighted_value, weighted_value_stddev)


def chi_sq(model, observed, uncertainty):
    """ Calculate the $\chi^2$ difference between the model and an
    observation with uncertainties """

    assert len(model) == len(observed)
    assert len(observed) == len(uncertainty)

    chi_sq_i = ((observed - model)**2)/uncertainty
    chi_sq = np.sum(chi_sq_i)
    return chi_sq


def lnprobfn(weights, benchmarks, data, stellar_parameters):
    
    # Only positive relative weights permitted    
    if np.any(0 > weights):
        return -np.inf

    num_benchmarks = len(benchmarks)

    assert num_benchmarks > 0
    assert len(stellar_parameters) > 0

    total_chi_sq = 0
    for i, stellar_parameter in enumerate(stellar_parameters):

        # Calculate weighted values and uncertainties for all benchmark stars
        weighted_values = np.array([calculate_weighted_value(weights, data[2*i, j, :], data[2*i + 1, j, :]) for j in xrange(num_benchmarks)])
        weighted_values, weighted_uncertainties = weighted_values[:, 0], weighted_values[:, 1]

        total_chi_sq += chi_sq(benchmarks[stellar_parameter], weighted_values, weighted_uncertainties)
 
    return -0.5*total_chi_sq


def in_acceptable_ranges(star, acceptable_ranges):

    if acceptable_ranges is None: return True

    for stellar_parameter, (min_range, max_range) in acceptable_ranges.iteritems():
        if min_range > star[stellar_parameter] or star[stellar_parameter] > max_range:
            return False

    return True


def prepare_data(benchmarks_filename, node_results_filenames,
    stellar_parameters, acceptable_ranges=None):
    """ Loads expected stellar parameters from the Gaia benchmarks file, as well
    as measured stellar parameter from each of the Gaia-ESO Survey nodes """

    num_nodes = len(node_results_filenames)
    repr_node = lambda filename: "_".join(os.path.basename(filename).split("_")[3:]).rstrip(".fits")

    # Don't supply errors as stellar parameters; we will figure it out.
    assert not any([stellar_parameter.startswith("e_") for stellar_parameter in stellar_parameters])
    stellar_parameters_copy = []
    for stellar_parameter in stellar_parameters:
        stellar_parameters_copy.extend([
            stellar_parameter,
            "e_{0}".format(stellar_parameter)
            ])

    # Load the benchmarks
    logger.info("Loading Gaia benchmark stars..")
    benchmarks = load_benchmarks(benchmarks_filename)

    # Get the data from each node
    logger.info("Loading results from Gaia-ESO Survey nodes..")
    nodes = [load_node_results(node_result_filename) for node_result_filename in node_results_filenames]

    for i, node_result_filename in enumerate(node_results_filenames):
        logger.info("{0:15s} node will have a reference weighting parameter $w_{{{1}}}$".format(
            repr_node(node_result_filename), i))

    # Check that every benchmark star has been observed by every node
    logger.info("Checking validity of stellar parameters ({0}) in node results filenames..".format(
        ", ".join(stellar_parameters_copy)))

    node_stellar_parameters = {}
    for stellar_parameter in stellar_parameters_copy:
        node_stellar_parameters[stellar_parameter] = []

    all_snrs_sampled = []
    for benchmark in benchmarks:

        benchmark_results = {}
        
        # Initialise
        for stellar_parameter in stellar_parameters_copy:
            benchmark_results[stellar_parameter] = []

        for node, node_results_filename in zip(nodes, node_results_filenames):
            # Match by FILENAME
            if not benchmark["FILENAME"] in node["FILENAME"]:
                logger.warn("{0} node does not contain the benchmark star {1}".format(
                    repr_node(node_results_filename), benchmark["Object"]))

                for stellar_parameter in stellar_parameters_copy:
                    benchmark_results[stellar_parameter].append(np.nan)

                continue

            # Check for valid entries of stellar parameters
            index = np.where(node["FILENAME"] == benchmark["FILENAME"])[0]
            if len(index) > 1:
                logger.warn("{0} node has more than one entry of the benchmark star {1} (matched by FILENAME = {2})".format(
                    repr_node(node_results_filename), benchmark["Object"], benchmark["FILENAME"]))

            problems = False
            for stellar_parameter in stellar_parameters_copy:

                # Some WG11 nodes use FeH ([Fe/H]) where as some nodes use [M/H]
                # This is annoying.
                if stellar_parameter in ("MH", "FeH"):
                    if np.all(~np.isfinite(node["MH"])):
                        logger.debug("The {0} node has *only* non-finite measurements of 'MH', so we are using 'FeH' instead.".format(
                            repr_node(node_results_filename)))
                        reference_stellar_parameter = "FeH"

                    else:
                        logger.debug("The {0} node has *only* non-finite measurements of 'FeH', so we are using 'MH' instead".format(
                            repr_node(node_results_filename)))
                        reference_stellar_parameter = "MH"

                elif stellar_parameter in ("e_MH", "e_FeH"):
                    if np.all(~np.isfinite(node["e_MH"])):
                        logger.debug("The {0} node has *only* non-finite measurements of 'e_MH', so we are using 'e_FeH' instead.".format(
                            repr_node(node_results_filename)))
                        reference_stellar_parameter = "e_FeH"

                    else:
                        logger.debug("The {0} node has *only* non-finite measurements of 'e_FeH', so we are using 'e_MH' instead".format(
                            repr_node(node_results_filename)))
                        reference_stellar_parameter = "e_MH"

                else:
                    reference_stellar_parameter = stellar_parameter

                if not np.isfinite(node[index][reference_stellar_parameter]):
                    logger.warn("{0} node has a non-finite measurement of {1} for the benchmark star {2}".format(
                        repr_node(node_results_filename), reference_stellar_parameter, benchmark["Object"]))

                    problems = True
                    benchmark_results[stellar_parameter].append(np.nan)

                #elif Bad Flags?

                elif not in_acceptable_ranges(node[index], acceptable_ranges):
                    benchmark_results[stellar_parameter].append(np.nan)

                else:
                    # Build data arrays  
                    benchmark_results[stellar_parameter].append(node[index][reference_stellar_parameter][0])

            if not problems:
                logger.debug("Node {0} states S/N ratio for {1} is {2}".format(
                    repr_node(node_results_filename), benchmark["Object"], node[index]["SNR"][0]))
                all_snrs_sampled.append(node[index]["SNR"][0])

        for stellar_parameter in stellar_parameters_copy:
            node_stellar_parameters[stellar_parameter].append(benchmark_results[stellar_parameter])

    # Arrayify
    data = np.array([node_stellar_parameters[stellar_parameter] for stellar_parameter in stellar_parameters_copy])
    
    return (benchmarks, data, all_snrs_sampled)


def calculate_weights(benchmarks, data, stellar_parameters,
    nwalkers=200, burn=500, sample=1000, threads=None):
    """ Calculates optimal relative weights for each Gaia-ESO node """

    # Don't provide any error stellar parameters, we will figure it out
    assert not any([stellar_parameter.startswith("e_") for stellar_parameter in stellar_parameters])

    # Uniform priors
    logger.info("Initialising priors..")
    lnprob0, state0, ndim = None, None, len(node_results_filenames)
    p0 = [np.random.uniform(low=0.5, high=1.5, size=ndim) for i in xrange(nwalkers)]
    logger.debug("Priors for first walker: {0}".format(p0[0]))

    threads = threads if threads is not None else cpu_count()
    logger.info("Initialising sampler with {0} threads..".format(threads))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobfn,
        args=(benchmarks, data, stellar_parameters), threads=threads)

    # Burn in and sample
    t_init = time()
    total_steps = burn + sample
    mean_acceptance_fractions = np.zeros(total_steps)
    for i, (pos, lnprob, state) in enumerate(sampler.sample(p0,
        lnprob0=lnprob0, rstate0=state0, iterations=total_steps)):
        
        t_elapsed = time() - t_init
        t_to_completion = (t_elapsed/(i + 1)) * (total_steps - i) if i + 1 != total_steps else 0
        mean_acceptance_fractions[i] = np.mean(sampler.acceptance_fraction)

        logger.info("At iteration {0} of {1}. Time to completion is ~{2:.0f} secs. Mean acceptance fraction: {3:.3f}".format(
            i + 1, total_steps, t_to_completion, mean_acceptance_fractions[i]))

    # Show posterior values
    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
    posteriors = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    optimal_weights = np.zeros((ndim, 3))

    logger.info("Posterior quantiles:")
    for i, (posterior_value, positive_stddev, negative_stddev) in enumerate(posteriors):
        optimal_weights[i, :] = [posterior_value, positive_stddev, negative_stddev]
        logger.info("$w_{{{0:2.00f}}}$ = {1:6.3f} +/- ({2:+6.3f}, {3:-6.3f})".format(
            i, posterior_value, positive_stddev, negative_stddev))

    # Calculate benchmark properties using optimal weights
    units = {
        "teff": "K",
        "logg": "dex",
        "feh": "dex",
    }
    num_benchmarks = len(benchmarks)

    for i, stellar_parameter in enumerate(stellar_parameters):

        logger.info("Calculating optimally weighted {0} for benchmark stars..".format(stellar_parameter))
        unit = " " + units[stellar_parameter.lower()] if stellar_parameter.lower() in units.keys() else ""

        # Calculate optimally weighted properties
        optimally_weighted_values = np.array([calculate_weighted_value(optimal_weights[:, 0], data[2*i, j, :], data[2*i + 1, j, :]) for j in xrange(num_benchmarks)])
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


# Do the things
if __name__ == "__main__":

    # Benchmark data
    benchmarks_filename = "data/benchmarks.txt"

    # iDR2.0: ~November 2013
    node_results_filenames = glob("data/iDR2.0/WG11NodeParamsDR2/GES_iDR2_WG11_*.fits")
    # iDR2.1: ~February 2014
    node_results_filenames = glob("data/iDR2.1/GES_iDR2_WG11_*.fits")

    # Clean up the data files: remove 'Recommended' parameter files and exclude
    # the results from node 'ULB' because they have withdrawn???
    node_results_filenames = [filename for filename in node_results_filenames \
        if not filename.endswith("Recommended.fits")]
    #node_results_filenames = [filename for filename in node_results_filenames \
    #    if not filename.endswith("_ULB.fits")]

    # Shuffle the nodes around
    shuffle(node_results_filenames)

    # Specify stellar parameters to optimally weight
    # Only some nodes (IACAIP and Nice) actually specify MH (i.e., [M/H]) where
    # as all other nodes actually specify "FeH", [Fe/H]

    # Here we will specify 'MH' and the code will handle this internally
    stellar_parameters = ["TEFF", "LOGG", "MH"]

    # Loads them data
    benchmarks, data, all_snrs_sampled = prepare_data(benchmarks_filename, node_results_filenames,
        stellar_parameters, acceptable_ranges=None)

    # Seed
    #np.random.seed(888)

    if False:
        # Hammer
        num_nodes = len(node_results_filenames)
        nwalkers, burn, sample = 500, 500, 500
        optimal_weights, sampler, mean_acceptance_fractions = calculate_weights(
            benchmarks, data, stellar_parameters, nwalkers=nwalkers, burn=burn,
            sample=sample)

        # Visualise
        # - Acceptance
        fig = plot_acceptance_fractions(mean_acceptance_fractions, burn=burn)
        fig.savefig("acceptance.png")

        # - All sampled parameters, their $\chi^2$ and $L$ values

        # - Triangle
        samples = sampler.chain[:, burn:, :].reshape((-1, num_nodes))
        fig = triangle.corner(samples, labels=["$w_{{{0}}}$".format(i) for i in xrange(num_nodes)], verbose=True)
        fig.savefig("triangle.png")
