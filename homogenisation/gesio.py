# coding: utf-8

""" Input/Output functions for Gaia-ESO Survey data """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@cam.ast.ac.uk>"

# Standard libraries
import logging
import os
from glob import glob

# Third party libraries
import numpy as np
import pyfits

logger = logging.getLogger(__name__)

__all__ = ["load_benchmarks", "load_node_results", "prepare_data"]


def in_acceptable_ranges(star, acceptable_ranges):
    """ Returns whether the measured stellar parameters are within
    acceptable ranges """

    if acceptable_ranges is None: return True

    for stellar_parameter, (min_range, max_range) in acceptable_ranges.iteritems():
        if (min_range is not None and min_range > star[stellar_parameter][0]) \
        or (max_range is not None and star[stellar_parameter][0] > max_range):
            return False

    return True


def load_benchmarks(filename):
    """ Loads the Gaia benchmarks and expected stellar parameters as
    a record array """

    data = np.loadtxt(filename, dtype=str)
    max_filenames_len, max_cname_len, max_object_len = (max(map(len, data[:, i])) for i in xrange(3))

    benchmarks = np.core.records.fromarrays(data.T,
        names=["FILENAME", "CNAME", "Object", "TEFF",
            "e_TEFF", "LOGG", "e_LOGG", "MH"],
        formats=[
            "|S{0:.0f}".format(max_filenames_len),
            "|S{0:.0f}".format(max_cname_len),
            "|S{0:.0f}".format(max_object_len),
            "f8", "f8", "f8", "f8", "f8"])

    return benchmarks


def load_node_results(filename):
    """ Loads the spectroscopic results from a Gaia-ESO Survey node
    as an array """

    with pyfits.open(filename) as image:
        data = image[1].data
    return data


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

    # Should we remove any nodes that have absolutely no data to contribute?
    return (benchmarks, data)
