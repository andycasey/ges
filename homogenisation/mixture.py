# coding: utf-8

""" Model stellar parameter results from individual Gaia-ESO Survey nodes
    and evaluate how they compare to non-spectroscopic measurements """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@cam.ast.ac.uk>"

# Standard libraries
import logging
import os

# Third party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystan



def compile_model(filename):
    """ Read a STAN model file and return the compiled model """

    model_path = os.path.abspath(filename)
    with open(model_path, "r") as fp:
        model_code = fp.read()
    return pystan.StanModel(model_code=model_code)


# Load the benchmark data

# Load a node data

    # For each stellar parameter, fit a linear relationship

    # Plot the results for each node. Which are outliers?



def main():
    """Do the things"""

    # Check if we have already loaded the data
    global benchmarks, node_data, stellar_parameters, node_results_filenames

    try: benchmarks
    except NameError:
        logger.info("Loading data..")
        node_results_filenames = glob("data/iDR2.1/GES_iDR2_WG11_*.fits")
        remove_nodes = ("Recommended", )
        node_results_filenames = [filename for filename in node_results_filenames \
            if "_".join(os.path.basename(filename).split("_")[3:]).rstrip(".fits")\
            not in remove_nodes]

        # Load the data
        stellar_parameters = ("TEFF", "LOGG")
        benchmarks, node_data = prepare_data("data/benchmarks.txt",
            node_results_filenames, stellar_parameters)
    else:
        logger.info("Using pre-loaded data. Stellar parameters are: {0}".format(
            ", ".join(stellar_parameters)))


    # Compile the STAN model
    model = compile_model("models/mixture-model-with-uncertainties.stan")

    # Look at each node individually
    num_stellar_parameters, num_benchmarks, num_nodes = node_data.shape
    for i in enumerate(num_nodes):

        for j, stellar_parameter in enumerate(stellar_parameters):

            node_measurements = node_data[2*j, :, i]
            node_uncertainties = node_data[2*j + 1, :, i]

            non_spectroscopic_measurements = benchmarks[stellar_parameter]
            non_spectroscopic_uncertainties = benchmarks["e_{0}".format(stellar_parameter)]

            isfinite = np.isfinite(node_measurements) \
                & np.isfinite(node_uncertainties)

            # Collate the data
            data = {
                "N": sum(isfinite),
                "x_measured": non_spectroscopic_measurements[isfinite],
                "x_uncertainties": non_spectroscopic_uncertainties[isfinite],
                "y_measured": node_measurements[isfinite],
                "y_uncertainties": node_measurements[isfinite]
            }

            # Fit the mixture model to the data
            fit = model.sampling(data)

            print("Node {0}, stellar parameter: {1}".format(
                i, stellar_parameter))
            print(fit)


            trace_figure = fit.traceplot()

            samples = fit.extract(permuted=True)
            parameters = pd.DataFrame({"m": samples["m"], "b": samples["b"]})

            # Predictive model
            pred_x = np.linspace(
                np.min(data["x_measured"]),
                np.max(data["x_measured"]),
                1000)

            linear_model = lambda b, m, *theta: pd.Series({"fitted": m * pred_x + b})

            median_parameters = parameters.median()
            yhat = model(median_parameters)

            # get the predicted values for each chain
            chain_predictions = parameters.apply(linear_model, axis=1)

            data_figure = plt.figure()
            ax = data_figure.add_subplot(111)

            num_chains = 50
            indices = np.random.choice(300, num_chains)

            for index in indices:
                ax.plot(pred_x, chain_predictions.iloc[index, 0], color="lightgrey")

            #  Plot the data
            ax.errorbar(data["x_measured"], data["y_measured"],
                xerr=data["x_uncertainties"], yerr=data["y_uncertainties"],
                fmt=None, facecolor="k", ecolor="k", zorder=10)
            ax.plot(data["x_measured"], data["y_measured"], 'ko', zorder=10)

            # Plot the fitted values values
            ax.plot(pred_x, yhat["fitted"], "k", lw=2)

            # labels
            ax.set_xlabel("{0} ('true')".format(stellar_parameter))
            ax.set_ylabel("{0} (spectral node {1})".format(
                stellar_parameter, i))

            raise a
