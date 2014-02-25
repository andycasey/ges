# coding: utf-8
 
""" Weight Gaia-ESO Survey node data based on minimum Euclidean distance """
 
from __future__ import division, print_function
 
__author__ = "Andy Casey <arc@cam.ast.ac.uk>"
 
# Standard libraries
import logging
import os
from glob import glob
 
# Third party libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

# Module-specific imports
from gesio import load_benchmarks, load_node_results, prepare_data
from plot import boxplots
 
__all__ = ["get_weights", "main"]

logging.basicConfig(filename="homogenisation.log", level=logging.INFO)
logger = logging.getLogger(__name__)



def get_weights(benchmarks, node_data, stellar_parameters, scales=None):
    """Determine weights for Best Linear Unbiased
    Estimate of parameters"""

    num_nodes = node_data.shape[2]
    
    def func(weights, benchmarks, node_data, stellar_parameters,
        scales):
        """ Calculate the Euclidean distance for all benchmark star
        measurements """

        L2 = 0
        for i, stellar_parameter in enumerate(stellar_parameters):
            for j, benchmark in enumerate(benchmarks):

                isfinite = np.isfinite(node_data[2*i, j, :])

                # Normalise the weights
                #normalised_weights = (weights/sum(weights))*(num_nodes/sum(isfinite))
                normalised_weights = weights[isfinite]/sum(weights[isfinite])

                L2_i = normalised_weights * (benchmark[stellar_parameter] - node_data[2*i, j, isfinite])
                if scales is not None and stellar_parameter in scales.keys():
                    L2_i *= scales[stellar_parameter]

                L2 += np.sum(L2_i**2)

        return L2
   
    # Initial guess: all nodes give equal weight
    x0 = np.array([1.0/num_nodes] * num_nodes)

    return scipy.optimize.fmin_slsqp(func, x0=x0,
        args=(benchmarks, node_data, stellar_parameters, scales, ),
        eqcons=[lambda x, b, n_d, s_p, s: sum(x) - 1],
        disp=False)


def euclidean(measurements, uncertainties, scales=None):
    """ Calculate the optimally?-weighted parameter based on minimum
    total Euclidean distance to the Gaia benchmark stars """



def main():
    """Do the things"""

    # Check if we have already loaded the data
    global benchmarks, node_data, stellar_parameters

    try: benchmarks
    except NameError:
    	logger.info("Loading data..")
        node_results_filenames = glob("data/iDR2.1/GES_iDR2_WG11_*.fits")
        remove_nodes = ("Recommended", )
        node_results_filenames = [filename for filename in node_results_filenames \
            if "_".join(os.path.basename(filename).split("_")[3:]).rstrip(".fits") not in remove_nodes]

        # Load the data
        stellar_parameters = ("TEFF", "LOGG", "MH")
        benchmarks, node_data = prepare_data("data/benchmarks.txt", node_results_filenames,
            stellar_parameters)
    else:
    	logger.info("Using pre-loaded data")

    # Calculate weights based on minimal Euclidean distance
    stellar_parameters = ("TEFF", "LOGG", "MH")

    num_nodes = node_data.shape[2]
    recommended_measurements = np.zeros(map(len, [stellar_parameters, benchmarks]))
    weights = get_weights(benchmarks, node_data, stellar_parameters,
        scales={
            "TEFF": 1./np.ptp(benchmarks["TEFF"]),
            "LOGG": 1./np.ptp(benchmarks["LOGG"]),
            "MH": 1./np.ptp(benchmarks["MH"])
        })

    for j, stellar_parameter in enumerate(stellar_parameters):
        for i, benchmark in enumerate(benchmarks):

            node_measurements = node_data[2*j, i, :]
            isfinite = np.isfinite(node_measurements)

            # Normalise the weights
            normalised_weights = weights[isfinite]/sum(weights[isfinite])
            
            m_euclidean = np.sum((normalised_weights * node_measurements[isfinite]))
            recommended_measurements[j, i] = m_euclidean
            
    # Visualise the differences
    figs = boxplots(benchmarks, node_data[::2, :, :], stellar_parameters,
        labels=("$\Delta{}T_{\\rm eff}$ (K)", "$\Delta{}\log{g}$ (dex)", "$\Delta{}$[Fe/H] (dex)"),
        recommended_values=recommended_measurements)
    [fig.savefig("euclidean-benchmarks-{0}.pdf".format(stellar_parameter.lower())) \
    	for fig, stellar_parameter in zip(figs, stellar_parameters)]

if __name__ == "__main__":
	main()