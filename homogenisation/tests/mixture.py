# coding: utf-8

""" A toy mixture model with uncertainties in STAN """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@cam.ast.ac.uk>"

# Standard libraries
import logging

# Third party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystan


def main():
    # Load the data for the toy model
    ids, x, y, xerr, yerr, pxy = np.loadtxt("../data/hogg-toy-model.data", unpack=True)

    with open("models/mixture-model-with-uncertainties.stan", "r") as fp:
        model_code = fp.read()

    # Fit the model
    fit = pystan.stan(model_code=model_code, iter=10000, chains=8,
        data={
            "x_measured": x, "x_uncertainty": xerr,
            "y_measured": y, "y_uncertainty": yerr,
            "N": len(x)
        })

    print(fit)
    fit.traceplot()

    samples = fit.extract(permuted=True)
    parameters = pd.DataFrame({"m": samples["m"], "b": samples["b"]})

    # Predictive model
    pred_x = np.arange(0, 300)
    model = lambda theta: pd.Series({"fitted": theta[0] + theta[1] * pred_x})

    median_parameters = parameters.median()

    yhat = model(median_parameters)

    # get the predicted values for each chain
    chain_predictions = parameters.apply(model, axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    num_chains = 50
    indices = np.random.choice(300, num_chains)

    for i, index in enumerate(indices):
        ax.plot(pred_x, chain_predictions.iloc[index, 0], color="lightgrey")

    #  data
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=None, facecolor="k", ecolor="k", zorder=10)
    ax.plot(x, y, 'ko', zorder=10)

    # fitted values
    ax.plot(pred_x, yhat["fitted"], "k", lw=2)

    # supplementals
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 750)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.show()


if __name__ == "__main__":
    result = main()
