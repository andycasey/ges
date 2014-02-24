# coding: utf-8
 
""" Visualise differences between nodes and recommended values """
 
from __future__ import division, print_function
 
__author__ = "Andy Casey <arc@cam.ast.ac.uk>"
 
# Third party libraries
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["boxplots"]

def boxplots(benchmarks, node_data, stellar_parameters, labels=None,
    recommended_values=None, recommended_uncertainties=None,
    colours=("b", "k", "r")):
    """Create box plots highlighting the difference between expected stellar
    parameters from non-spectroscopic methods, and those measured by each
    Gaia-ESO Survey node.

    Inputs
    ------
    benchmarks : table of benchmarks 

    node_data : shape (N, M, O)
        N = stellar parameters
        M = num benchmarks
        O = num nodes

    recommended_values : array of shape (M, N)

    recommended_uncertainties : array of shape (M, N)

    labels = list of length N

    colors = list of length 3 (outlines, median, recommended)
    """

    num_benchmarks, num_nodes = node_data.shape[1:]

    figs = []
    for i, stellar_parameter in enumerate(stellar_parameters):

        fig = plt.figure()
        fig.subplots_adjust(bottom=0.20, right=0.95, top=0.95)
        ax = fig.add_subplot(111)

        data = node_data[i, :, :] - np.array([benchmarks[stellar_parameter]] * num_nodes).T

        ax.plot([0, num_benchmarks + 1], [0, 0], ":", c="#666666", zorder=-10)
        bp = ax.boxplot([[v for v in row if np.isfinite(v)] for row in data], widths=0.45,
            patch_artist=True)

        assert len(bp["boxes"]) == num_benchmarks
        
        ylims = ax.get_ylim()
        # Get 5% in y-direction
        text_y_percent = 3
        text_y_position = (text_y_percent/100.) * np.ptp(ylims) + ylims[0]
        for j in xrange(num_benchmarks):
            num_finite = np.sum(np.isfinite(node_data[i, j, :]))
            ax.text(j + 1, text_y_position, num_finite, size=10,
                horizontalalignment="center")

        text_y_position = (2.5*text_y_percent/100.) * np.ptp(ylims) + ylims[0]
        ax.text(num_benchmarks/2., text_y_position, "Measurements", size=10,
            horizontalalignment="center", fontweight="bold")
        # Set y-lims back to where they should be
        ax.set_ylim(ylims)

        # Hide spines and tick positions
        [ax.spines[border].set_visible(False) for border in ("top", "right")]
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")

        # Label axes
        if labels is not None:
            ax.set_ylabel(labels[i])
        else:
            ax.set_ylabel("Delta {0}".format(stellar_parameter))

        ax.xaxis.set_ticklabels([label.replace("_", " ") for label in benchmarks["Object"]],
            rotation=90)
        
        # Set colours
        plt.setp(bp["medians"], color=colours[0], linewidth=2)
        plt.setp(bp["fliers"], color=colours[1])
        plt.setp(bp["caps"], visible=False)
        plt.setp(bp["whiskers"], color=colours[1], linestyle="solid", linewidth=0.5)
        plt.setp(bp["boxes"], color=colours[1], linewidth=2, facecolor="w")

        ax.spines["left"]._linewidth = 0.5
        ax.spines["bottom"]._linewidth = 0.5

        # Draw recommended values if they exist
        if recommended_uncertainties is not None and recommended_values is not None:
            ax.errorbar(np.arange(1, num_benchmarks + 1), recommended_values[i, :] \
                - benchmarks[stellar_parameter], yerr=recommended_uncertainties[i, :],
                fmt=None, c=colours[2], zorder=50, elinewidth=1, capsize=1,
                ecolor=colours[2])

        if recommended_values is not None:
            ax.scatter(np.arange(1, num_benchmarks + 1), recommended_values[i, :] \
                - benchmarks[stellar_parameter], marker="o", c=colours[2],
                zorder=100, linewidth=0)

        figs.append(fig)

    return tuple(figs)