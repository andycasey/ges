# coding: utf-8
 
""" Visualise differences between nodes and recommended values """
 
from __future__ import division, print_function
 
__author__ = "Andy Casey <arc@cam.ast.ac.uk>"
 
# Third party libraries
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

__all__ = ["boxplots", "histograms"]

def histograms(benchmarks, node_data, stellar_parameters, parameter_labels=None,
    node_labels=None, recommended_values=None, colours=("b", "k", "r")):
    """ Show histograms comparing the dispersion around the Gaia benchmark
    stars compared to the recommended values """

    num_stellar_parameters, num_benchmarks, num_nodes = node_data.shape

    bins = 10

    factor = 2.0
    lbmargin = 0.25 * factor
    trmargin = 0.25 * factor
    whspace = 0.10 

    num_y_plots = num_nodes
    if recommended_values is not None: num_y_plots += 1

    xdim = lbmargin + factor * num_stellar_parameters + (num_stellar_parameters - 1.) * whspace + trmargin
    ydim = lbmargin + factor * num_y_plots * (num_y_plots - 1.) * whspace + trmargin

    #print(xdim, ydim)
    fig, axes = plt.subplots(num_y_plots, num_stellar_parameters, figsize=(xdim, ydim))
    fig.subplots_adjust(bottom=0.05, top=0.95, wspace=whspace)
    for j, stellar_parameter in enumerate(stellar_parameters):

        distributions = []
        for i in xrange(num_nodes):
            distribution = node_data[j, :, i] - benchmarks[stellar_parameter]
            isfinite = np.isfinite(distribution)

            distributions.append(distribution[isfinite])

        # Get ranges of distributions for this stellar parameter
        abs_range = np.max(map(lambda x: 1.5*np.std(x) if len(x) > 0 else -1, np.abs(distributions)))

        # Draw histograms
        for i, distribution in enumerate(distributions):
            if len(distribution) != 0: 
                axes[i, j].hist(distribution,
                    bins=np.linspace(-abs_range, +abs_range, bins),
                    facecolor="#666666")

                mu = np.median(distribution)
                sigma = np.std(distribution)
                axes[i, j].text(0.9, 0.90, "$\mu = {0:+5.2f}$".format(mu),
                    horizontalalignment="right", fontsize="10", transform=axes[i, j].transAxes)
                axes[i, j].text(0.9, 0.80, "$\sigma = {0:5.2f}$".format(sigma),
                    horizontalalignment="right", fontsize="10", transform=axes[i, j].transAxes)

            # Adjust spines and axes
            [axes[i, j].spines[border].set_visible(False) for border in ("top", "right")]

            axes[i, j].xaxis.set_major_locator(MaxNLocator(5))
            if i + 1 != len(distributions) or recommended_values is not None:
                axes[i, j].xaxis.set_ticklabels([])

            axes[i, j].set_xlim(-abs_range, +abs_range)
            axes[i, j].yaxis.set_ticks_position("left")
            axes[i, j].xaxis.set_ticks_position("bottom")
            axes[i, j].spines["left"]._linewidth = 0.5
            axes[i, j].spines["bottom"]._linewidth = 0.5

        # If we have recommended values, show their distribution too:
        if recommended_values is not None:
            distribution = recommended_values[j, :] - benchmarks[stellar_parameter]
            axes[i + 1, j].hist(distribution,
                bins=np.linspace(-abs_range, +abs_range, bins),
                facecolor="#4682b4")

            mu = np.median(distribution)
            sigma = np.std(distribution)
            axes[i + 1, j].text(0.9, 0.90, "$\mu = {0:+5.2f}$".format(mu),
                horizontalalignment="right", fontsize="10", transform=axes[i + 1, j].transAxes)
            axes[i + 1, j].text(0.9, 0.80, "$\sigma = {0:5.2f}$".format(sigma),
                horizontalalignment="right", fontsize="10", transform=axes[i + 1, j].transAxes)

            [axes[i + 1, j].spines[border].set_visible(False) for border in ("top", "right")]

            axes[i + 1, j].xaxis.set_major_locator(MaxNLocator(5))
            if i + 1 != len(distributions) or recommended_values is not None:
                axes[i + 1, j].xaxis.set_ticklabels([])

            axes[i + 1, j].set_xlim(-abs_range, +abs_range)
            axes[i + 1, j].yaxis.set_ticks_position("left")
            axes[i + 1, j].xaxis.set_ticks_position("bottom")
            axes[i + 1, j].spines["left"]._linewidth = 0.5
            axes[i + 1, j].spines["bottom"]._linewidth = 0.5


        # Show x-axis for bottom row
        axes[-1, j].xaxis.set_visible(True)

        if parameter_labels is not None:
            axes[-1, j].set_xlabel(parameter_labels[j])

        else:
            axes[-1, j].set_xlabel(stellar_parameter)


    # Set all ylims as the same
    for i, row in enumerate(axes):

        max_ylim = int(np.max([ax.get_ylim()[1] for ax in row]))
        for ax in row:
            ax.set_ylim(0, max_ylim)
            ax.set_yticks([0, max_ylim])

        # Only show y-ticks on the LHS
        row[0].set_yticklabels([0, max_ylim])
        row[0].set_ylabel("N")
        
        if len(row) > 0:
            [ax.set_yticklabels([]) for ax in row[1:]]

        if num_stellar_parameters % 2:
            index = int(np.ceil(num_stellar_parameters/2)) - 1
            loc = "center"

        else:
            index = 0
            loc = "left"

        if node_labels is not None and i != len(node_labels):
            row[index].set_title(node_labels[i], loc=loc)

        if recommended_values is not None and i == len(node_labels):
            row[index].set_title("Recommened values", loc=loc)

    return fig



def boxplots(benchmarks, node_data, stellar_parameters, labels=None,
    recommended_values=None, recommended_uncertainties=None, sort=True,
    summarise=True, colours=("b", "k", "r")):
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

        # Sort?
        sort_indices = np.arange(len(data)) if not sort else np.argsort(benchmarks[stellar_parameter])
        data = data[sort_indices]

        ax.plot([0, num_benchmarks + 1], [0, 0], ":", c="#666666", zorder=-10)
        bp = ax.boxplot([[v for v in row if np.isfinite(v)] for row in data], widths=0.45,
            patch_artist=True)

        assert len(bp["boxes"]) == num_benchmarks
        
        ylims = ax.get_ylim()
        # Get 5% in y-direction
        text_y_percent = 3
        text_y_position = (text_y_percent/100.) * np.ptp(ylims) + ylims[0]
        for j in xrange(num_benchmarks):
            num_finite = np.sum(np.isfinite(data[j]))
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

        ax.xaxis.set_ticklabels([label.replace("_", " ") for label in benchmarks["Object"][sort_indices]],
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
            ax.errorbar(np.arange(1, num_benchmarks + 1), recommended_values[i, sort_indices] \
                - benchmarks[stellar_parameter][sort_indices], yerr=recommended_uncertainties[i, sort_indices],
                fmt=None, c=colours[2], zorder=50, elinewidth=1, capsize=1,
                ecolor=colours[2])

        if recommended_values is not None:
            ax.scatter(np.arange(1, num_benchmarks + 1), recommended_values[i, sort_indices] \
                - benchmarks[stellar_parameter][sort_indices], marker="o", c=colours[2],
                zorder=100, linewidth=0)

        if summarise:

            median_diffs = [benchmarks[stellar_parameter][j] - np.median(node_data[i, j, np.isfinite(node_data[i, j, :])]) for j in xrange(node_data.shape[1])]
            mu_diff_median = np.median(median_diffs)
            sigma_diff_median = np.std(median_diffs)

            xpos, ypos = 0.75, 0.95
            ax.text(xpos, ypos, "$\mu = {0:+5.2f},\,\sigma = {1:5.2f}$".format(mu_diff_median, sigma_diff_median),
                color=colours[0], transform=ax.transAxes, horizontalalignment="left")

            if recommended_values is not None:
                recommended_diffs = recommended_values[i, :] - benchmarks[stellar_parameter]
                mu_diff_recommended = np.median(recommended_diffs)
                sigma_diff_recommended = np.std(recommended_diffs)

                xpos, ypos = 0.75, 0.90
                ax.text(xpos, ypos, "$\mu = {0:+5.2f},\,\sigma = {1:5.2f}$".format(mu_diff_recommended, sigma_diff_recommended),
                    color=colours[2], transform=ax.transAxes, horizontalalignment="left")

        figs.append(fig)

    return tuple(figs)