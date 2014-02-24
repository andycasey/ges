# coding: utf-8
 
""" Visualise differences between nodes and recommended values """
 
from __future__ import division, print_function
 
__author__ = "Andy Casey <arc@cam.ast.ac.uk>"
 
# Third party libraries
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["boxplots"]

def boxplots(benchmarks, node_data, stellar_parameters,
	recommended_values=None, recommended_uncertainties=None, labels=None):
	"""Draw box plots"""

	widths = 0.45
	colors = ["b", "k"]
	recommended_colour = "r"

	num_benchmarks, num_nodes = node_data.shape[1:]

	figs = []
	for i, stellar_parameter in enumerate(stellar_parameters):

		fig = plt.figure()
		fig.subplots_adjust(bottom=0.20, right=0.95, top=0.95)
		ax = fig.add_subplot(111)

		data = node_data[i, :, :] - np.array([benchmarks[stellar_parameter]] * num_nodes).T

		ax.plot([0, num_benchmarks + 1], [0, 0], ":", c="#666666", zorder=-10)
		bp = ax.boxplot([[v for v in row if np.isfinite(v)] for row in data], widths=widths,
			patch_artist=True)

		assert len(bp["boxes"]) == num_benchmarks
		
		# Get 5% in y-direction
		text_y_percent = 3
		text_y_position = (text_y_percent/100.) * np.ptp(ax.get_ylim()) + ax.get_ylim()[0]
		for j in xrange(num_benchmarks):
			num_finite = np.sum(np.isfinite(node_data[i, j, :]))
			ax.text(j + 1, text_y_position, num_finite, size=10,
				horizontalalignment="center")

		text_y_position = (2.5*text_y_percent/100.) * np.ptp(ax.get_ylim()) + ax.get_ylim()[0]
		ax.text(num_benchmarks/2., text_y_position, "Measurements", size=10,
			horizontalalignment="center", fontweight="bold")

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
		plt.setp(bp["medians"], color=colors[0], linewidth=2)
		plt.setp(bp["fliers"], color=colors[1])
		plt.setp(bp["caps"], visible=False)
		plt.setp(bp["whiskers"], color=colors[1], linestyle="solid", linewidth=0.5)
		plt.setp(bp["boxes"], color=colors[1], linewidth=2, facecolor="w")

		ax.spines["left"]._linewidth = 0.5
		ax.spines["bottom"]._linewidth = 0.5

		# Draw recommended values if they exist
		if recommended_uncertainties is not None and recommended_values is not None:
			ax.errorbar(np.arange(1, num_benchmarks + 1), recommended_values[i, :] \
				- benchmarks[stellar_parameter], yerr=recommended_uncertainties[i, :],
				fmt=None, c=recommended_colour, zorder=50, elinewidth=1, capsize=1,
				ecolor=recommended_colour)

		if recommended_values is not None:
			ax.scatter(np.arange(1, num_benchmarks + 1), recommended_values[i, :] \
				- benchmarks[stellar_parameter], marker="o", c=recommended_colour,
				zorder=100, linewidth=0)

		figs.append(fig)

	return tuple(figs)