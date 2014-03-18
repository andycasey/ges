
""" Generate a STAN model for all observed data points """

import logging
import os
from glob import glob
from textwrap import dedent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mixture import compile_model
from gesio import load_benchmarks, load_node_results, prepare_data

logger = logging.getLogger(__name__)


def generate_model(all_data, benchmarks):

	parameters = "real<lower=0> epsilon;\n"
	model = "epsilon ~ normal(0, 1e2);\n"

	transformed_parameters = ""
	model_data = ""
	model_bracket = ""
	model_bracket_declarations = ""
	transformed_parameter_declarations = ""

	data = {}
	num_stellar_parameters, num_benchmarks, num_nodes = all_data.shape
	for n in xrange(num_nodes):

		node_data = all_data[0, :, n]
		node_uncertainties = all_data[1, :, n]
		finite = np.isfinite(node_data) * np.isfinite(node_uncertainties)
		if sum(finite) == 0 or np.any(node_uncertainties < 0): continue


		model_data += \
			"""
			  int<lower=2> N_{node};
			  real x_measured_{node}[N_{node}];
			  real x_uncertainty_{node}[N_{node}];
			  real y_measured_{node}[N_{node}];
			  real y_uncertainty_{node}[N_{node}];
			""".format(node=n)

		# Provide the actual data
		data.update({
			# Number of finite measurements by this node
			"N_{0}".format(n): sum(finite),
			# Non-spectroscopic measurements 
			"x_measured_{0}".format(n): benchmarks["TEFF"][finite],
			# Non-spectroscopic uncertainties
			"x_uncertainty_{0}".format(n): benchmarks["e_TEFF"][finite],
			# Spectroscopic measurements from this node
			"y_measured_{0}".format(n): node_data[finite],
			# Spectroscopic uncertainties from this node
			"y_uncertainty_{0}".format(n): node_uncertainties[finite],
			})

		parameters += \
			"""
			  real<lower=0,upper=2> m_{node};
			  real<lower=-1000,upper=1000> b_{node};
			  real<lower=0,upper=1> p_{node};
			  real<lower=0> Yb_{node};
			  real<lower=0> Vb_{node};
			  real x_{node}[N_{node}];
			""".format(node=n)

		transformed_parameter_declarations += \
			"real mu_{node}[N_{node}];\n".format(node=n)

		transformed_parameters += \
			"""
  			  for(i in 1:N_{node})
				mu_{node}[i] <- b_{node} + m_{node}*x_{node}[i];
			""".format(node=n)

		model += \
		"""
		  x_{node} ~ normal(x_measured_{node}, x_uncertainty_{node});
		  y_measured_{node} ~ normal(mu_{node}, y_uncertainty_{node});

		  m_{node} ~ uniform(0, 2);
		  b_{node} ~ uniform(-1000, 1000);
		  p_{node} ~ uniform(0, 1);
		  Yb_{node} ~ normal(0, 1e3);
		  Vb_{node} ~ normal(0, 1e3);
		""".format(node=n)

		model_bracket_declarations += \
			"""
			  real log_p_{node};
			  real log1m_p_{node};
			""".format(node=n)

		model_bracket += \
			"""
			  log_p_{node} <- log(p_{node});
			  log1m_p_{node} <- log1m(p_{node});
			  for (i in 1:N_{node})
				increment_log_prob(log_sum_exp(
			  	   log_p_{node}  + normal_log(y_measured_{node}[i], mu_{node}[i], y_uncertainty_{node}[i] + epsilon),
			  	  log1m_p_{node} + normal_log(y_measured_{node}[i], Yb_{node}, Vb_{node})));
			""".format(node=n)

	model_data, parameters, transformed_parameters, model, model_bracket = \
		map(dedent, [model_data, parameters, transformed_parameters, model, model_bracket])

	# Bring it all together
	generated_model = \
	"""
	data {{
	  {model_data}
	}}
	parameters {{
	  {parameters}
	}}
	transformed parameters {{
	  {transformed_parameter_declarations}
	  {transformed_parameters}
	}}
	model {{
	{model}
	  {{
		{model_bracket_declarations}
	    {model_bracket}
	  }}
	}}""".format(model_data=model_data, parameters=parameters,
		transformed_parameter_declarations=transformed_parameter_declarations,
		transformed_parameters=transformed_parameters, model=model,
		model_bracket_declarations=model_bracket_declarations,
		model_bracket=model_bracket)

	generated_model = dedent(generated_model)

	return (generated_model, data)



if __name__ == "__main__":

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
	    stellar_parameters = ("TEFF",)# "LOGG")
	    benchmarks, node_data = prepare_data("data/benchmarks.txt",
		node_results_filenames, stellar_parameters)
	else:
	    logger.info("Using pre-loaded data. Stellar parameters are: {0}".format(
		", ".join(stellar_parameters)))

	num_nodes = 3
	model_code, data = generate_model(node_data[:, :, :num_nodes], benchmarks)

	with open("model.stan", "w") as fp:
		fp.write(model_code)

	model = compile_model("model.stan")

	fit = model.sampling(data, iter=15000)

 	trace_figure = fit.traceplot()
	trace_figure.subplots_adjust(bottom=0.05, hspace=0.70,
		top=0.95, right=0.95, left=0.10)
	trace_figure.savefig("mixture-stan-trace.png")

	samples = fit.extract(permuted=True)
	for i in xrange(num_nodes):

		parameters = pd.DataFrame({
			"m_{0}".format(i): samples["m_{0}".format(i)],
			"b_{0}".format(i): samples["b_{0}".format(i)]})

		# Predictive model
		pred_x = np.linspace(
			np.min(data["x_measured_{0}".format(i)]),
			np.max(data["x_measured_{0}".format(i)]),
			1000)

		linear_model = lambda theta: pd.Series({"fitted": theta[1] * pred_x + theta[0]})

		median_parameters = parameters.median()
		yhat = linear_model(median_parameters)

		# get the predicted values for each chain
		chain_predictions = parameters.apply(linear_model, axis=1)

		data_figure = plt.figure()
		ax = data_figure.add_subplot(111)

		num_chains = 50
		indices = np.random.choice(300, num_chains)

		for index in indices:
			ax.plot(pred_x, chain_predictions.iloc[index, 0], color="lightgrey")

		#  Plot the data
		ax.errorbar(data["x_measured_{0}".format(i)], data["y_measured_{0}".format(i)],
			xerr=data["x_uncertainty_{0}".format(i)], yerr=data["y_uncertainty_{0}".format(i)],
			fmt=None, facecolor="k", ecolor="k", zorder=10)
		ax.plot(data["x_measured_{0}".format(i)], data["y_measured_{0}".format(i)], 'ko', zorder=10)

		# Plot the fitted values values
		ax.plot(pred_x, yhat["fitted"], "k", lw=2)

		ax.set_xlabel("teff ('true')")
		ax.set_ylabel("teff (node {0})".format(i))

		data_figure.savefig("mixture-stan-{0}.png".format(i))
