
import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas

from pystan import StanModel

import gesio


def generate_model(nodes=2, dimensions=("teff", "logg"), model_covariance=True, model_node_variances=False,
	model_intrinsic_variances=False, model_outliers=True):

	data_code = ""

	# Let's do the data code first
	for node in xrange(nodes):
		data_code += \
		"""
		// Node {n}
		int<lower=2> N_benchmarks_node{n};
		""".format(n=node)

		for dimension in dimensions:

			data_code += \
			"""
			//	- Spectroscopic measurements
			real sp_{dim}_node{n}_measured[N_benchmarks_node{n}];
			real sp_{dim}_node{n}_sigma[N_benchmarks_node{n}];

			//	- Non-specroscopic measurements
			real ns_{dim}_node{n}_measured[N_benchmarks_node{n}];
			real ns_{dim}_node{n}_sigma[N_benchmarks_node{n}];
			""".format(n=node, dim=dimension)

	model = """
	data {{
	{data_code}
	}}
	""".format(data_code=data_code)
	
	# Now let's do the parameter code
	parameters_code = ""
	for node in xrange(nodes):

		for dimension in dimensions:
			parameters_code += \
			"""
			real<lower=0> m_{dim}_node{n};
			real b_{dim}_node{n};
			""".format(dim=dimension, n=node)

	# Model the outliers
	if model_outliers:
		parameters_code += \
		"""
		real<lower=0,upper=1> alpha;
		"""
		for dimension in dimensions:
			parameters_code += \
			"""
			real outlier_{dim}_mu;
			real<lower=0> outlier_{dim}_sigma;
			""".format(dim=dimension)

	# Model the intrinsic variance?
	if model_intrinsic_variances:
		for dimension in dimensions:
			parameters_code += \
			"""
			real<lower=0> s_{dim}_intrinsic;
			""".format(dim=dimension)

	# Model the node variances?
	if model_node_variances:
		for node in nodes:
			for dimension in dimensions:
				parameters_code += \
				"""
				real<lower=0> s_{dim}_node{n};
				""".format(dim=dimension, n=node)

	# Model the covariances?
	if model_covariance:
		# We need the permutations
		# approximate...
		if len(dimensions) > 2:
			raise NotImplementedError

		parameters_code += \
		"""
		real<lower=-1,upper=1> rho_{dim_1}_{dim_2};
		""".format(dim_1=dimensions[0], dim_2=dimensions[1])

	model += """
	parameters {{
	{parameters_code}
	}}
	""".format(parameters_code=parameters_code)

	# Generate model code
	model_code = """
	// Declarations

	vector[{ndim}] ns_vector;
	vector[{ndim}] sp_vector;
	matrix[{ndim},{ndim}] covariance;
	""".format(ndim=len(dimensions))

	if model_outliers:
		model_code += \
		"""
		real log_alpha;
		real log1m_alpha;

		alpha ~ uniform(0, 1);

		log_alpha <- log(alpha);
		log1m_alpha <- log1m(alpha);
		"""

	for node in xrange(nodes):
		for dimension in dimensions:
			model_code += \
			"""
			m_{dim}_node{n} ~ normal(1, 0.1);
			b_{dim}_node{n} ~ normal(0, 5);
			""".format(n=node, dim=dimension)

	# Model the covariance?
	if model_covariance:
		model_code += \
		"""
		rho_{dim_1}_{dim_2} ~ uniform(-1, 1);
		""".format(dim_1=dimensions[0], dim_2=dimensions[1])

	# Model the outliers?
	if model_outliers:
		for dimension in dimensions:
			model_code += \
			"""
			outlier_{dim}_mu ~ normal(0, 1e3);
			outlier_{dim}_sigma ~ normal(0, 1e3);
			""".format(dim=dimension)

	# Model intrinsic variances?
	if model_intrinsic_variances:
		for dimension in dimensions:
			model_code += \
			"""
			s_{dim}_intrinsic ~ normal(0, 1e2);
			""".format(dim=dimension)

	# Model node variances?
	if model_node_variances:
		for node in xrange(nodes):
			for dimension in dimensions:
				model_code += \
				"""
				s_{dim}_node{n} ~ normal(0, 1e2);
				""".format(dim=dimension, n=node)

	# The good stuff
	for node in xrange(nodes):

		model_code += \
		"""
		for (i in 1:N_benchmarks_node{n}) {{

			// Non-spectroscopic measurements
			ns_vector[1] <- ns_{d1}_node{n}_measured[i];
			ns_vector[2] <- ns_{d2}_node{n}_measured[i];

			// Spectroscopic measurements
			sp_vector[1] <- m_{d1}_node{n} * sp_{d1}_node{n}_measured[i] + b_{d1}_node{n};
			sp_vector[2] <- m_{d2}_node{n} * sp_{d2}_node{n}_measured[i] + b_{d2}_node{n};

			// Covariance matrix
			covariance[1,1] <- pow(sp_{d1}_node{n}_sigma[i], 2) {s_intrinsic_d1} {s_node_d1};
			covariance[2,2] <- pow(sp_{d2}_node{n}_sigma[i], 2) {s_intrinsic_d2} {s_node_d2};
			covariance[1,2] <- sqrt(covariance[1,1] * covariance[2,2]) * rho_{d1}_{d2};
			covariance[2,1] <- sqrt(covariance[1,1] * covariance[2,2]) * rho_{d1}_{d2};

			increment_log_prob(
				{log_alpha} multi_normal_log(ns_vector, sp_vector, covariance)
			);
		""".format(d1=dimensions[0], d2=dimensions[1], n=node,
			s_intrinsic_d1="+ s_{d1}_intrinsic".format(d1=dimensions[0]) if model_intrinsic_variances else "",
			s_intrinsic_d2="+ s_{d2}_intrinsic".format(d2=dimensions[1]) if model_intrinsic_variances else "",
			s_node_d1="+ s_{d1}_node{n}".format(d1=dimensions[0], n=node) if model_node_variances else "",
			s_node_d2="+ s_{d2}_node{n}".format(d2=dimensions[1], n=node) if model_node_variances else "",
			log_alpha="log_alpha + " if model_outliers else "")

		# Modelling the outliers?
		if model_outliers:
			for j, dimension in enumerate(dimensions):
				model_code += \
				"""
				increment_log_prob(
					log1m_alpha + normal_log(sp_vector[{j}], outlier_{dim}_mu, outlier_{dim}_sigma)
				);
				""".format(j=j + 1, dim=dimension)

		model_code += "}\n"

	model += \
	"""
	model {{
	{model_code}
	}}
	""".format(model_code=model_code)

	return model


# Build the real data dict
nodes = 4
dimensions = ("teff", "logg")

# Ok, here is our real data:
try: benchmarks
except NameError:
    node_results_filenames = glob("data/iDR2.1/GES_iDR2_WG11_*.fits")
    remove_nodes = ("Recommended", )
    node_results_filenames = [filename for filename in node_results_filenames \
	if "_".join(os.path.basename(filename).split("_")[3:]).rstrip(".fits")\
		not in remove_nodes]

    # Load the data
    benchmarks, node_data = gesio.prepare_data("data/benchmarks.txt",
		node_results_filenames, map(str.upper, dimensions))
else:
    print("Using pre-loaded data. Stellar parameters are: {0}".format(", ".join(dimensions)))

# Put the data in the format we need for the model
data = {} 
for node in xrange(nodes):

	# Just get finite values from first stellar parameter axes
	finite = np.isfinite(node_data[0, :, node])
	data["N_benchmarks_node{0}".format(node)] = sum(finite)

	for i, dimension in enumerate(dimensions):

		s = {"dim": dimension, "n": node}
		data["sp_{dim}_node{n}_measured".format(**s)] = node_data[2*i, finite, node]
		data["sp_{dim}_node{n}_sigma".format(**s)] = node_data[2*i + 1, finite, node]
		data["ns_{dim}_node{n}_measured".format(**s)] = benchmarks[dimension.upper()][finite]
		data["ns_{dim}_node{n}_sigma".format(**s)] = benchmarks["e_{0}".format(dimension.upper())][finite]


# Generate the model
code = generate_model(nodes=nodes, dimensions=dimensions, model_covariance=True, model_node_variances=False,
	model_intrinsic_variances=False, model_outliers=True)
model = StanModel(model_code=code)

print("Optimizing...")
op = model.optimizing(data=data)

print("Fitting...")
fit = model.sampling(data=data, pars=op["par"], iter=20000)

subplots_adjust = { "left": 0.10, "bottom": 0.05, "right": 0.95, "top": 0.95,
	"wspace": 0.20, "hspace": 0.45
	}

# Plot the m, b parameters for each node
dimensions_traced = []
for node in xrange(nodes):
	node_dimensions = \
		["m_{dim}_node{n}".format(dim=dimension, n=node) for dimension in dimensions] \
	  + ["b_{dim}_node{n}".format(dim=dimension, n=node) for dimension in dimensions]
	dimensions_traced.extend(node_dimensions)

	fig = fit.traceplot(node_dimensions)
	fig.subplots_adjust(**subplots_adjust)
	fig.savefig("trace-node-{0}.jpg".format(node))

plots_per_trace = 4
other_dimensions = sorted(list(set(op["par"]).difference(dimensions_traced)))

for i in xrange(int(np.floor(len(other_dimensions)/plots_per_trace) + 1)):
	fig = fit.traceplot(other_dimensions[plots_per_trace*i:(i+1)*plots_per_trace])
	fig.subplots_adjust(**subplots_adjust)
	fig.savefig("trace-{0}.jpg".format(i))


# Draw from the distributions 
samples = fit.extract(permuted=True)

for node in xrange(nodes):

	data_figure = plt.figure()
	
	for i, dimension in enumerate(dimensions):

		ax = data_figure.add_subplot(len(dimensions), 1, i + 1)
		
		strs = {"dim": dimension, "n": node}
		parameters = pandas.DataFrame({
			"m": samples["m_{dim}_node{n}".format(**strs)],
			"b": samples["b_{dim}_node{n}".format(**strs)]})

		# Predictive model
		x = np.linspace(
		    np.min(data["ns_{dim}_node{n}_measured".format(**strs)]),
		    np.max(data["ns_{dim}_node{n}_measured".format(**strs)]),
		    1000)

		linear_model = lambda theta: pandas.Series({"fitted": theta[1] * x + theta[0]})

		yhat = linear_model(parameters.median())

		# get the predicted values for each chain
		chain_predictions = parameters.apply(linear_model, axis=1)

		num_chains = 50
		indices = np.random.choice(300, num_chains)

		for index in indices:
		    ax.plot(x, chain_predictions.iloc[index, 0], color="lightgrey")

		#  Plot the data
		ax.errorbar(
			x=data["ns_{dim}_node{n}_measured".format(**strs)],
			y=data["sp_{dim}_node{n}_measured".format(**strs)],
		    xerr=data["ns_{dim}_node{n}_sigma".format(**strs)],
		    yerr=data["sp_{dim}_node{n}_sigma".format(**strs)],
		    fmt=None, facecolor="k", ecolor="k", zorder=10)
		ax.plot(
			data["ns_{dim}_node{n}_measured".format(**strs)],
			data["sp_{dim}_node{n}_measured".format(**strs)],
			'ko', zorder=10)

		# Plot the fitted values values
		ax.plot(x, yhat["fitted"], "k", lw=2)

		ax.set_ylim(ax.get_xlim())
		ax.set_xlabel("{dim} (NS)".format(**strs))
		ax.set_ylabel("{dim} (Node {n})".format(**strs))

	data_figure.savefig("draw-fits-node{n}.jpg".format(**strs))



