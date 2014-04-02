
import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas

from pystan import StanModel

import gesio


def generate_model(data, dimensions=("teff", "logg")):

	benchmarks, nodes = data.shape[1:]

	data_code = \
	"""
	int<lower=2> N_benchmarks;
	int<lower=1> N_nodes;
	"""

	for dimension in dimensions:
		data_code += \
		"""
		//	- Non-specroscopic measurements
		real ns_{dim}_measured[N_benchmarks];
		real ns_{dim}_sigma[N_benchmarks];
		""".format(dim=dimension)

	# Let's do the data code first
	for node in xrange(nodes):
		for dimension in dimensions:

			data_code += \
			"""
			//	- Spectroscopic measurements
			real sp_{dim}_node{n}_measured[N_benchmarks];
			real sp_{dim}_node{n}_sigma[N_benchmarks];
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
	parameters_code += \
	"""
	real<lower=0,upper=1> outlier_fraction;
	"""
	for dimension in dimensions:
		parameters_code += \
		"""
		real outlier_{dim}_mu;
		real<lower=0> outlier_{dim}_variance;
		""".format(dim=dimension)

	# Model the intrinsic variance
	for dimension in dimensions:
		parameters_code += \
		"""
		real<lower=0> s_{dim}_intrinsic;
		""".format(dim=dimension)

	# Model the node variances
	for node in xrange(nodes):
		for dimension in dimensions:
			parameters_code += \
			"""
			real<lower=0> s_{dim}_node{n};
			""".format(dim=dimension, n=node)

	model += \
	"""
	parameters {{
	{parameters_code}
	}}
	""".format(parameters_code=parameters_code)

	# Generate model code
	model_code = \
	"""
	real log_outlier_fraction;
	real log1m_outlier_fraction;
	"""

	for node in xrange(nodes):
		for dimension in dimensions:
			model_code += \
			"""
			m_{dim}_node{n} ~ normal(1, 0.5);
			b_{dim}_node{n} ~ normal(0, 500);
			""".format(n=node, dim=dimension)

	# Model the outliers
	model_code += \
	"""
	outlier_fraction ~ uniform(0, 1);

	log_outlier_fraction <- log(outlier_fraction);
	log1m_outlier_fraction <- log1m(outlier_fraction);
	"""

	for dimension in dimensions:
		model_code += \
		"""
		outlier_{dim}_mu ~ normal(0, 1e3);
		outlier_{dim}_variance ~ normal(0, 1e3);
		""".format(dim=dimension)

	# Model intrinsic variances
	for dimension in dimensions:
		model_code += \
		"""
		s_{dim}_intrinsic ~ normal(0, {value});
		""".format(dim=dimension, value=1e2 if dimension == "teff" else 1)

	# Model node variances
	for node in xrange(nodes):
		for dimension in dimensions:
			model_code += \
			"""
			s_{dim}_node{n} ~ normal(0, {value});
			""".format(dim=dimension, n=node, value=1e2 if dimension == "teff" else 1)

	# For each stellar parameter
	for i, dimension in enumerate(dimensions):

		# For each benchmark
		for j in xrange(benchmarks):

			finite = np.isfinite(data[2*i, j, :]) * (data[2*i+1, j, :] > 0) 
			if sum(finite) == 0: continue

			finite_indices = np.where(finite)[0]
			
			# sum(finite) is how many nodes measured it

			model_code += \
			"""
			{{
			vector[{n_finite}] yi;
			vector[{n_finite}] transformed_yi;
			vector[{n_finite}] sp_uncertainty;
			matrix[{n_finite},{n_finite}] covariance;
			""".format(n_finite=sum(finite))

			# The non-spectroscopic vector
			for k in xrange(1, 1 + sum(finite)):
				model_code += \
				"yi[{k}] <- sp_{dim}_node{n}_measured[{jp1}];\n".format(n=finite_indices[k-1], dim=dimension, k=k, jp1=j+1)

			# The spectroscopic vector
			for k in xrange(1, 1 + sum(finite)):
				model_code += \
				"""
				transformed_yi[{k}] <- m_{dim}_node{n} * ns_{dim}_measured[{jp1}] + b_{dim}_node{n};
				sp_uncertainty[{k}] <- outlier_{dim}_variance + pow(sp_{dim}_node{n}_sigma[{jp1}], 2) + pow(s_{dim}_intrinsic, 2) + pow(s_{dim}_node{n}, 2);
				""".format(k=k, dim=dimension, n=finite_indices[k-1], jp1=j+1)



			# The covariance matrix
			for k in xrange(1, 1 + sum(finite)):
				for l in xrange(1, 1 + sum(finite)):
					if k == l:
						model_code += \
						"covariance[{k},{l}] <- pow(sp_{dim}_node{n}_sigma[{jp1}], 2) + pow(s_{dim}_intrinsic, 2) + pow(s_{dim}_node{n}, 2);\n".format(
							dim=dimension, n=finite_indices[k-1], jp1=j+1, k=k, l=l)
					else:
						model_code += \
						"covariance[{k},{l}] <- pow(s_{dim}_intrinsic, 2);\n".format(dim=dimension, k=k, l=l)

			# Increment the log likelihood
			model_code += \
			"""
			increment_log_prob(log_sum_exp(
				 log1m_outlier_fraction  + multi_normal_log(yi, transformed_yi, covariance),
				 log_outlier_fraction + normal_log(yi, outlier_{dim}_mu, sp_uncertainty)
			));
			
			}}
			""".format(dim=dimension, jp1=j+1);



	model += \
	"""
	model {{
	{model_code}
	}}
	""".format(model_code=model_code)

	return model


# Build the real data dict
nodes = 9
dimensions = ("teff", )#"logg")

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

    node_data = node_data[:, :, :nodes]

else:
    print("Using pre-loaded data. Stellar parameters are: {0}".format(", ".join(dimensions)))



code = generate_model(data=node_data, dimensions=dimensions)

# Put the data in the format we need for the model
data = {
	"N_benchmarks": node_data.shape[1],
	"N_nodes": node_data.shape[2]
}

for node in xrange(nodes):

	#Set non-finite measurements and uncertainties (as determined by teff) to -1
	finite = np.isfinite(node_data[0, :, node])
	node_data[:, ~finite, node] = -1
	
	for i, dimension in enumerate(dimensions):

		s = {"dim": dimension, "n": node}
		data["sp_{dim}_node{n}_measured".format(**s)] = node_data[2*i, :, node]
		data["sp_{dim}_node{n}_sigma".format(**s)] = node_data[2*i + 1, :, node]
		data["ns_{dim}_measured".format(**s)] = benchmarks[dimension.upper()]
		data["ns_{dim}_sigma".format(**s)] = benchmarks["e_{0}".format(dimension.upper())]		

# Generate the model
with open("model.stan", "w") as fp:
	fp.write(code)

print(node_data.shape)
model = StanModel(model_code=code)

print("Optimizing...")
op = model.optimizing(data=data)

print("Optimized Values: \n{0}".format(op["par"]))
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
		    np.min(data["ns_{dim}_measured".format(**strs)]),
		    np.max(data["ns_{dim}_measured".format(**strs)]),
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
			x=data["ns_{dim}_measured".format(**strs)],
			y=data["sp_{dim}_node{n}_measured".format(**strs)],
		    xerr=data["ns_{dim}_sigma".format(**strs)],
		    yerr=data["sp_{dim}_node{n}_sigma".format(**strs)],
		    fmt=None, facecolor="k", ecolor="k", zorder=10)
		ax.plot(
			data["ns_{dim}_measured".format(**strs)],
			data["sp_{dim}_node{n}_measured".format(**strs)],
			'ko', zorder=10)

		# Plot the fitted values values
		ax.plot(x, yhat["fitted"], "k", lw=2)

		ax.set_ylim(ax.get_xlim())
		ax.set_xlabel("{dim} (NS)".format(**strs))
		ax.set_ylabel("{dim} (Node {n})".format(**strs))

	data_figure.savefig("draw-fits-node{n}.jpg".format(**strs))



