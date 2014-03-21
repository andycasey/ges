
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas

from pystan import StanModel


model_code = """
data {

    // Two dimensional case (teff, logg) for 2 nodes

    // Node 0
    int<lower=2> N_benchmarks_node0;
    
    //	- Spectroscopic measurements
    real sp_teff_node0_measured[N_benchmarks_node0];
    real sp_teff_node0_sigma[N_benchmarks_node0];
    real sp_logg_node0_measured[N_benchmarks_node0];
    real sp_logg_node0_sigma[N_benchmarks_node0];

    //	- Non-spectroscopic measurements
    real ns_teff_node0_measured[N_benchmarks_node0];
    real ns_teff_node0_sigma[N_benchmarks_node0];
    real ns_logg_node0_measured[N_benchmarks_node0];
    real ns_logg_node0_sigma[N_benchmarks_node0];

    // Node 1
    int<lower=2> N_benchmarks_node1;

    //	- Spectroscopic measurements
    real sp_teff_node1_measured[N_benchmarks_node1];
    real sp_teff_node1_sigma[N_benchmarks_node1];
    real sp_logg_node1_measured[N_benchmarks_node1];
    real sp_logg_node1_sigma[N_benchmarks_node1];

    //	- Non-spectroscopic measurements
    real ns_teff_node1_measured[N_benchmarks_node1];
    real ns_teff_node1_sigma[N_benchmarks_node1];
    real ns_logg_node1_measured[N_benchmarks_node1];
    real ns_logg_node1_sigma[N_benchmarks_node1];
}  
parameters {

	// Parameter dispersions
	//	- Intrinsic
	//real<lower=0> s_teff_intrinsic;
	//real<lower=0> s_logg_intrinsic;

	//	- Node 0
	//real<lower=0> s_teff_node0;
	//real<lower=0> s_logg_node0;

	//	- Node 1
	//real<lower=0> s_teff_node1;
	//real<lower=0> s_logg_node1;

	// Parameter covariances
	// Commenting out parameter covariance because this is a toy model
	real<lower=-1,upper=1> rho_teff_logg;

	// Spectroscopic to non-spectroscopic parameter relations
	//	- Node 0
	real m_teff_node0;
	real b_teff_node0;
	real m_logg_node0;
	real b_logg_node0;

	//	- Node 1
	real m_teff_node1;
	real b_teff_node1;
	real m_logg_node1;
	real b_logg_node1;

	// Outlier distributions
	real alpha;
	real outlier_teff_mu;
	real<lower=0> outlier_teff_sigma;
	real outlier_logg_mu;
	real<lower=0> outlier_logg_sigma;

}
model {

	// Declarations

	real log_alpha;
    real log1m_alpha;

    vector[2] ns_vector;
    vector[2] sp_vector;
    matrix[2,2] covariance;

    alpha ~ normal(0, 1);

    log_alpha <- log(alpha);
	log1m_alpha <- log1m(alpha);

    // Initialise the model
    m_teff_node0 ~ normal(1, 0.1);
    b_teff_node0 ~ normal(0, 50);

    m_teff_node1 ~ normal(1, 0.1);
    b_teff_node1 ~ normal(0, 50);

    m_logg_node0 ~ normal(1, 0.1);
    b_logg_node0 ~ normal(0, 0.5);

    m_logg_node1 ~ normal(1, 0.1);
    b_logg_node1 ~ normal(0, 0.5);

    rho_teff_logg ~ uniform(-1, 1);

    outlier_teff_mu ~ normal(0, 1e3);
    outlier_teff_sigma ~ normal(0, 1e3);

    outlier_logg_mu ~ normal(0, 1);
    outlier_logg_sigma ~ normal(0, 1);

    //s_teff_intrinsic ~ normal(0, 1e2);
    //s_logg_intrinsic ~ normal(0, 1);

    //s_teff_node0 ~ normal(0, 10);
    //s_logg_node0 ~ normal(0, 0.5);

    //s_teff_node1 ~ normal(0, 10);
    //s_logg_node1 ~ normal(0, 0.5);

    

    // 	- Node 0
    // We have to sample from a multi-dimensional gaussian and the covariance matrix.

    // Then we need to compare this value against a vector containing the non-spectroscopic
    // values (ns_teff_node0, ns_logg_node0)

    for (i in 1:N_benchmarks_node0) {

    	// Non-spectroscopic measurements
    	ns_vector[1] <- ns_teff_node0_measured[i];
    	ns_vector[2] <- ns_logg_node0_measured[i];

    	// Spectroscopic measurements
    	sp_vector[1] <- m_teff_node0 * sp_teff_node0_measured[i] + b_teff_node0;
    	sp_vector[2] <- m_logg_node0 * sp_logg_node0_measured[i] + b_logg_node0;

    	// Covariance matrix
    	covariance[1,1] <- pow(sp_teff_node0_sigma[i], 2); // + s_teff_node0 + s_teff_intrinsic;
    	covariance[2,2] <- pow(sp_logg_node0_sigma[i], 2); // + s_logg_node0 + s_logg_intrinsic;
    	covariance[1,2] <- sqrt(covariance[1,1] * covariance[2,2]) * rho_teff_logg;
    	covariance[2,1] <- sqrt(covariance[1,1] * covariance[2,2]) * rho_teff_logg;

    	// Now we need to add to the log likelihood by sampling from this multi-dimensional
    	// Gaussian of spectroscopic measurements and compare it to the non-spectroscopic
    	// measurements. We also need to scale by our outlier parameter.

    	increment_log_prob(
    		log_alpha + multi_normal_log(ns_vector, sp_vector, covariance)
    	);

		// We also need to add to the log likelihood by considering the outliers
		// We will do this in separate dimensions (teff, logg) because we are not considering
		// the case that the outliers are covariant in the same way we believe the measurements
		// are

		increment_log_prob(log_sum_exp(
			log1m_alpha + normal_log(sp_vector[1], outlier_teff_mu, outlier_teff_sigma),
			log1m_alpha + normal_log(sp_vector[2], outlier_teff_mu, outlier_teff_sigma)
		));
    }


	// 	- Node 1
    // Copypasta from above.

    for (i in 1:N_benchmarks_node1) {

    	// Non-spectroscopic measurements
    	ns_vector[1] <- ns_teff_node1_measured[i];
    	ns_vector[2] <- ns_logg_node1_measured[i];

    	// Spectroscopic measurements
    	sp_vector[1] <- m_teff_node1 * sp_teff_node1_measured[i] + b_teff_node1;
    	sp_vector[2] <- m_logg_node1 * sp_logg_node1_measured[i] + b_logg_node1;

    	// Covariance matrix
    	covariance[1,1] <- pow(sp_teff_node1_sigma[i], 2); // + s_teff_node1 + s_teff_intrinsic;
    	covariance[2,2] <- pow(sp_logg_node1_sigma[i], 2); // + s_logg_node1 + s_logg_intrinsic;
    	covariance[1,2] <- sqrt(covariance[1,1] * covariance[2,2]) * rho_teff_logg;
    	covariance[2,1] <- sqrt(covariance[1,1] * covariance[2,2]) * rho_teff_logg;
    	
    	// Now we need to add to the log likelihood by sampling from this multi-dimensional
    	// Gaussian of spectroscopic measurements and compare it to the non-spectroscopic
    	// measurements. We also need to scale by our outlier parameter.

    	increment_log_prob(
    		log_alpha + multi_normal_log(ns_vector, sp_vector, covariance)
    	);

		// We also need to add to the log likelihood by considering the outliers
		// We will do this in separate dimensions (teff, logg) because we are not considering
		// the case that the outliers are covariant in the same way we believe the measurements
		// are

		increment_log_prob(log_sum_exp(
			log1m_alpha + normal_log(sp_vector[1], outlier_teff_mu, outlier_teff_sigma),
			log1m_alpha + normal_log(sp_vector[2], outlier_teff_mu, outlier_teff_sigma)
		));
    }
}"""

# Ok, here is our toy data:
with open("toy.data", "r") as fp:
	data = json.load(fp)

model = StanModel(model_code=model_code)

print("Optimizing...")
op = model.optimizing(data=data)

print("Fitting...")
fit = model.sampling(data=data, pars=op["par"], iter=20000)

subplots_adjust = { "left": 0.10, "bottom": 0.05, "right": 0.95, "top": 0.95,
	"wspace": 0.20, "hspace": 0.45
	}

nodes = range(2)
dimensions = ("teff", "logg")

# Plot the m, b parameters for each node
dimensions_traced = []
for node in nodes:
	node_dimensions = \
		["m_{dim}_node{n}".format(dim=dimension, n=node) for dimension in dimensions] \
	  + ["b_{dim}_node{n}".format(dim=dimension, n=node) for dimension in dimensions]
	dimensions_traced.extend(node_dimensions)

	fig = fit.traceplot(node_dimensions)
	fig.subplots_adjust(**subplots_adjust)
	fig.savefig("trace-node-{0}.jpg".format(node))

# Plot the posterior of intrinsic and node scatter
scatter_dimensions = ["s_{dim}_intrinsic".format(dim=dimension) for dimension in dimensions]
[scatter_dimensions.extend(["s_{dim}_node{n}".format(dim=dimension, n=node) \
	for dimension in dimensions]) for node in nodes]
scatter_dimensions = list(set(scatter_dimensions).intersection(op["par"]))

if len(scatter_dimensions) > 0:
	# Plot the posteriors of the scatter
	fig = fit.traceplot()
	fig.subplots_adjust(**subplots_adjust)
	fig.savefig("scatter.jpg")

# Plot the posterior of the outlier parameters
outlier_dimensions = set(op["par"]).difference(dimensions_traced + scatter_dimensions)
fig = fit.traceplot(outlier_dimensions)
fig.subplots_adjust(**subplots_adjust)
fig.savefig("outlier-dimensions.jpg")


# Draw from the distributions 
samples = fit.extract(permuted=True)

for node in nodes:

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



