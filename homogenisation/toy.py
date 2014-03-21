
import json
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
	real<lower=0> s_teff_intrinsic;
	real<lower=0> s_logg_intrinsic;

	//	- Node 0
	real<lower=0> s_teff_node0;
	real<lower=0> s_logg_node0;

	//	- Node 1
	real<lower=0> s_teff_node1;
	real<lower=0> s_logg_node1;

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

    s_teff_intrinsic ~ normal(0, 1e2);
    s_logg_intrinsic ~ normal(0, 1);

    s_teff_node0 ~ normal(0, 10);
    s_logg_node0 ~ normal(0, 0.5);

    s_teff_node1 ~ normal(0, 10);
    s_logg_node1 ~ normal(0, 0.5);

    

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
    	covariance[1,1] <- pow(sp_teff_node0_sigma[i], 2) + s_teff_node0 + s_teff_intrinsic;
    	covariance[2,2] <- pow(sp_logg_node0_sigma[i], 2) + s_logg_node0 + s_logg_intrinsic;
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
    	covariance[1,1] <- pow(sp_teff_node1_sigma[i], 2) + s_teff_node1 + s_teff_intrinsic;
    	covariance[2,2] <- pow(sp_logg_node1_sigma[i], 2) + s_logg_node1 + s_logg_intrinsic;
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


print("Fitting")
fit = model.sampling(data=data, pars=op["par"], iter=5000)

nodes, dimensions = range(2), ("teff", "logg")
dimensions_traced = []
for node in nodes:
	node_dimensions = \
		["m_{dim}_node{n}".format(dim=dimension, n=node) for dimension in dimensions] \
	  + ["b_{dim}_node{n}".format(dim=dimension, n=node) for dimension in dimensions]
	fig = fit.traceplot(node_dimensions)
	fig.savefig("trace-node-{0}.pdf".format(node))
	dimensions_traced.extend(node_dimensions)

other_dimensions = set(op["par"].keys()).difference(dimensions_traced)
fig = fit.traceplot(other_dimensions)
fig.savefig("other-dimensions.pdf")

