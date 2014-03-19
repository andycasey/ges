from pystan import stan, StanModel

model_code = """
data {

    // Dimensionality (teff, logg, etc)
    int<lower=1> N_dim;

    // Nodes named alpha, beta, ...
    int<lower=2> N_bm_alpha;
    real alpha_teff_measured[N_bm_alpha];
    real<lower=0> alpha_teff_uncertainty[N_bm_alpha];

    //ns stands for 'non-spectroscopic'
    real ns_measured_teff_alpha[N_bm_alpha];
    real<lower=0> ns_uncertainty_teff_alpha[N_bm_alpha];

    real alpha_logg_measured[N_bm_alpha];
    real<lower=0> alpha_logg_uncertainty[N_bm_alpha];
    real ns_measured_logg_alpha[N_bm_alpha];
    real<lower=0> ns_uncertainty_logg_alpha[N_bm_alpha];

    int<lower=2> N_bm_beta;
    real beta_teff_measured[N_bm_beta];
    real<lower=0> beta_teff_uncertainty[N_bm_beta];
    real ns_measured_teff_beta[N_bm_beta];
    real<lower=0> ns_uncertainty_teff_beta[N_bm_beta];
    
    real beta_logg_measured[N_bm_beta];
    real<lower=0> beta_logg_uncertainty[N_bm_beta];
    real ns_measured_logg_beta[N_bm_beta];
    real<lower=0> ns_uncertainty_logg_beta[N_bm_beta];
}  
parameters {

    // Spectroscopic/non-spectroscopic parameter relations for alpha
    real<lower=0,upper=2> m_teff_alpha;
    real<lower=-1000,upper=1000> b_teff_alpha;
    real<lower=0,upper=2> m_logg_alpha;
    real<lower=-5,upper=5> b_logg_alpha;

    // Non-spectroscopic measured distributions for alpha node
    real ns_teff_alpha[N_bm_alpha];
    real ns_logg_alpha[N_bm_alpha];

    // The measured distributions of teff and logg by alpha node
    real teff_alpha[N_bm_alpha];
    real logg_alpha[N_bm_alpha];

    // Spectroscopic/non-spectroscopic parameter relations for beta
    real<lower=0,upper=2> m_teff_beta;
    real<lower=-1000,upper=1000> b_teff_beta;
    real<lower=0,upper=2> m_logg_beta;
    real<lower=-5,upper=5> b_logg_beta;

    // Non-spectroscopic measured distributions for beta node
    real ns_teff_beta[N_bm_beta];
    real ns_logg_beta[N_bm_beta];

    // The measured distributions of teff and logg by beta node
    real teff_beta[N_bm_beta];
    real logg_beta[N_bm_beta];

    // Intrinsic dispersions
    real<lower=0> s_intrinsic_teff;
    real<lower=0> s_intrinsic_logg;

    // Node dispersions
    real<lower=0> s_alpha_teff;
    real<lower=0> s_alpha_logg;
    real<lower=0> s_beta_teff;
    real<lower=0> s_beta_logg;

    // Latent outlier variable p0 (avoiding alpha, gamma, etc as they
    // refer to nodes)
    real<lower=0,upper=1> p0;
    
    // Model background
    real<lower=0> mu_background_teff;
    real<lower=0> sigma_background_teff;

    real<lower=0> mu_background_logg;
    real<lower=0> sigma_background_logg;
}
transformed parameters {
    // Transform onto non-spectroscopic scale
    real sp_teff_alpha[N_bm_alpha];
    real sp_logg_alpha[N_bm_alpha];

    real sp_teff_beta[N_bm_beta];
    real sp_logg_beta[N_bm_beta];

    // Alpha node
    for (i in 1:N_bm_alpha)
    {
      sp_teff_alpha[i] <- m_teff_alpha*teff_alpha[i] + b_teff_alpha;
      sp_logg_alpha[i] <- m_logg_alpha*logg_alpha[i] + b_logg_alpha;
    }

    // Beta node
    for (i in 1:N_bm_beta)
    {
      sp_teff_beta[i] <- m_teff_beta*teff_beta[i] + b_teff_beta;
      sp_logg_beta[i] <- m_logg_beta*logg_beta[i] + b_logg_beta;
    }
}
model {

    // Outlier variables
    real log_p0;
    real log1m_p0;

    // Create our vectors and matrices. Currently N_dim assumed to be 2 (teff, logg),
    // but this can be generalised
    vector[N_dim] ns_measurements;
    vector[N_dim] sp_measurements;
    matrix[N_dim,N_dim] covariance;
    vector[N_dim] sp_correlated_measurements;

    // Non-spectroscopic measurements
    ns_teff_alpha ~ normal(ns_measured_teff_alpha, ns_uncertainty_teff_alpha);
    ns_teff_beta ~ normal(ns_measured_teff_beta, ns_uncertainty_teff_beta);

    ns_logg_alpha ~ normal(ns_measured_logg_alpha, ns_uncertainty_logg_alpha);
    ns_logg_beta ~ normal(ns_measured_logg_beta, ns_uncertainty_logg_beta);

    // Node measurement variances
    s_alpha_teff ~ normal(0, 1e2);
    s_beta_teff ~ normal(0, 1e2);

    s_alpha_logg ~ normal(0, 1);
    s_beta_logg ~ normal(0, 1);

    // Intrinsic variances
    s_intrinsic_teff ~ normal(0, 1e2);
    s_intrinsic_logg ~ normal(0, 1);

    // Measurements (EXAMPLE ONLY -- this is for if we were not using a covariance matrix)
    //y_teff_alpha ~ normal(sp_teff_alpha, pow(alpha_teff_uncertainty, 2) + s_alpha_teff + s_intrinsic_teff);
    //y_teff_beta ~ normal(sp_teff_beta, pow(beta_teff_uncertainty, 2) + s_beta_teff + s_intrinsic_teff);

    //y_logg_alpha ~ normal(sp_logg_alpha, pow(logg_uncertainty_alpha, 2) + s_alpha_logg + s_intrinsic_logg);
    //y_logg_beta ~ normal(sp_logg_beta, pow(logg_uncertainty_beta, 2) + s_beta_logg + s_intrinsic_logg);

    {
        // Outlier variables
        log_p0 <- log(p0);
        log1m_p0 <- log1m(p0);

        // Do alpha node first
        for (i in 1:N_bm_alpha)
        {
            // Non-spectroscopic measurements
            ns_measurements[1] <- ns_teff_alpha[i];
            ns_measurements[2] <- ns_logg_alpha[i];

            // Transformed spectroscopic measurements for alpha node
            sp_measurements[1] <- sp_teff_alpha[i];
            sp_measurements[2] <- sp_logg_alpha[i];

            // Covariance matrix
            // Consider the simple case with no covariance:
            covariance[1,2] <- 0;
            covariance[2,1] <- 0;
            covariance[1,1] <- pow(alpha_teff_uncertainty[i], 2) + s_alpha_teff + s_intrinsic_teff;
            covariance[2,2] <- pow(alpha_logg_uncertainty[i], 2) + s_alpha_logg + s_intrinsic_logg;

            sp_correlated_measurements ~ multi_normal(sp_measurements, covariance);

            // Adding these separately,.. doesn't feel right.
            increment_log_prob(log_sum_exp(
                 log_p0  + normal_log(ns_measurements, sp_correlated_measurements[1], pow(alpha_teff_uncertainty[i], 2) + s_alpha_teff + s_intrinsic_teff),
                 
                // Not considering covariance in background measurements -- should we be applying the
                // same covariance matrix to the background parameters?
                log1m_p0 + normal_log(ns_teff_alpha[i], mu_background_teff, sigma_background_teff)
            ));
            
            increment_log_prob(log_sum_exp(
                 log_p0  + normal_log(ns_measurements, sp_correlated_measurements[2], pow(alpha_logg_uncertainty[i], 2) + s_alpha_logg + s_intrinsic_logg),
                log1m_p0 + normal_log(ns_logg_alpha[i], mu_background_logg, sigma_background_logg)
            ));
        }

        // Repeat for beta node
        for (i in 1:N_bm_beta)
        {
            // Non-spectroscopic measurements
            ns_measurements[1] <- ns_teff_beta[i];
            ns_measurements[2] <- ns_logg_beta[i];

            // Transformed spectroscopic measurements for beta node
            sp_measurements[1] <- sp_teff_beta[i];
            sp_measurements[2] <- sp_logg_beta[i];

            // Covariance matrix
            // Consider the simple case with no covariance:
            covariance[1,2] <- 0;
            covariance[2,1] <- 0;
            covariance[1,1] <- pow(beta_teff_uncertainty[i], 2) + s_beta_teff + s_intrinsic_teff;
            covariance[2,2] <- pow(beta_logg_uncertainty[i], 2) + s_beta_logg + s_intrinsic_logg;

            sp_correlated_measurements ~ multi_normal(sp_measurements, covariance);

            // Adding these separately,.. doesn't feel right.
            increment_log_prob(log_sum_exp(
                 log_p0  + normal_log(ns_measurements, sp_correlated_measurements[1], pow(beta_teff_uncertainty[i], 2) + s_beta_teff + s_intrinsic_teff),
                 
                // Not considering covariance in background measurements -- should we be applying the
                // same covariance matrix to the background parameters?
                log1m_p0 + normal_log(ns_teff_beta[i], mu_background_teff, sigma_background_teff)
            ));
            
            increment_log_prob(log_sum_exp(
                 log_p0  + normal_log(ns_measurements, sp_correlated_measurements[2], pow(beta_logg_uncertainty[i], 2) + s_beta_logg + s_intrinsic_logg),
                log1m_p0 + normal_log(ns_logg_beta[i], mu_background_logg, sigma_background_logg)
            ));
        }
    }

}
"""

sm = StanModel(model_code=model_code)
