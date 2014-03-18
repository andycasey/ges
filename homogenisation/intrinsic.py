model_code = """
data {

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

    // The measured distributions of teff and logg by alpha node
    real teff_alpha[N_bm_alpha];
    real logg_alpha[N_bm_alpha];

    // Spectroscopic/non-spectroscopic parameter relations for beta
    real<lower=0,upper=2> m_teff_beta;
    real<lower=-1000,upper=1000> b_teff_beta;
    real<lower=0,upper=2> m_logg_beta;
    real<lower=-5,upper=5> b_logg_beta;

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

    // Latent outlier variable
    real<lower=0,upper=1> p_0;
    
    // Model background
    real<lower=0> mu_background_teff;
    real<lower=0> sigma_background_teff;

    real<lower=0> mu_background_logg;
    real<lower=0> sigma_background_logg;
}
transformed parameters {
    // Transform onto non-spectroscopic scale
    real transformed_teff_alpha[N_bm_alpha];
    real transformed_logg_alpha[N_bm_alpha];

    real transformed_teff_beta[N_bm_beta];
    real transformed_logg_beta[N_bm_beta];

    // Alpha node
    for(i in 1:N_bm_alpha)
      transformed_teff_alpha[i] <- m_teff_alpha*teff_alpha[i] + b_teff_alpha
      transformed_logg_alpha[i] <- m_logg_alpha*logg_alpha[i] + b_logg_alpha

    // Beta node
    for(i in 1:N_bm_beta)
      transformed_teff_beta[i] <- m_teff_beta*teff_beta[i] + b_teff_beta
      transformed_logg_beta[i] <- m_logg_beta*logg_beta[i] + b_logg_beta
}
model {

    // Non-spectroscopic measurements
    ns_teff_alpha ~ normal(teff_ns_measured_alpha, teff_ns_uncertainty_alpha);
    ns_teff_beta ~ normal(teff_ns_measured_beta, teff_ns_uncertainty_beta);

    logg_true_alpha ~ normal(logg_ns_measured_alpha, logg_ns_uncertainty_alpha);
    logg_true_beta ~ normal(logg_ns_measured_beta, logg_ns_uncertainty_beta);

    // Node dispersions
    s_alpha_teff ~ normal(0, 1e2);
    s_beta_teff ~ normal(0, 1e2);

    s_alpha_logg ~ normal(0, 1);
    s_beta_logg ~ normal(0, 1);

    // Intrinsic dispersions
    s_intrinsic_teff ~ normal(0, 1e2);
    s_intrinsic_logg ~ normal(0, 1);

    // Background
    background_teff ~ normal(mu_background_teff, sigma_background_teff);
    background_logg ~ normal(mu_background_logg, sigma_background_logg);

    // Measurements
    y_teff_alpha ~ normal(mu_teff_alpha, teff_uncertainty_alpha + eps_teff_alpha + eps_intrinsic_teff)
    y_teff_beta ~ normal(mu_teff_beta, teff_uncertainty_beta + eps_teff_beta + eps_intrinsic_teff)

    y_logg_alpha ~ normal(mu_logg_alpha, logg_uncertainty_alpha + eps_logg_alpha + eps_intrinsic_logg)
    y_logg_beta ~ normal(mu_logg_beta, logg_uncertainty_beta + eps_logg_beta + eps_intrinsic_logg)


}
"""