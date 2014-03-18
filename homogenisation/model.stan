
	data {

int<lower=2> N_0;
real x_measured_0[N_0];
real x_uncertainty_0[N_0];
real y_measured_0[N_0];
real y_uncertainty_0[N_0];

int<lower=2> N_1;
real x_measured_1[N_1];
real x_uncertainty_1[N_1];
real y_measured_1[N_1];
real y_uncertainty_1[N_1];

int<lower=2> N_2;
real x_measured_2[N_2];
real x_uncertainty_2[N_2];
real y_measured_2[N_2];
real y_uncertainty_2[N_2];

	}
	parameters {
	  real<lower=0> epsilon;

			  real<lower=0,upper=2> m_0;
			  real<lower=-1000,upper=1000> b_0;
			  real<lower=0,upper=1> p_0;
			  real<lower=0> Yb_0;
			  real<lower=0> Vb_0;
			  real x_0[N_0];

			  real<lower=0,upper=2> m_1;
			  real<lower=-1000,upper=1000> b_1;
			  real<lower=0,upper=1> p_1;
			  real<lower=0> Yb_1;
			  real<lower=0> Vb_1;
			  real x_1[N_1];

			  real<lower=0,upper=2> m_2;
			  real<lower=-1000,upper=1000> b_2;
			  real<lower=0,upper=1> p_2;
			  real<lower=0> Yb_2;
			  real<lower=0> Vb_2;
			  real x_2[N_2];

	}
	transformed parameters {
	  real mu_0[N_0];
real mu_1[N_1];
real mu_2[N_2];


  			  for(i in 1:N_0)
				mu_0[i] <- b_0 + m_0*x_0[i];

  			  for(i in 1:N_1)
				mu_1[i] <- b_1 + m_1*x_1[i];

  			  for(i in 1:N_2)
				mu_2[i] <- b_2 + m_2*x_2[i];

	}
	model {
	epsilon ~ normal(0, 1e2);

		  x_0 ~ normal(x_measured_0, x_uncertainty_0);
		  y_measured_0 ~ normal(mu_0, y_uncertainty_0);

		  m_0 ~ uniform(0, 2);
		  b_0 ~ uniform(-1000, 1000);
		  p_0 ~ uniform(0, 1);
		  Yb_0 ~ normal(0, 1e3);
		  Vb_0 ~ normal(0, 1e3);

		  x_1 ~ normal(x_measured_1, x_uncertainty_1);
		  y_measured_1 ~ normal(mu_1, y_uncertainty_1);

		  m_1 ~ uniform(0, 2);
		  b_1 ~ uniform(-1000, 1000);
		  p_1 ~ uniform(0, 1);
		  Yb_1 ~ normal(0, 1e3);
		  Vb_1 ~ normal(0, 1e3);

		  x_2 ~ normal(x_measured_2, x_uncertainty_2);
		  y_measured_2 ~ normal(mu_2, y_uncertainty_2);

		  m_2 ~ uniform(0, 2);
		  b_2 ~ uniform(-1000, 1000);
		  p_2 ~ uniform(0, 1);
		  Yb_2 ~ normal(0, 1e3);
		  Vb_2 ~ normal(0, 1e3);

	  {

			  real log_p_0;
			  real log1m_p_0;

			  real log_p_1;
			  real log1m_p_1;

			  real log_p_2;
			  real log1m_p_2;


			  log_p_0 <- log(p_0);
			  log1m_p_0 <- log1m(p_0);
			  for (i in 1:N_0)
				increment_log_prob(log_sum_exp(
			  	   log_p_0  + normal_log(y_measured_0[i], mu_0[i], y_uncertainty_0[i] + epsilon),
			  	  log1m_p_0 + normal_log(y_measured_0[i], Yb_0, Vb_0)));

			  log_p_1 <- log(p_1);
			  log1m_p_1 <- log1m(p_1);
			  for (i in 1:N_1)
				increment_log_prob(log_sum_exp(
			  	   log_p_1  + normal_log(y_measured_1[i], mu_1[i], y_uncertainty_1[i] + epsilon),
			  	  log1m_p_1 + normal_log(y_measured_1[i], Yb_1, Vb_1)));

			  log_p_2 <- log(p_2);
			  log1m_p_2 <- log1m(p_2);
			  for (i in 1:N_2)
				increment_log_prob(log_sum_exp(
			  	   log_p_2  + normal_log(y_measured_2[i], mu_2[i], y_uncertainty_2[i] + epsilon),
			  	  log1m_p_2 + normal_log(y_measured_2[i], Yb_2, Vb_2)));

	  }
	}