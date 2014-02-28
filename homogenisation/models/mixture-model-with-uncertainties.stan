data {
  int<lower=2> N;
  real x_measured[N];
  real<lower=0> x_uncertainty[N];
  real y_measured[N];
  real<lower=0> y_uncertainty[N];
  real<lower=0> xy_covariance[N];
}
parameters {
  real theta;
  real b_t;
  real<lower=0> V;
  real x[N];
}
transformed parameters {

  real delta[N];
  real sigma_sq[i];


  for(i in 1:N)
    delta[i] = dot_product([-sin(theta), cos(theta)], x[i], y[i]) - b_t
    sigma_sq[i] = dot_product(dot_product([-sin(theta), cos(theta)], [[square(x_uncertainty[i]), xy_covariance[i]], [xy_covariance[i], square(y_uncertainty[i])]), [[-sin(theta)], [cos(theta)]])


}
model {
  x ~ normal(x_measured, x_uncertainty);

  for (i in 1:N)
    increment_log_prob(-0.5 * log(sigma_sq[i] + V))
    increment_log_prob(log_sum_exp(
      -normal_log(delta)
    ))

}



data {
 int<lower=2> N; // number of observations
 real x_measured[N];
 real y_measured[N];
 real x_uncertainty[N];
 real y_uncertainty[N];
}
parameters {
 real<lower=0> m;
 real<lower=0> b;
 real<lower=0,upper=1> p1;
 real<lower=0> Yb;
 real<lower=0> Vb;
 real x[N];
}
transformed parameters {
 real mu[N];

 for(i in 1:N)
  mu[i] <- b + m*x[i];
}
model {
 x ~ normal(x_measured, x_uncertainty);
 y_measured ~ normal(mu, y_uncertainty);

 m ~ normal(0, 5);
 b ~ normal(0, 50);
 p1 ~ uniform(0, 1); //rec
 Yb ~ normal(0, 1e2);
 Vb ~ normal(0, 1e2); //req

 {
  real log_p1;
  real log1m_p1;
  log_p1 <- log(p1);
  log1m_p1 <- log1m(p1);
  for (n in 1:N)
   increment_log_prob(log_sum_exp(
    log_p1 + normal_log(y_measured[n], mu[n], y_uncertainty[n]),
    log1m_p1 + normal_log(y_measured[n], Yb, Vb)));
 }
}
