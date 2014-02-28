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
