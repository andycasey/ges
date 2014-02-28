# coding: utf-8

""" Model results from a Gaia-ESO Survey Node """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@cam.ast.ac.uk>"

# Standard libraries
import logging

# Third party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystan


#def main():
# Load the data for the toy model
ids, x, y, xerr, yerr, pxy = np.loadtxt("hogg_data.dat", unpack=True)

model_code = """
data {
 int<lower = 0> N; // number of observations
 real x_measured[N];
 real y_measured[N];
 real x_uncertainty[N];
 real y_uncertainty[N];
}
parameters {
 real m;
 real b;
 real<lower=0,upper=1> p1;
 real Yb;
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

 m ~ normal(2.25, 0.1);
 b ~ normal(34, 5);
 p1 ~ uniform(0, 1);
 Yb ~ normal(400, 50);
 Vb ~ normal(0, 100);

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
"""


# Fit the model
fit = pystan.stan(model_code=model_code, iter=1000, chains=4,
    data={
        "x_measured": x, "x_uncertainty": xerr,
        "y_measured": y, "y_uncertainty": yerr,
        "N": len(x)
    })

print(fit)
fit.traceplot()

samples = fit.extract(permuted=True)
parameters = pd.DataFrame({"m": samples["m"], "b": samples["b"]})

# Predictive model
pred_x = np.arange(0, 300)
model = lambda theta: pd.Series({"fitted": theta[0] + theta[1] * pred_x})

median_parameters = parameters.median()

yhat = model(median_parameters)

# get the predicted values for each chain
chain_predictions = parameters.apply(model, axis=1)

fig = plt.figure()
ax = fig.add_subplot(111)

num_chains = 50
indices = np.random.choice(300, num_chains)

for i, index in enumerate(indices):
    ax.plot(pred_x, chain_predictions.iloc[index, 0], color="lightgrey")

#  data
ax.errorbar(x, y, yerr=yerr, fmt=None, facecolor="k", ecolor="k")
ax.plot(x, y, 'ko')

# fitted values
ax.plot(pred_x, yhat["fitted"], "k", lw=2)

# supplementals
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.show()



#if __name__ == "__main__":
#    result = main()
