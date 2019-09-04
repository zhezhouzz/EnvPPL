from __future__ import print_function
import math
import os
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
import numpy as np
from functools import reduce

# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
n_steps = 2 if smoke_test else 2000

# enable validation (e.g. validate parameters of distributions)
assert pyro.__version__.startswith('0.4.1')
pyro.enable_validation(True)

# clear the param store in case we're in a REPL
pyro.clear_param_store()

def trace_analyze(trace):
    assert(len(trace) > 1)
    res = []
    cur = trace[0]
    for i in range(1, len(trace)):
        next = trace[i]
        action = 1
        res.append([action, cur, next])
        cur = next
    return res

data = [
    [1.0, 1.0, 2.0], [1.0, 0.0, 1.0], [1.0, 2.0, 2.0],
    [1.0, 1.0, 2.0], [1.0, 0.0, 1.0], [1.0, 2.0, 2.0],
    [1.0, 1.0, 2.0], [1.0, 0.0, 1.0], [1.0, 2.0, 2.0],
    [1.0, 1.0, 2.0], [1.0, 0.0, 1.0], [1.0, 2.0, 2.0],
    [1.0, 1.0, 2.0], [1.0, 0.0, 1.0], [1.0, 2.0, 2.0],
    [1.0, 1.0, 2.0], [1.0, 0.0, 1.0], [1.0, 2.0, 2.0],
    [1.0, 1.0, 2.0], [1.0, 0.0, 1.0], [1.0, 2.0, 2.0],
    [1.0, 1.0, 2.0], [1.0, 0.0, 1.0], [1.0, 2.0, 2.0]]

# data = torch.tensor(reduce((lambda x, y: x + y), list(map(trace_analyze, traces))))
data = torch.tensor(data).double()
print(data)

def model(data):
    alpha = torch.tensor(np.full((2, 3), 4.0)).detach()
    beta = torch.tensor(np.full((2, 3), 4.0)).detach()
    mat_f = []
    for i in range(2):
        line = []
        for j in range(3):
            # print(i, j)
            line.append(pyro.sample("latent_{}{}".format(i, j), dist.Beta(alpha[i][j], beta[i][j])))
        mat_f.append(line)
    for i in range(len(data)):
        action = data[i][0]
        state = data[i][1]
        next = data[i][2]
        if action == 0:
            effect = state - next
        else:
            effect = next - state
        x = pyro.sample("obs_{}".format(i), dist.Bernoulli(mat_f[int(action)][int(state)]), obs=effect)

def guide(data):
    alpha = pyro.param("alpha_q", torch.tensor(np.full((2, 3), 4.0)).detach(),
                                constraint=constraints.positive)
    beta = pyro.param("beta_q", torch.tensor(np.full((2, 3), 4.0)).detach(),
                                constraint=constraints.positive)
    for i in range(2):
        for j in range(3):
            pyro.sample("latent_{}{}".format(i, j), dist.Beta(alpha[i][j], beta[i][j]))

# setup the optimizer
adam_params = {"lr": 0.1, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# do gradient steps
for step in range(n_steps):
    svi.step(data)
    if step % 100 == 0:
        print('.', end='')

alpha_q = pyro.param("alpha_q")
beta_q = pyro.param("beta_q")
print(alpha_q, beta_q)
inferred_mean = alpha_q / (alpha_q + beta_q)
# factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
# inferred_std = inferred_mean * math.sqrt(factor)
print(inferred_mean)
# print(inferred_mean, inferred_std)

