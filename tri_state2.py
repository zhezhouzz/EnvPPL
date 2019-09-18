from __future__ import absolute_import, division, print_function

import argparse
import logging

import torch

import data
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc.api import MCMC
import numpy as np

logging.basicConfig(format='%(message)s', level=logging.INFO)
pyro.enable_validation(__debug__)
pyro.set_rng_seed(0)

y = torch.tensor([
    1, 2, 3, 4,
    2, 3, 4,
    4,
    2, 1, 2, 3, 4]).type(torch.Tensor)
data = torch.tensor([
    [0, 1], [1, 1], [2, 1], [3, 1],
    [1, 1], [2, 1], [3, 1],
    [4, 1],
    [1, 1], [2, 0], [1, 1], [2, 1], [3, 1]]).type(torch.Tensor)

def model(data):
    a0 = pyro.sample('a0', dist.Normal(torch.zeros(1) - 0.4, 10 * torch.ones(1)))
    a1 = pyro.sample('a1', dist.Normal(torch.zeros(1) + 1.6, 10 * torch.ones(1)))
    a2 = pyro.sample('a2', dist.Normal(torch.zeros(1) + 0.8, 10 * torch.ones(1)))
    res = []
    for i in range(len(data)):
        # state = dist.Categorical(probs=torch.ones(3)/torch.isum(torch.ones(3))).sample()
        # state_next = dist.Categorical(probs=torch.ones(3)/torch.sum(torch.ones(3))).sample()
        # action = dist.Categorical(probs=torch.ones(2)/torch.sum(torch.ones(2))).sample()
        state = data[i][0]
        action = data[i][1]
        theta = a2 * state + action * a1 + a0
        res.append(theta)
    theta = torch.tensor(res)
    return pyro.sample("obs", dist.Normal(theta, 0.1 * torch.ones([len(data)]).type(torch.Tensor)))
    # return pyro.sample("obs", dist.Normal(torch.tensor(res, dtype=torch.float64), 0.1 * torch.ones(1, dtype=torch.float64)))


def conditioned_model(model, d, y):
    # print("y: {}". format(y))
    return poutine.condition(model, data={"obs": torch.tensor(y)})(d)


def main(args):
    nuts_kernel = NUTS(conditioned_model, jit_compile=args.jit,)
    mcmc = MCMC(nuts_kernel,
                num_samples=args.num_samples,
                warmup_steps=args.warmup_steps,
                num_chains=args.num_chains)
    mcmc.run(model, data, y)
    mcmc.summary(prob=0.8)
    # a0 = mcmc.get_samples(10)['a0']
    # a1 = mcmc.get_samples(10)['a1']
    # print("a0 = {}, a1 = {}".format(a0, a1))
    # tau = mcmc.get_samples(1)['tau']
    # theta = mu + tau * eta
    # print(pyro.sample("obs", dist.Normal(theta, data.sigma)))


if __name__ == '__main__':
    # assert pyro.__version__.startswith('0.4.0')
    parser = argparse.ArgumentParser(description='Eight Schools MCMC')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='number of MCMC samples (default: 1000)')
    parser.add_argument('--num-chains', type=int, default=1,
                        help='number of parallel MCMC chains (default: 1)')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='number of MCMC samples for warmup (default: 1000)')
    parser.add_argument('--jit', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
