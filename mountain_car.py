from __future__ import absolute_import, division, print_function

import argparse
import logging

import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc.api import MCMC
import numpy as np

logging.basicConfig(format='%(message)s', level=logging.INFO)
pyro.enable_validation(__debug__)
pyro.set_rng_seed(0)

min_position = -1.2
max_position = 0.6
max_speed = 0.07
goal_position = 0.5
goal_velocity = 0

force=torch.tensor(0.001)
gravity=torch.tensor(0.0025)

low = np.array([min_position, -max_speed])
high = np.array([max_position, max_speed])

def step(state, action):
    position, velocity = state
    velocity += (action-1)*force + np.cos(3*position)*(-gravity)
    velocity = np.clip(velocity, -max_speed, max_speed)
    position += velocity
    position = np.clip(position, min_position, max_position)
    if (position==min_position and velocity<0): velocity = torch.tensor(0.0)
    return (position, velocity)

def model(data):
    a0 = pyro.sample('a0', dist.Normal(torch.zeros(1) + 3.0, 10 * torch.ones(1)))
    for i in range(100):
        position = dist.Uniform(min_position, max_position).sample()
        velocity = dist.Uniform(-max_speed, max_speed).sample()
        action = dist.Categorical(probs=torch.ones(3)/torch.sum(torch.ones(3))).sample()
        # print(action)
        # print("init: {} {} {}".format(position, velocity, action))
        p, v = step((position, velocity), action)
        # print("p v: {} {}".format(p, v))
        # print((action-1)*force)
        # print(a0*position)
        # print((action-1)*force + np.cos(a0*position)*(-gravity))
        e1 = (action-1)*force
        e2 = torch.cos(a0*position)*(-gravity)
        velocity = velocity + e1 + e2
        velocity = torch.clamp(velocity, -max_speed, max_speed)
        position = position + velocity
        position = torch.clamp(position, min_position, max_position)
        if (position==min_position and velocity<0): velocity = torch.tensor(0.0)
        pyro.sample("obs_p_{}".format(i), dist.Normal(position, 0.1 * torch.ones([1]).type(torch.Tensor)), obs = p)
        pyro.sample("obs_v_{}".format(i), dist.Normal(velocity, 0.1 * torch.ones([1]).type(torch.Tensor)), obs = v)

def mcmc_solver():
    nuts_kernel = NUTS(model, jit_compile=False,)
    mcmc = MCMC(nuts_kernel,
                num_samples=1000,
                warmup_steps=1000,
                num_chains=1)
    mcmc.run(model)
    mcmc.summary(prob=0.8)

def test(num):
    state = torch.tensor(0.0), torch.tensor(0.0)
    for i in range(num):
        print(state)
        next = step(state, torch.tensor(2.0))
        state = next

mcmc_solver()
# test(10)
