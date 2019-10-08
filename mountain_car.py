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

force=torch.tensor(0.001).type(torch.Tensor)
gravity=torch.tensor(0.0025).type(torch.Tensor)

dataInput =[
    (0.0, 0.03, 0),
    (0.2, 0.03, 0),
    (0.4, 0.03, 0),
    (0.5, 0.03, 0),
    (-0.3, 0.03, 0),
    (-0.7, 0.03, 0),
    (-1.0, 0.03, 0),
    (0.0, -0.03, 0),
    (0.2, -0.03, 0),
    (0.4, -0.03, 0),
    (0.5, -0.03, 0),
    (-0.3, -0.03, 0),
    (-0.7, -0.03, 0),
    (-1.0, -0.03, 0),
    (0.0, 0.03, 1),
    (0.2, 0.03, 1),
    (0.4, 0.03, 1),
    (0.5, 0.03, 1),
    (-0.3, 0.03, 1),
    (-0.7, 0.03, 1),
    (-1.0, 0.03, 1),
    (0.0, -0.03, 1),
    (0.2, -0.03, 1),
    (0.4, -0.03, 1),
    (0.5, -0.03, 1),
    (-0.3, -0.03, 1),
    (-0.7, -0.03, 1),
    (-1.0, -0.03, 1),
    (0.0, 0.03, 2),
    (0.2, 0.03, 2),
    (0.4, 0.03, 2),
    (0.5, 0.03, 2),
    (-0.3, 0.03, 2),
    (-0.7, 0.03, 2),
    (-1.0, 0.03, 2),
    (0.0, -0.03, 2),
    (0.2, -0.03, 2),
    (0.4, -0.03, 2),
    (0.5, -0.03, 2),
    (-0.3, -0.03, 2),
    (-0.7, -0.03, 2),
    (-1.0, -0.03, 2)
]

def step(state, action):
    position, velocity = state
    velocity = velocity + (action-1)*force + torch.cos(3*position)*(-gravity)
    velocity = torch.clamp(velocity, -max_speed, max_speed)
    position = position + velocity
    position = torch.clamp(position, min_position, max_position)
    if (position==min_position and velocity<0): velocity = torch.tensor(0.0)
    return (position, velocity)

def model(data):
    a0 = pyro.sample('a1', dist.Normal(torch.zeros(1), 10 * torch.ones(1)))
    a1 = pyro.sample('a2', dist.Normal(torch.zeros(1), 10 * torch.ones(1)))
    a2 = pyro.sample('a3', dist.Normal(torch.zeros(1), 10 * torch.ones(1)))
    a3 = pyro.sample('a4', dist.Normal(torch.zeros(1), 10 * torch.ones(1)))
    for i in range(len(data)):
        position = data[i][0]
        velocity = data[i][1]
        action = data[i][2]
        velocityExpected = data[i][3]
        velocity = a0 + a1 * velocity + a2 * position + a3 * velocity * position
        velocity = torch.clamp(velocity, -max_speed, max_speed)
        pyro.sample("obs_v_{}".format(i), dist.Normal(velocity, 0.1 * torch.ones([1]).type(torch.Tensor)), obs = velocityExpected)

def mcmc_solver():
    data = []
    for (p, v, a) in dataInput:
        _, vNext = step((torch.tensor(p), torch.tensor(v)), torch.tensor(a))
        data.append([p, v, a, vNext])
    nuts_kernel = NUTS(model, jit_compile=False,)
    mcmc = MCMC(nuts_kernel,
                    num_samples=100,
                    warmup_steps=100,
                    num_chains=1)
    data = torch.tensor(data)
    mcmc.run(data)
    mcmc.summary(prob=0.8)

def test(num):
    state = torch.tensor(0.0).type(torch.Tensor), torch.tensor(0.0).type(torch.Tensor)
    for i in range(num):
        print(state)
        next = step(state, torch.tensor(2.0).type(torch.Tensor))
        state = next

mcmc_solver()
# test(10)
