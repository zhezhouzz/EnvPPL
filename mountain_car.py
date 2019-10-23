from __future__ import absolute_import, division, print_function

import sys
import argparse
import logging

import math
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

def space_maker (l):
    l.reverse()
    intermid = []
    for (tp, (dmin, dmax, step)) in l:
        res = []
        for d in np.arange(dmin, dmax, step):
            if tp == "discrete":
                d = int(d)
            else:
                d = float(d)
            if len(intermid) == 0:
                res.append([d])
            else:
                res = res + [[d] + one for one in intermid]
        intermid = res
    return intermid

d0 = space_maker([("continous", (-0.3, 0.0, 0.1)), ("continous", (-0.3, 0.3, 0.1)), ("discrete", (0, 3, 1))])
d1 = space_maker([("continous", (0.0, 0.3, 0.1)), ("continous", (-0.3, 0.3, 0.1)), ("discrete", (0, 3, 1))])
d2 = space_maker([("continous", (0.3, 0.6, 0.1)), ("continous", (-0.3, 0.3, 0.1)), ("discrete", (0, 3, 1))])

data_input = [d0, d1, d2]

def step(state, action):
    position, velocity = state
    velocity = velocity + (action-1)*force + torch.cos(3*position)*(-gravity)
    velocity = torch.clamp(velocity, -max_speed, max_speed)
    position = position + velocity
    position = torch.clamp(position, min_position, max_position)
    if (position==min_position and velocity<0): velocity = torch.tensor(0.0)
    return (position, velocity)

def stepAppr(a0, a1, state, action):
    a0 = torch.tensor(a0).type(torch.Tensor)
    a1 = torch.tensor(a1).type(torch.Tensor)
    position, velocity = state
    velocity = velocity + (action-1)*force + torch.cos(a0*position)*(-gravity)
    velocity = torch.clamp(velocity, -max_speed, max_speed)
    position = position + a1 * velocity
    position = torch.clamp(position, min_position, max_position)
    if (position==min_position and velocity<0): velocity = torch.tensor(0.0)
    return (position, velocity)

def model(data):
    a0 = pyro.sample('a0', dist.Normal(torch.zeros(1), 10 * torch.ones(1)))
    a1 = pyro.sample('a1', dist.Normal(torch.zeros(1), 10 * torch.ones(1)))
    # a2 = pyro.sample('a2', dist.Normal(torch.zeros(1), 10 * torch.ones(1)))
    # a3 = pyro.sample('a3', dist.Normal(torch.zeros(1), 10 * torch.ones(1)))
    for i in range(len(data)):
        position = data[i][0]
        velocity = data[i][1]
        action = data[i][2]
        p, v = step((position, velocity), action)
        p, v = step((p, v), action)
        velocity = velocity + (action-1)*force + torch.cos(a0*position)*(-gravity)
        velocity = torch.clamp(velocity, -max_speed, max_speed)
        position = position + a1 * velocity
        position = torch.clamp(position, min_position, max_position)
        if (position==min_position and velocity<0): velocity = torch.tensor(0.0)
        # print("f0 = {}".format(f0))
        # print("(p, v) = ({}, {})".format(p, v))
        # print("(position, velocity) = ({}, {})".format(position, velocity))
        # p, v = step((p, v), action)
        pyro.sample("obs_v_{}".format(i), dist.Normal(velocity, 0.0001 * torch.ones([1]).type(torch.Tensor)), obs = v)
        pyro.sample("obs_p_{}".format(i), dist.Normal(position - data[i][0], 0.0001 * torch.ones(1).type(torch.Tensor)), obs = p - data[i][0])

def mcmc_solver(idx):
    data = []
    for (p, v, a) in data_input[idx]:
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

def variance(l):
    sum = 0.0
    for (a, b) in l:
        sum = sum + (a - b) * (a - b)
    return math.sqrt(sum/len(l))
def test(idx, a0, a1):
    plist = []
    vlist = []
    for (p, v, a) in data_input[idx]:
        orires = step((torch.tensor(p), torch.tensor(v)), torch.tensor(a))
        op, ov = step(orires, torch.tensor(a))
        ap, av = stepAppr(a0, a1, (torch.tensor(p), torch.tensor(v)), torch.tensor(a))
        # print((op, ap), (ov, av))
        plist.append((op.item(), ap.item()))
        vlist.append((ov.item(), av.item()))
    print("position variance: {}".format(variance(plist)))
    print("velocity variance: {}".format(variance(vlist)))

if sys.argv[1] == 'train':
    for idx in range(3):
        mcmc_solver(idx)
elif sys.argv[1] == 'testall':
    test(0, 0.01, 1.99)
    test(1, -0.02, 1.99)
    test(2, -2.53, 2.00)
elif sys.argv[1] == 'test':
    test(int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))


# d0: 0.01, 1.99
