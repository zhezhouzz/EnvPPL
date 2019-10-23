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

gravity = torch.tensor(9.8).type(torch.Tensor)
masscart = torch.tensor(1.0).type(torch.Tensor)
masspole = torch.tensor(0.1).type(torch.Tensor)
total_mass = torch.tensor(masspole + masscart).type(torch.Tensor)
length = torch.tensor(0.5).type(torch.Tensor) # actually half the pole's length
polemass_length = torch.tensor(masspole * length).type(torch.Tensor)
force_mag = torch.tensor(10.0).type(torch.Tensor)
tau = torch.tensor(0.02).type(torch.Tensor)

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
    return torch.tensor(intermid).type(torch.Tensor)

d0 = space_maker([("continous", (-1, 1.0, 0.5)), ("continous", (-1.0, 1.0, 0.5)),
                  ("continous", (-6*math.pi/360, 6*math.pi/360, 0.05)),
                  ("continous", (-6*math.pi/360, 6*math.pi/360, 0.05)),
                  ("discrate", (0, 2, 1))])
d1 = space_maker([("continous", (4.1, 4.5, 0.2)), ("continous", (-1.0, 1.0, 0.5)),
                  ("continous", (-6*math.pi/360, 6*math.pi/360, 0.05)),
                  ("continous", (-6*math.pi/360, 6*math.pi/360, 0.05)),
                  ("discrate", (0, 2, 1))])
print(len(d0))
data_input = [d1]

def step(data):
    x = data[0]
    x_dot = data[1]
    theta = data[2]
    theta_dot = data[3]
    action = data[4]
    force = (2.0 * action - 1.0)*force_mag
    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)
    temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta* temp) / (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
    xacc  = temp - polemass_length * thetaacc * costheta / total_mass
    x  = x + tau * x_dot
    x_dot = x_dot + tau * xacc
    theta = theta + tau * theta_dot
    theta_dot = theta_dot + tau * thetaacc
    # print([x, x_dot, theta, theta_dot])
    # print(torch.stack([x, x_dot, theta, theta_dot]))
    return torch.stack([x, x_dot, theta, theta_dot])

def stepAppr(data, a):
    a0 = torch.tensor(-1.546).type(torch.Tensor)
    a1 = torch.tensor(1.37).type(torch.Tensor)
    a2 = torch.tensor(0.0135).type(torch.Tensor)
    x = data[0]
    x_dot = data[1]
    theta = data[2]
    theta_dot = data[3]
    action = data[4]
    force = (2.0 * action - 1.0)*force_mag
    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)
    temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta* temp) / (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
    xacc  = temp - polemass_length * thetaacc * costheta / total_mass
    x  = a0 + a1 * x + a2 * x_dot
    x_dot = x_dot + tau * xacc
    theta = theta + tau * theta_dot
    theta_dot = theta_dot + tau * thetaacc
    return torch.stack([x, x_dot, theta, theta_dot])

def model(data):
    a0 = torch.squeeze(pyro.sample('a0', dist.Normal(torch.zeros(1), 10 * torch.ones(1))) * 0.1)
    a1 = torch.squeeze(pyro.sample('a1', dist.Normal(torch.zeros(1) + 1.0, 10 * torch.ones(1))))
    a2 = torch.squeeze(pyro.sample('a2', dist.Normal(torch.zeros(1) + 2.0, 10 * torch.ones(1))) * 0.01)
    # a2 = pyro.sample('a2', dist.Normal(torch.zeros(1), 10 * torch.ones(1)))
    # a3 = pyro.sample('a3', dist.Normal(torch.zeros(1), 10 * torch.ones(1)))
    for i in range(len(data)):
        x = data[i][0]
        x_dot = data[i][1]
        theta = data[i][2]
        theta_dot = data[i][3]
        action = data[i][4]
        # print("data: {}".format(data[i]))
        if x > 4.4:
            ori = step(data[i])
            ori[0] = ori[0] + 0.2
        else:
            ori = step(data[i])
        force = (2.0 * action - 1.0)*force_mag
        # costheta = torch.squeeze(theta * a1 + a0)
        # sintheta = torch.squeeze(a2 + a3 * theta)
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
        thetaacc = (gravity * sintheta - costheta* temp) / (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
        xacc  = temp - polemass_length * thetaacc * costheta / total_mass
        x  = a0 + a1 * x + a2 * x_dot
        x_dot = x_dot + tau * xacc
        theta = theta + tau * theta_dot
        theta_dot = theta_dot + tau * thetaacc
        # print([x, x_dot, theta, theta_dot])
        appr = torch.stack([x, x_dot, theta, theta_dot])
        pyro.sample("obs_0_{}".format(i), dist.Normal(appr[0], 0.01 * torch.ones([1]).type(torch.Tensor)), obs = ori[0])
        pyro.sample("obs_1_{}".format(i), dist.Normal(appr[1], 0.01 * torch.ones([1]).type(torch.Tensor)), obs = ori[1])
        pyro.sample("obs_2_{}".format(i), dist.Normal(appr[2], 0.01 * torch.ones([1]).type(torch.Tensor)), obs = ori[2])
        pyro.sample("obs_3_{}".format(i), dist.Normal(appr[3], 0.01 * torch.ones([1]).type(torch.Tensor)), obs = ori[3])

def mcmc_solver(idx):
    nuts_kernel = NUTS(model, jit_compile=False,)
    mcmc = MCMC(nuts_kernel,
                    num_samples=100,
                    warmup_steps=100,
                    num_chains=1)
    data = torch.tensor(data_input[idx])
    mcmc.run(data)
    mcmc.summary(prob=0.8)

def variance(l):
    sum = 0.0
    for (a, b) in l:
        sum = sum + (a - b) * (a - b)
    return math.sqrt(sum/len(l))
def test(idx, a0):
    reslist = [[], [], [], []]
    for d in data_input[idx]:
        orires = step(d)
        if d[0] > 4.4:
            orires[0] = orires[0] + 0.2
        # op, ov = step(orires, torch.tensor(a))
        appr = stepAppr(d, a0)
        # print((op, ap), (ov, av))
        va = [ (o.item(), a.item())for (o, a) in zip(orires, appr)]
        for i in range(len(reslist)):
            reslist[i].append(va[i])
    varlist = [variance(l) for l in reslist]
    namelist = ["x", "x'", "θ", "θ'"]
    for (name, v) in zip(namelist, varlist):
        print("{} variance: {}".format(name, v))

if sys.argv[1] == 'train':
    for idx in range(len(data_input)):
        mcmc_solver(idx)
elif sys.argv[1] == 'testall':
    test(0, 0.01, 1.99)
    test(1, -0.02, 1.99)
    test(2, -2.53, 2.00)
elif sys.argv[1] == 'test':
    test(int(sys.argv[2]), [0.02])


# d0: 0.01, 1.99
