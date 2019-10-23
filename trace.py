import numpy as np
import torch
import math

min_position = -1.2
max_position = 0.6
max_speed = 0.07
goal_position = 0.5
goal_velocity = 0
force = 0.001
gravity = 0.0025

def step(state, action):
    position, velocity = state
    velocity = velocity + (action-1)*force + np.cos(3*position)*(-gravity)
    velocity = np.clip(velocity, -max_speed, max_speed)
    position = position + velocity
    position = np.clip(position, min_position, max_position)
    if (position==min_position and velocity<0): velocity = 0.0
    done = bool(position >= goal_position and velocity >= goal_velocity)
    reward = -1.0
    return ((position, velocity), done, reward)

def stepAppr(state, action):
    position, velocity = state
    def f0(position):
        if position < -0.3:
            return 3.0
        elif position < 0.0:
            return 0.01
        elif position < 0.3:
            return -0.02
        else:
            return 3.0
    velocity = velocity + (action-1)*force + np.cos(f0(position)*position)*(-gravity)
    velocity = np.clip(velocity, -max_speed, max_speed)
    def f1(position):
        if position < -0.3:
            return 1.0
        elif position < 0.0:
            return 1.99
        elif position < 0.3:
            return 1.99
        else:
            return 1.0
    position = position + f1(position)*velocity
    position = np.clip(position, min_position, max_position)
    if (position==min_position and velocity<0): velocity = 0.0
    done = bool(position >= goal_position and velocity >= goal_velocity)
    reward = -1.0
    return ((position, velocity), done, reward)


def agent(turnaround, state):
    position, velocity = state
    if (position < turnaround) and (velocity < 0):
        tmp = np.random.choice(3, 1, p=[0.5, 0.0, 0.5])
        return tmp[0]
    else:
        tmp = np.random.choice(3, 1, p=[0.5, 0.0, 0.5])
        return tmp[0]

class Agent:
    def __init__(self, turnaround):
        self.turnaround = turnaround
        self.turned = False

    def act(self, state):
        position, velocity = state
        if (position < self.turnaround):
            self.turned = True;
        if(self.turned):
            tmp = np.random.choice(3, 1, p=[0.25, 0.0, 0.75])
            return tmp[0]
        else:
            tmp = np.random.choice(3, 1, p=[0.75, 0.0, 0.25])
            return tmp[0]

epoch_num = 100
state = 0.0, 0.0
agent0 = Agent(-1.0)
agent0 = Agent(-0.7)
trace = []
success = False
for i in range(epoch_num):
    trace.append(state)
    action = agent0.act(state)
    # state, done, reward = step(state, action)
    state, done, reward = stepAppr(state, action)
    if done:
        success = True
        trace.append(state)
        break
print(success)
print(len(trace))
print(trace)
