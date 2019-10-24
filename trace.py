import numpy as np
import math

min_position = -1.2
max_position = 0.6
max_speed = 0.07
goal_position = 0.5
goal_velocity = 0
force = 0.001
gravity = 0.0025

def piecesMaker(l, default):
    def aux(position):
        for (bound, piece) in l:
            if position < bound:
                return piece(position)
        return default(position)
    return aux

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
    f0 = piecesMaker([(-0.45, lambda _:1.0), (-0.3, lambda _:0.74), (-0.15, lambda _:1.2), (0.0, lambda _:1.52)], lambda _:1.0)
    f1 = piecesMaker([(-0.45, lambda _:1.0), (0.0, lambda _:1.6)], lambda _: 1.0)
    velocity = velocity + (action-1)*force*f1(position) + f0(position)*np.cos(3.0 * position)*(-gravity)
    velocity = np.clip(velocity, -max_speed, max_speed)
    f2 = piecesMaker([(-0.45, lambda _:1.0), (0.0, lambda _: 2.0)], lambda _: 1.0)
    position = position + f2(position)*velocity
    position = np.clip(position, min_position, max_position)
    if (position==min_position and velocity<0): velocity = 0.0
    done = bool(position >= goal_position and velocity >= goal_velocity)
    reward = -1.0
    return ((position, velocity), done, reward)

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
            tmp = np.random.choice(3, 1, p=[1.0, 0.0, 0.0])
            return tmp[0]

def run(epoch_num = 20, state = (0.0, 0.0), turnaround = -0.7, appr = False):
    agent0 = Agent(turnaround)
    trace = []
    success = False
    for i in range(epoch_num):
        trace.append(state)
        action = agent0.act(state)
        if not appr:
            state, done, reward = step(state, action)
        else:
            state, done, reward = stepAppr(state, action)
        if done:
            success = True
            trace.append(state)
            break
    return(success, trace)

print(run())
