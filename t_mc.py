min_position = -1.2
max_position = 0.6
max_speed = 0.07
goal_position = 0.5
goal_velocity = 0

import torch
force=torch.tensor(0.001)
gravity=torch.tensor(0.0025)

def step(state, action, a0, a1):
    position, velocity = state
    velocity = velocity + (action + a1)*force + torch.cos(a0*position)*(-gravity)
    velocity = torch.clamp(velocity, -max_speed, max_speed)
    position = position + velocity
    position = torch.clamp(position, min_position, max_position)
    if (position==min_position and velocity<0): velocity = torch.tensor(0.0)
    return (position, velocity)


def test(num):
    state = torch.tensor(0.0).type(torch.Tensor), torch.tensor(0.0).type(torch.Tensor)
    for i in range(num):
        print(state)
        next = step(state, torch.tensor(2.0), 3.0, -1.0)
        next2 = step(state, torch.tensor(2.0), 0.73, -1.28)
        print("{} vs. {}".format(next, next2))
        state = next

test(10)
