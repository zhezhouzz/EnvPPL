import matplotlib.pyplot as plt

trace = [(0.0, 0.0), (-0.006965, -0.0035), (-0.016914999987932833, -0.004999999993936097), (-0.03382999990469401, -0.008499999958171444), (-0.057709999536768544, -0.011999999815112834)]

def trans(trace):
    ps = [p for p,v in trace]
    vs = [v for p,v in trace]
    return ps, vs

plt.polt(xs, ys)
plt.show()
