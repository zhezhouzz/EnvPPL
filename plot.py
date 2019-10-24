import matplotlib.pyplot as plt
import trace
import argparse

def split(trace):
    ps = [p for p,v in trace]
    vs = [v for p,v in trace]
    return ps, vs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("epoch_num", type = int)
    parser.add_argument("epoch_num_appr", type = int)
    parser.add_argument("turnaround", type = float)
    args = parser.parse_args()
    success, t = trace.run(epoch_num = args.epoch_num, turnaround = args.turnaround, appr = False)
    success, t_appr = trace.run(epoch_num = args.epoch_num_appr, turnaround = args.turnaround, appr = True)
    # print(success)
    ps1, vs1 = split(t)
    ps2, vs2 = split(t_appr)
    plt.plot(ps1, vs1, '.b--', ps2, vs2, '.r--')
    plt.xlabel('position')
    plt.ylabel('velocity')
    plt.show()
