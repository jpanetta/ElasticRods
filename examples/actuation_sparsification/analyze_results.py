from glob import glob
import re
from collections import defaultdict

numTrials = 7
dataForP = [defaultdict(list) for t in range(numTrials)]

for path in glob('results/stdout_*_L*_*.txt'):
    eta, p, trial = path.split('_')[1:]
    p = p[1:] # trim L
    trial = int(trial[0:-4]) # trim file extension

    actuatedJoints = []
    distances = []
    for line in open(path, 'r'):
        m = re.match('Actuated joints: ([0-9]+)/', line)
        if m: actuatedJoints.append(int(m.group(1)))
        m = re.match('Distance: (.*)', line)
        if m: distances.append(float(m.group(1)))
    best = sorted(zip(actuatedJoints, distances))[0] # sparsest solution found (with lowest distance to break ties)
    best = list(zip(actuatedJoints, distances))[-1]
    if (best[0] > 35): continue
    if (best[0] < 2): continue
    if (best[1] > 6): continue
    dataForP[trial][p].append((eta, best[0], best[1]))

#for t in range(numTrials):
#    for key in dataForP{t]:
#        dataForP[t][key].sort()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 8})

for t in range(numTrials):
    fig = plt.figure(num=None, figsize=(4, 3), dpi=96)
    ax1 = fig.add_subplot(111)
    ax1.set_xlim(0, 30)

    #ax1.set_yscale('log')
    ax1.scatter([d[1] for d in dataForP[t]['1.0']], [d[2] for d in dataForP[t]['1.0']], label='p = 1', s=8)
    # ax1.scatter([d[1] for d in dataForP[t]['0.25']], [d[2] for d in dataForP[t]['0.25']], label='p = 1/4')
    # ax1.scatter([d[1] for d in dataForP[t]['0.33']], [d[2] for d in dataForP[t]['0.33']], label='p = 1/3')
    ax1.scatter([d[1] for d in dataForP[t]['0.5']], [d[2] for d in dataForP[t]['0.5']], label='p = 1/2', s=8)
    plt.legend(loc='upper right');
    plt.xlabel('# Actuations')
    plt.ylabel('Distance Objective')
    plt.savefig('dist_tradeoff_%i.pdf' % t, bbox_inches='tight', pad_inches=0)
