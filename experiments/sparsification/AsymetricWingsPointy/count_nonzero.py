import numpy as np
import sys

path = 'torques.txt'
if (len(sys.argv) == 2):
    path = sys.argv[1];

torques = np.loadtxt(path)
nonzero = sorted(torques[np.where(torques > 0.0001)])
print(nonzero)
print(len(nonzero))

