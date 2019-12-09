#!/usr/bin/python3
from multiprocessing import Pool, TimeoutError
import math, os, sys, subprocess
import numpy as np
etaValues = np.linspace(0.0001, 0.4, 201)
pValues = [1.0, 0.5, 0.33, 0.25]

jobs = []
for trial in range(4):
    for eta in etaValues:
        for p in pValues:
            jobs.append("run.sh {:.6f} {} {}".format(eta, p, trial))

num_chunks, chunk_idx = map(int, sys.argv[1:])
chunk_size = math.ceil(len(jobs) / num_chunks)

chunk_start = chunk_size * chunk_idx
chunk_end = min(chunk_start + chunk_size, len(jobs))

def run_job(cmd_string):
    subprocess.call(['bash'] + cmd_string.strip().split())

pool = Pool(processes = 5)
pool.map(run_job, jobs[chunk_size * chunk_idx:chunk_end])
