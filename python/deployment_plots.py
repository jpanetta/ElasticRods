from matplotlib import pyplot as plt
from typing import NamedTuple
import numpy as np

class StressRecord(NamedTuple):
    maxBendingStress: float
    maxTwistingStress: float

class StressRecorder:
    def __init__(self):
        self.actuationAngle = []
        self.stressRecords = []
    def log(self, iterate, linkage):
        self.actuationAngle.append(linkage.averageJointAngle)
        self.stressRecords.append(StressRecord(np.max([s.rod.bendingStresses() for s in linkage.segments()]),
                                               np.max([s.rod.twistingStresses() for s in linkage.segments()])))

def energy_plot(actuationAngles, convergenceReports):
    energies = [r.energy[-1] for r in convergenceReports]
    bendingEnergies = [r.customData[-1]['energy_bend'] for r in convergenceReports]
    stretchingEnergies = [r.customData[-1]['energy_stretch'] for r in convergenceReports]
    twistingEnergies = [r.customData[-1]['energy_twist'] for r in convergenceReports]
    aa = actuationAngles
    plt.plot(aa, energies, label='total energy')
    plt.plot(aa, bendingEnergies, label='bending')
    plt.plot(aa, stretchingEnergies, label='stretching')
    plt.plot(aa, twistingEnergies, label='twisting')
    plt.ylabel('Energy (N mm)')
    plt.xlabel('Actuation angle (rad)')
    plt.legend()

def stress_plot(stressRecorder):
    bendingStresses = [r.maxBendingStress for r in stressRecorder.stressRecords]
    twistingStresses = [r.maxTwistingStress for r in stressRecorder.stressRecords]

    aa = stressRecorder.actuationAngle
    plt.plot(aa, bendingStresses, label='Bending stress')
    plt.plot(aa, twistingStresses, label='Twisting stress')
    plt.ylabel('Stress (MPa)')
    plt.xlabel('Actuation angle (rad)')
    plt.legend()

def bending_stress_comparison_plot(stressRecorder1, label1, stressRecorder2, label2):
    bendingStresses1  = [r.maxBendingStress  for r in stressRecorder1.stressRecords]
    twistingStresses1 = [r.maxTwistingStress for r in stressRecorder1.stressRecords]
    bendingStresses2  = [r.maxBendingStress  for r in stressRecorder2.stressRecords]
    twistingStresses2 = [r.maxTwistingStress for r in stressRecorder2.stressRecords]

    aa1 = stressRecorder1.actuationAngle
    aa2 = stressRecorder2.actuationAngle

    plt.plot(aa1, bendingStresses1, label=label1)
    plt.plot(aa2, bendingStresses2, label=label2)

    plt.ylabel('Stress (MPa)')
    plt.xlabel('Actuation angle (rad)')
    plt.legend()

def twisting_stress_comparison_plot(stressRecorder1, label1, stressRecorder2, label2):
    twistingStresses1  = [r.maxBendingStress  for r in stressRecorder1.stressRecords]
    twistingStresses1 = [r.maxTwistingStress for r in stressRecorder1.stressRecords]
    twistingStresses2  = [r.maxBendingStress  for r in stressRecorder2.stressRecords]
    twistingStresses2 = [r.maxTwistingStress for r in stressRecorder2.stressRecords]

    aa1 = stressRecorder1.actuationAngle
    aa2 = stressRecorder2.actuationAngle

    plt.plot(aa1, twistingStresses1, label=label1)
    plt.plot(aa2, twistingStresses2, label=label2)

    plt.ylabel('Stress (MPa)')
    plt.xlabel('Actuation angle (rad)')
    plt.legend()

def energy_comparison_plot(actuationAngles1, convergenceReports1, label1, actuationAngles2, convergenceReports2, label2):
    energies1 = [r.energy[-1] for r in convergenceReports1]
    energies2 = [r.energy[-1] for r in convergenceReports2]

    aa1 = actuationAngles1
    aa2 = actuationAngles2
    plt.plot(aa1, energies1, label=label1)
    plt.plot(aa2, energies2, label=label2)
    plt.ylabel('Energy (N mm)')
    plt.xlabel('Actuation angle (rad)')
    plt.legend()
