import elastic_rods
import numpy as np
from numpy.linalg import norm
import math, random
from elastic_rods import EnergyType, compute_equilibrium
import pickle

# Rotate v around axis using Rodrigues' rotation formula
def rotatedVector(sinThetaAxis, cosTheta, v):
    sinThetaSq = np.dot(sinThetaAxis, sinThetaAxis)
    # More robust handling of small rotations:
    # Use identity (1 - cosTheta) / (sinTheta^2) = 1/2 sec(theta / 2)^2
    #  ~= 1 / 2 sec(sinTheta / 2)^2
    # (small angle approximation for theta)
    normalization = 0
    if (sinThetaSq > 1e-6):
        normalization = (1 - cosTheta) / sinThetaSq
    else:
        tmp = math.cos(0.5 * math.sqrt(sinThetaSq))
        normalization = 0.5 / (tmp * tmp)
    return sinThetaAxis * (np.dot(sinThetaAxis, v) * normalization) + cosTheta * v + np.cross(sinThetaAxis, v)

# Apply a random perturbation to the joint z positions to try to break symmetry.
def perturb_joints(linkage, zPerturbationEpsilon = 1e-3):
    dofs = np.array(linkage.getDoFs())
    zCoordDoFs = np.array(linkage.jointPositionDoFIndices())[2::3]
    dofs[zCoordDoFs] += 2 * zPerturbationEpsilon * (np.random.random_sample(len(zCoordDoFs)) - 0.5)
    linkage.setDoFs(dofs)

# Drive the linkage open either by opening a particular joint or by setting an
# average opening angle.
class AngleStepper:
    def __init__(self, useTargetAngleConstraint, linkage, jointIdx, fullAngle, numSteps):
        self.linkage = linkage
        self.joint = linkage.joint(jointIdx) # Ignored if useTargetAngleConstraint

        self.useTargetAngleConstraint = useTargetAngleConstraint
        if (useTargetAngleConstraint):
            self.currentAngle = self.linkage.averageJointAngle;
        else:
            self.currentAngle = self.joint.alpha

        self.thetaStep = fullAngle / numSteps

    def step(self):
        self.currentAngle += self.thetaStep
        if (not self.useTargetAngleConstraint):
            self.joint.alpha = self.currentAngle

# Drive open the linkage by opening the angle at jointIdx
def open_linkage(linkage, jointIdx, fullAngle, numSteps, view = None,
                 zPerturbationEpsilon = 0, equilibriumSolver = compute_equilibrium,
                 finalEquilibriumSolver = None, earlyStopIt = None, verbose = True,
                 maxNewtonIterationsIntermediate = 15,
                 useTargetAngleConstraint = False,
                 outPathFormat = None,
                 iterationCallback = None):
    j = linkage.joint(jointIdx)
    jdo = linkage.dofOffsetForJoint(jointIdx)
    if (useTargetAngleConstraint):
        fixedVars = list(range(jdo, jdo + 6)) # fix rigid motion by constrainting the joint orientation only
    else:
        fixedVars = list(range(jdo, jdo + 7)) # fix the orientation and angle at the driving joint

    stepper = AngleStepper(useTargetAngleConstraint, linkage, jointIdx, fullAngle, numSteps)

    def reportIterate(it):
        print("\t".join(map(str, [stepper.joint.alpha, linkage.energy()]
                               + [linkage.energy(t) for t in EnergyType.__members__.values()])))
    convergenceReports = []
    actuationForces = []
    average_angles = []

    opts = elastic_rods.NewtonOptimizerOptions()
    opts.verbose = verbose
    opts.niter = maxNewtonIterationsIntermediate

    for it in range(1, numSteps + 1):
        if ((earlyStopIt != None) and it == earlyStopIt):
            return

        if (iterationCallback is not None): iterationCallback(it - 1, linkage)
        stepper.step()
        perturb_joints(linkage, zPerturbationEpsilon)

        tgtAngle = stepper.currentAngle if useTargetAngleConstraint else elastic_rods.TARGET_ANGLE_NONE
        print("target angle: ", tgtAngle)
        r = equilibriumSolver(tgtAngle, linkage, opts, fixedVars)
        # pickle.dump(linkage, open('open_post_step_{}.pkl'.format(it), 'wb'))

        convergenceReports.append(r)
        actuationForces.append(linkage.gradient()[linkage.jointAngleDoFIndices()[jointIdx]])
        average_angles.append(linkage.averageJointAngle)

        if (view is not None):
            view.update(False)

        if (outPathFormat is not None):
            linkage.saveVisualizationGeometry(outPathFormat.format(it), averagedMaterialFrames=True)

        reportIterate(it)

    if (finalEquilibriumSolver is None):
        finalEquilibriumSolver = equilibriumSolver
    opts.niter = 1000;
    tgtAngle = stepper.currentAngle if useTargetAngleConstraint else elastic_rods.TARGET_ANGLE_NONE
    r = finalEquilibriumSolver(tgtAngle, linkage, opts, fixedVars)
    convergenceReports.append(r)

    if (iterationCallback is not None):
        iterationCallback(len(convergenceReports) - 1, linkage)

    actuationForces.append(linkage.gradient()[linkage.jointAngleDoFIndices()[jointIdx]])
    average_angles.append(linkage.averageJointAngle)
    if (view is not None):
        view.update(False)
    return convergenceReports, actuationForces, average_angles
