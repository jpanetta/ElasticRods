import scipy, scipy.sparse, numpy as np
from scipy.sparse import csc_matrix
import elastic_rods, mode_viewer
from matplotlib import pyplot as plt
from open_linkage import open_linkage
from bending_validation import suppress_stdout

def deploymentPathAnalysis(linkage, fixedVars = None):
    if fixedVars is None:
        fixedVars = np.arange(6) + linkage.dofOffsetForJoint(linkage.centralJoint())
    return elastic_rods.DeploymentPathAnalysis(linkage, fixedVars)

def stiffnessGapThroughoutDeployment(linkage, targetAngleRad, steps, fixedVars = None):
    driver = linkage.centralJoint()
    if fixedVars is None:
        fixedVars = np.arange(6) + linkage.dofOffsetForJoint(driver)

    def equilibriumSolver(tgtAngle, l, opts, fv):
        opts.beta = 1e-8
        opts.gradTol = 1e-4
        opts.useIdentityMetric = False
        return elastic_rods.compute_equilibrium(l, tgtAngle, options=opts, fixedVars=fixedVars)

    openingAngles = []
    stiffnessGaps = []

    def iterationCB(it, l):
        openingAngles.append(l.averageJointAngle)
        stiffnessGaps.append(deploymentPathAnalysis(l, fixedVars).relativeStiffnessGap)
        if stiffnessGaps[-1] < 0: raise Exception('Negative stiffness gap')

    with suppress_stdout(): open_linkage(linkage, driver,
                                         targetAngleRad - linkage.averageJointAngle, steps, None,
                                         equilibriumSolver=equilibriumSolver,
                                         maxNewtonIterationsIntermediate=20, useTargetAngleConstraint=True, iterationCallback=iterationCB);
    return (openingAngles, stiffnessGaps)

def deploymentModeViewer(linkage, fixedVars = None):
    dpa = deploymentPathAnalysis(linkage, fixedVars)
    modes = np.column_stack([dpa.deploymentStep, dpa.secondBestDeploymentStep])
    energyIncrements = np.array([dpa.bestEnergyIncrement(1.0), dpa.secondBestEnergyIncrement(1.0)])
    return mode_viewer.ModeViewer(linkage, modes, energyIncrements, amplitude=1.0, normalize = False)

################################################################################
# Low energy deployment modes -- brute-force Python implementation (validation)
################################################################################
# Get the vector whose inner product with the DoF vector computes the average joint angle.
def averageJointAngleFunctional(linkage):
    a = np.zeros(linkage.numDoF())
    nj = linkage.numJoints()
    for j in range(nj):
        a[linkage.dofOffsetForJoint(j) + 6] = 1.0 / nj
    return a

def rigidMotionConstraint(linkage):
    R = np.zeros((linkage.numDoF(), 6))
    driver = linkage.centralJoint()
    for i in range(6):
        R[linkage.dofOffsetForJoint(driver) + i, i] = 1.0
    return R

# Brute-force KKT solve-based search for the lowest energy deployment deformations
# for a linear(ized) deployment measure a^T x
def lowestEnergyDeploymentModes(linkage):
    H = linkage.hessian()
    H.reflectUpperTriangle()
    H = csc_matrix(H.compressedColumn())
    a = averageJointAngleFunctional(linkage)
    R = rigidMotionConstraint(linkage)
    a_mat = a[:, np.newaxis]
    C = np.hstack([a_mat, R])
    numConstraints = C.shape[1]
    K = scipy.sparse.vstack([scipy.sparse.hstack([H, C]),
                                 np.hstack([C.transpose(), np.zeros((numConstraints, numConstraints))])
                                                        ])
    Crhs = np.zeros(numConstraints)
    Crhs[0] = 1.0
    d1 = scipy.sparse.linalg.spsolve(K.tocsc(), np.concatenate([-linkage.gradient(), Crhs]))[:-numConstraints]

    M = linkage.massMatrix(True, True)
    M.reflectUpperTriangle()
    M = csc_matrix(M.compressedColumn())
    Md1 = M @ d1

    C = np.hstack([C, Md1[:, np.newaxis]])
    numConstraints = C.shape[1]
    Crhs = np.zeros(numConstraints)
    Crhs[0] = 1.0
    K = scipy.sparse.vstack([scipy.sparse.hstack([H, C]),
                                 np.hstack([C.transpose(), np.zeros((numConstraints, numConstraints))])
                                                        ])
    d2 = scipy.sparse.linalg.spsolve(K.tocsc(), np.concatenate([-linkage.gradient(), Crhs]))[:-numConstraints]

    return [d1, d2]

################################################################################
# Additional validation
################################################################################
def validateEnergyIncrements(linkage, epsMin = 1e-6, epsMax = 1e-3, nsamples=20):
    def energyAt(dofs):
        currDoFs = linkage.getDoFs()
        linkage.setDoFs(dofs)
        result = linkage.energy()
        linkage.setDoFs(currDoFs)
        return result

    def gradientAt(dofs):
        currDoFs = linkage.getDoFs()
        linkage.setDoFs(dofs)
        result = linkage.gradient()
        linkage.setDoFs(currDoFs)
        return result

    eps = np.linspace(1e-6, 1e-3, nsamples)
    dpa = deploymentPathAnalysis(linkage)
    linearTerm = dpa.bestEnergyIncrement.linearTerm
    x = linkage.getDoFs()
    E0 = linkage.energy()

    energy_an_bei  = np.array([dpa.      bestEnergyIncrement(e) for e in eps]) - linearTerm * eps
    energy_an_sbei = np.array([dpa.secondBestEnergyIncrement(e) for e in eps]) - linearTerm * eps
    energy_bei  = np.array([energyAt(x + e * dpa.          deploymentStep) for e in eps]) - E0 - linearTerm * eps
    energy_sbei = np.array([energyAt(x + e * dpa.secondBestDeploymentStep) for e in eps]) - E0 - linearTerm * eps

    force_an_bei  = 2 * dpa.      bestEnergyIncrement.quadraticTerm * eps + linearTerm
    force_an_sbei = 2 * dpa.secondBestEnergyIncrement.quadraticTerm * eps + linearTerm

    force_bei  = np.array([gradientAt(x + e * dpa.          deploymentStep).dot(dpa.          deploymentStep) for e in eps])
    force_sbei = np.array([gradientAt(x + e * dpa.secondBestDeploymentStep).dot(dpa.secondBestDeploymentStep) for e in eps])

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(eps,  energy_an_bei, label='analytic best')
    plt.plot(eps, energy_an_sbei, label='analytic 2nd best')
    plt.plot(eps,     energy_bei, label='best')
    plt.plot(eps,    energy_sbei, label='2nd best')
    plt.legend()
    plt.xlabel('step scale')
    plt.title('Quadratic Energy Term')

    plt.subplot(1, 2, 2)
    plt.plot(eps,  force_an_bei, label='analytic best')
    plt.plot(eps, force_an_sbei, label='analytic 2nd best')
    plt.plot(eps,     force_bei, label='best')
    plt.plot(eps,    force_sbei, label='2nd best')
    plt.legend()
    plt.title('Force')
    plt.xlabel('step scale')
    plt.show()
