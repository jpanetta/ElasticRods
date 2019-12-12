from elastic_rods import EnergyType
from MeshFEM import sparse_matrices
import numpy as np
from numpy.linalg import norm
from scipy.sparse import csc_matrix

def getVars(l, variableRestLen=False):
    if (variableRestLen): return l.getExtendedDoFs()
    return l.getDoFs()

def setVars(l, dof, variableRestLen=False):
    if (variableRestLen): return l.setExtendedDoFs(dof)
    return l.setDoFs(dof)

def energyAt(l, dof, etype = EnergyType.Full, variableRestLen=False):
    prevDoF = getVars(l, variableRestLen)
    setVars(l, dof, variableRestLen)
    energy = l.energy(etype)
    setVars(l, prevDoF, variableRestLen)
    return energy

# Pybind11 methods/funcs apparently don't support `inspect.signature`,
# but at least their arg names are guaranteed to appear in the docstring... :(
def hasArg(func, argName):
    return argName in func.__doc__

def guardedEval(func, *args, **kwargs):
    '''
    Evaluate `func`, on the passed arguments, filtering out any unrecognized keyword arguments.
    '''
    return func(*args, **{k: v for k, v in kwargs.items() if hasArg(func, k)})

def gradientAt(l, dof, etype = EnergyType.Full, variableRestLen=False, updatedSource=False):
    prevDoF = getVars(l, variableRestLen)
    setVars(l, dof, variableRestLen)
    g = guardedEval(l.gradient, updatedSource=updatedSource, energyType=etype, variableRestLen=variableRestLen)
    setVars(l, prevDoF, variableRestLen)
    return g

def fd_gradient_test(obj, stepSize, etype=EnergyType.Full, direction=None, variableRestLen=False):
    grad = guardedEval(obj.gradient, updatedSource=False, energyType=etype)
    if (direction is None): direction = grad.copy()
    step = stepSize * direction
    x = obj.getDoFs()
    return [(energyAt(obj, x + step, etype, variableRestLen) - energyAt(obj, x - step, etype, variableRestLen)) / (2 * stepSize), np.dot(direction, grad)]

def gradient_convergence(linkage, minStepSize=1e-8, maxStepSize=1e-2, etype=EnergyType.Full, direction=None):
    if (direction is None): direction = np.random.uniform(-1, 1, linkage.numDoF())
    
    epsilons = np.logspace(np.log10(minStepSize), np.log10(maxStepSize), 100)
    errors = []

    for eps in epsilons:
        fd, an = fd_gradient_test(linkage, eps, etype=etype, direction=direction)
        err = np.abs(an - fd) / np.abs(an)
        errors.append(err)
    return (epsilons, errors, an)

def gradient_convergence_plot(linkage, minStepSize=1e-8, maxStepSize=1e-2, etype=EnergyType.Full, direction=None):
    from matplotlib import pyplot as plt
    eps, errors, ignore = gradient_convergence(linkage, minStepSize, maxStepSize, etype, direction)
    plt.title('Directional derivative fd test for gradient')
    plt.ylabel('Relative error')
    plt.xlabel('Step size')
    plt.loglog(eps, errors)
    plt.grid()

def fd_hessian_test(linkage, stepSize, etype=EnergyType.Full, direction=None, variableRestLen=False, infinitesimalTransportGradient=False):
    h = guardedEval(linkage.hessian, energyType=etype, variableRestLen=variableRestLen)
    h.reflectUpperTriangle()
    if (direction is None): direction = np.array(guardedEval(linkage.gradient, updatedSource=True, energyType=etype, variableRestLen=variableRestLen))

    H = csc_matrix(h.compressedColumn())
    dof = getVars(linkage, variableRestLen)
    return [(gradientAt(linkage, dof + stepSize * direction, etype, variableRestLen=variableRestLen, updatedSource=infinitesimalTransportGradient)
           - gradientAt(linkage, dof - stepSize * direction, etype, variableRestLen=variableRestLen, updatedSource=infinitesimalTransportGradient)) / (2 * stepSize),
            H * direction]

def fd_hessian_test_relerror_max(linkage, stepSize, etype=EnergyType.Full, direction=None):
    dgrad = fd_hessian_test(linkage, stepSize, etype, direction)
    relErrors = np.abs((dgrad[0] - dgrad[1]) / dgrad[0])
    idx = np.argmax(relErrors)
    return (idx, relErrors[idx], dgrad[0][idx], dgrad[1][idx])

def fd_hessian_test_relerror_norm(linkage, stepSize, etype=EnergyType.Full, direction=None, infinitesimalTransportGradient=False):
    dgrad = fd_hessian_test(linkage, stepSize, etype, direction, infinitesimalTransportGradient=infinitesimalTransportGradient)
    return norm(dgrad[0] - dgrad[1]) / norm(dgrad[0])

def hessian_convergence(linkage, minStepSize=1e-8, maxStepSize=1e-2, etype=EnergyType.Full, direction=None, infinitesimalTransportGradient=False):
    if (direction is None): direction = np.random.uniform(-1, 1, linkage.numDoF())
    
    epsilons = np.logspace(np.log10(minStepSize), np.log10(maxStepSize), 100)
    errors = [fd_hessian_test_relerror_norm(linkage, eps, etype, direction, infinitesimalTransportGradient) for eps in epsilons]
    return (epsilons, errors)

def hessian_convergence_plot(linkage, minStepSize=1e-8, maxStepSize=1e-2, etype=EnergyType.Full, direction=None, infinitesimalTransportGradient=False):
    from matplotlib import pyplot as plt
    eps, errors = hessian_convergence(linkage, minStepSize, maxStepSize, etype, direction, infinitesimalTransportGradient)
    plt.title('Directional derivative fd test for hessian')
    plt.ylabel('Relative error')
    plt.xlabel('Step size')
    plt.loglog(eps, errors)
    plt.grid()
