import elastic_rods, numpy as np
from MeshFEM import sparse_matrices
import  linkage_vis
import time
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from PlanarElastica import AnalyticRod
from io_redirection import suppress_stdout

roundOdd = lambda x: int(np.floor(x / 2) * 2 + 1)
def initialConfiguration(L, a, n, perturb=False):
    # slope = (1 - (a/L)**2)**(1/2) / (a / L)
    # Initialize the rod in a triangular configuration
    #  / \
    # /   \
    #0-----a
    slope = ((L / a)**2 - 1)**(0.5)
    height = slope * a / 2
    midpt = int(np.floor(n / 2))
    leftPts  = np.stack((np.linspace(-a / 2, 0, midpt + 1), np.linspace(0, height, midpt + 1), np.zeros((midpt + 1,)))).transpose()
    rightPts = np.stack((np.linspace( 0, a / 2, midpt + 1), np.linspace(height, 0, midpt + 1), np.zeros((midpt + 1,)))).transpose()
    # thetas = [t + 0.05 for t in r.thetas()] # perturb rotations out of the stationary configuration
    thetas = np.zeros(n - 1)
    if perturb:
        leftPts += 1e-2 * np.random.random_sample(leftPts.shape)
    return np.vstack((leftPts, rightPts[1:])), thetas

def restRod(L, n):
    return elastic_rods.ElasticRod([[x, 0, 0] for x in np.linspace(0, L, n)])

def bendingTestRod(L, a, n, perturb=False):
    n = roundOdd(n)
    r = restRod(L, n)
    r.bendingEnergyType = elastic_rods.BendingEnergyType.Bergou2008

    mat = elastic_rods.RodMaterial()
    mat.setEllipse(200, 0.3, 0.01, 0.005)
    r.setMaterial(mat)
    
    midpt = int(np.floor(n / 2))
    pts, thetas = initialConfiguration(L, a, n, perturb)
    r.setDeformedConfiguration(pts, thetas)
    fixedVars = [0, 1, 2, 3 * midpt + 2, 3 * (n - 1), 3 * (n - 1) + 1, 3 * (n - 1) + 2]
    return r, fixedVars

# Extract the height of the bent rod from the equilibrium configuration
# as well as the force needed to hold the rod ends.
def DERElasticaHeight(r): return r.deformedPoints()[int(r.numVertices() / 2)][1];
def DERElasticaForce(r): return r.gradient()[0]

def runTest(L, a, numVertices, gradTol = None):
    r, fixedVars = bendingTestRod(L, a, numVertices)
    t = time.time()
    opts = elastic_rods.NewtonOptimizerOptions()
    opts.verbose = False
    opts.niter = 1000
    opts.useIdentityMetric = False
    opts.useNegativeCurvatureDirection = True
    if (gradTol is not None):
        opts.gradTol = gradTol
    cr = elastic_rods.compute_equilibrium(r, options=opts, fixedVars=fixedVars)
    # if (gradTol is None): elastic_rods.compute_equilibrium_knitro(r, verbose=False, fixedVars=fixedVars, niter=1000)
    # else:                 elastic_rods.compute_equilibrium_knitro(r, verbose=False, fixedVars=fixedVars, niter=1000)
    simTime = time.time() - t
    
    arod = AnalyticRod(L, a)
    return r, arod, simTime, cr.numIters()

def runTestVisualization(L, a, numVertices, gradTol = None):
    with suppress_stdout():
        r, arod, simTime, niters = runTest(L, a, numVertices, gradTol)
    ax,ay = arod.pts(np.linspace(-np.pi/2, np.pi/2, 100))
    x, y  = np.column_stack(r.deformedPoints())[0:2, :]
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.plot(x, y, linewidth=8.0, color='orange')
    plt.plot(ax, ay, color='black')
    axes = plt.gca()
    axes.set_xlim([-L/2, L/2]);
    axes.set_ylim([0, L/2]);
    axes.set_aspect('equal', 'box');
    plt.show()
    
    relError = lambda x, x0: np.abs((x - x0) / x0)
    print("sim time: ", simTime)
    print("Height error: ",relError(DERElasticaHeight(r), arod.height()))
    print("Force error: ", relError(DERElasticaForce(r), arod.force(min(r.material(0).bendingStiffness.lambda_1, r.material(0).bendingStiffness.lambda_2))))

def simulationErrors(L, a, numVertices, gradTol = None, absError = True):
    r, arod, simTime, niters = runTest(L, a, numVertices, gradTol)
    if (absError): relError = lambda x, x0: np.abs((x - x0) / x0)
    else:          relError = lambda x, x0: (x - x0) / x0
    return ((relError(DERElasticaHeight(r), arod.height()),
            relError(DERElasticaForce(r), arod.force(min(r.material(0).bendingStiffness.lambda_1, r.material(0).bendingStiffness.lambda_2)))), niters)

def convergenceTest(a_div_L, N=150, gradTol = None):
    nv = range(3, N, 2)
    totalIters = 0
    errors = []

    with suppress_stdout():
        for i in nv:
            relErrors, niter = simulationErrors(1, a_div_L, i, gradTol, absError=False)
            totalIters += niter
            errors.append(relErrors)
    signedHeightErrors = [e[0] for e in errors]
    signedForceErrors = [e[1] for e in errors]
    heightErrors = np.abs(signedHeightErrors)
    forceErrors  = np.abs(signedForceErrors)
    from scipy.stats import linregress
    slope, intercept = linregress(-np.log(nv[1:]), np.log(forceErrors[1:]))[:2]
    print("Force  error = {:0.2f} h^{:0.2f}".format(np.exp(intercept), slope))
    slope, intercept = linregress(-np.log(nv[10:]), np.log(heightErrors[10:]))[:2]
    print("Height error = {:0.2f} h^{:0.2f}".format(np.exp(intercept), slope))

    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(nv,  forceErrors, c=np.sign( signedForceErrors), edgecolors='none', marker='.', cmap='rainbow', label= 'force')
    plt.scatter(nv, heightErrors, c=np.sign(signedHeightErrors), edgecolors='none', marker='+', cmap='rainbow', label='height')
    plt.legend()

    plt.title('Error Convergence for a/L = {}'.format(a_div_L))
    plt.xlabel('Num Rod Vertices')
    plt.ylabel('Relative Error')

    plt.show()

    return totalIters
