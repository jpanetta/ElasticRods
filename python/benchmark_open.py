import elastic_rods, sparse_matrices, pickle, scipy, linkage_vis, numpy as np, time
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm
from open_linkage import open_linkage

elastic_rods.set_max_num_tbb_threads(6)

# driver=48
# linkage = elastic_rods.RodLinkage('../examples/florin/20181114_134207_meshID_34b18bdf-8314-454a-96de-b2b2d7356db1.obj', 10)
driver=64
linkage = elastic_rods.RodLinkage('../examples/nonuniform_linkage.obj', 10)
mat = elastic_rods.RodMaterial('+', 2000, 0.3, [0.02, 0.02, 0.002, 0.002])
# mat = elastic_rods.RodMaterial('ellipse', 20000, 0.3, [0.02, 0.002])
linkage.setMaterial(mat)

elastic_rods.benchmark_reset()
elastic_rods.restlen_solve(linkage)

jdo = linkage.dofOffsetForJoint(driver)
fixedVars = list(range(jdo, jdo + 6)) # fix rigid motion for a single joint
fixedVars.append(jdo + 6) # constrain angle at the driving joint

elastic_rods.compute_equilibrium(linkage, fixedVars=fixedVars)

def equilibriumSolver(tgtAngle, l, opts, fv):
    opts.useIdentityMetric = False
    opts.beta = 1e-8
    return elastic_rods.compute_equilibrium(l, tgtAngle, options=opts, fixedVars=fv)

#cr = open_linkage(linkage, driver, np.pi/4, 25, None, zPerturbationEpsilon=0, equilibriumSolver=equilibriumSolver, verbose=1, useTargetAngleConstraint=False)
cr = open_linkage(linkage, driver, np.pi/8, 25, None, zPerturbationEpsilon=0, equilibriumSolver=equilibriumSolver, verbose=1, useTargetAngleConstraint=True)
# cr = open_linkage(linkage, driver, np.pi/4, 10, None, zPerturbationEpsilon=1e-5, equilibriumSolver=elastic_rods.compute_equilibrium_knitro)

elastic_rods.benchmark_report()

linkage.saveVisualizationGeometry('opened.msh')

# import convergence_reporting
# convergence_reporting.gen_plots(cr, 'nu_open_convergence_{}.png')
