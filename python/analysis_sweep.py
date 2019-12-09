import elastic_rods, pickle, scipy, numpy as np, time
from MeshFEM import sparse_matrices
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm
from open_linkage import open_linkage
import os, sys

if ('matplotlib' not in sys.modules):
    import matplotlib
    matplotlib.use('Agg')
from matplotlib import pyplot as plt

from io_redirection import redirect_stdout_stderr

plt.figure(num=None, figsize=(24, 18), dpi=96, facecolor='w', edgecolor='k')

elastic_rods.set_max_num_tbb_threads(6)

def run(linkage_path, base_out_path):
    name = os.path.splitext(os.path.basename(linkage_path))[0]
    out_path = base_out_path + '/' + name
    if (os.path.exists(out_path)): return # skip already run examples
    try: os.makedirs(out_path)
    except: pass

    linkage = elastic_rods.RodLinkage(linkage_path, 20)
    mat = elastic_rods.RodMaterial('+', 2000, 0.3, [2, 2, 0.2, 0.2])
    linkage.setMaterial(mat)

    linkage.saveVisualizationGeometry('{}/input.msh'.format(out_path))

    with redirect_stdout_stderr('{}/stdout.txt'.format(out_path), '{}/stderr.txt'.format(out_path)):
        elastic_rods.benchmark_reset()
        print("Restlen solve")
        elastic_rods.restlen_solve(linkage)

        driver = linkage.centralJoint()
        jdo = linkage.dofOffsetForJoint(driver)
        fixedVars = list(range(jdo, jdo + 6)) # fix global motion by fixing orientation/position of central joint

        def equilibriumSolver(tgtAngle, l, opts, fv):
            opts.beta = 1e-8
            opts.gradTol = 1e-6
            opts.useIdentityMetric = True
            if (tgtAngle != None): return elastic_rods.compute_equilibrium(l, tgtAngle, options=opts, fixedVars=fv)
            else:                  return elastic_rods.compute_equilibrium(l,           options=opts, fixedVars=fv)

        # Compute rest configuration
        print("Initial equilibrium solve")
        opts = elastic_rods.NewtonOptimizerOptions()
        opts.niter = 100
        cr = equilibriumSolver(None, linkage, opts, fixedVars)
        linkage.saveVisualizationGeometry('{}/equilibrium_0.msh'.format(out_path))
        pickle.dump(linkage, open('{}/equilibrium_0.pkl'.format(out_path), 'wb'))
        linkage.writeLinkageDebugData('{}/equilibrium_0_linkage_data.msh'.format(out_path))

        # closed_linkage = elastic_rods.RodLinkage(linkage)

        convergenceReports = [cr]
        average_angles = [linkage.averageJointAngle]
        for i in range(1,6):
            print("Opening iterations {}".format(i))
            cr, forces, aa = open_linkage(linkage, driver, np.pi/12, 25, None, zPerturbationEpsilon=0, equilibriumSolver=equilibriumSolver, verbose=1, maxNewtonIterationsIntermediate=50, useTargetAngleConstraint=True)
            # The last convergence report/average angle report is just a more accurate version of the final opening step.
            cr[-2] = cr[-1]
            aa[-2] = aa[-1]
            linkage.saveVisualizationGeometry('{}/equilibrium_{}_pi_12.msh'.format(out_path, i))
            # elastic_rods.linkage_deformation_analysis(closed_linkage, linkage, "{}/defo_analysis_{}_pi_12.msh".format(out_path, i))
            pickle.dump(linkage, open('{}/equilibrium_{}_pi_12.pkl'.format(out_path, i), 'wb'))
            linkage.writeLinkageDebugData('{}/equilibrium_{}_pi_12_linkage_data.msh'.format(out_path, i))
            convergenceReports += cr[:-1]
            average_angles     += aa[:-1]

        elastic_rods.benchmark_report()

    # Extract statistics
    energy = [cr.energy[-1] for cr in convergenceReports]
    energy_bend    = [cr.customData[-1]['energy_bend'   ] for cr in convergenceReports]
    energy_twist   = [cr.customData[-1]['energy_twist'  ] for cr in convergenceReports]
    energy_stretch = [cr.customData[-1]['energy_stretch'] for cr in convergenceReports]

    # Generate plots
    plt.cla()
    plt.title('Opening Analysis: {}'.format(name))
    plt.xlabel('Average Joint Angle')
    plt.ylabel('Energy')
    plt.plot(average_angles, energy, average_angles, energy_stretch, average_angles, energy_bend, average_angles, energy_twist)
    plt.legend(['full', 'stretch', 'bend', 'twist'])
    plt.savefig('{}/energy.png'.format(out_path))

# Analyze all grid collections
failfile = open('failures_new_2.txt', 'w')
for collection in os.listdir('../sweep/'):
    collection_dir = '../sweep/' + collection
    for example in os.listdir(collection_dir):
        try:    run(collection_dir + '/' + example, '/scratch/grid_sweep_new/' + collection)
        except:
            failfile.write("{}/{}\n".format(collection, example))
            failfile.flush()
