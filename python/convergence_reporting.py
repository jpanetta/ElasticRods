import elastic_rods, numpy as np, sys

if ('matplotlib' not in sys.modules):
    import matplotlib
    matplotlib.use('Agg')

from matplotlib import pyplot as plt

def plot_energy(report):
    iterations = np.arange(report.numIters() + 1)
    plt.title('Optimization Convergence')
    plt.xlabel('Newton Iterations')
    plt.ylabel('Energy error')
    plt.yscale('log')
    plt.scatter(iterations[:-1], np.array(report.energy[:-1]) - report.energy[-1], marker='+', edgecolor='none', c=report.indefinite[:-1], cmap='rainbow')

def plot_gradnorm(report):
    iterations = np.arange(report.numIters() + 1)
    plt.title('Optimization Convergence')
    plt.xlabel('Newton Iterations')
    plt.yscale('log')
    plt.ylabel('Gradient norm')
    plt.scatter(iterations, report.gradientNorm, marker='+', edgecolor='none', c=report.indefinite[:], cmap='rainbow')

def plot_steplength(report):
    iterations = np.arange(report.numIters() + 1)
    plt.title('Optimization Convergence')
    plt.xlabel('Newton Iterations')
    plt.yscale('log')
    plt.ylabel('Step length')
    plt.scatter(iterations, report.stepLength, marker='+', edgecolor='none', c=report.indefinite[:], cmap='rainbow')

def gen_plots(cr, outPathFormat):
    for report_num, report in enumerate(cr):
        if (report.numIters() == 0): continue

        plt.figure(num=None, figsize=(12, 9), dpi=72, facecolor='w', edgecolor='k')

        plot_energy(report)
        plt.savefig(outPathFormat.format(str(report_num) + "_energy"))
        plt.cla()

        plot_gradnorm(report)
        plt.savefig(outPathFormat.format(str(report_num) + "_gradNorm"))
        plt.cla()

        plot_steplength(report)
        plt.savefig(outPathFormat.format(str(report_num) + "_stepLength"))
