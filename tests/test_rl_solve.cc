#include <iostream>
#include <fstream>
#include "../RodLinkage.hh"
#include "../restlen_solve.hh"
#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/GlobalBenchmark.hh>

int main(int argc, const char * argv[]) {
    if ((argc != 4) && (argc != 5)) {
        std::cout << "usage: " << argv[0] << " linkage.msh cross_section.json constrained_joint_idx [numprocs]" << std::endl;
        exit(-1);
    }
    const std::string &linkageGraph = argv[1];
    const size_t constrained_joint_idx = std::stoi(argv[3]);

    RodMaterial mat(*CrossSection::load(argv[2]));

    RodLinkage linkage(linkageGraph, 10);
    // RodLinkage linkage(linkageGraph, 3);
    linkage.setMaterial(mat);

    NewtonOptimizerOptions opts;
    opts.niter = 1000;
    restlen_solve(linkage, opts);

    linkage.saveVisualizationGeometry("rlo.msh");

    const size_t jo = linkage.dofOffsetForJoint(constrained_joint_idx);
    std::vector<size_t> fixedVars{jo, jo + 1, jo + 2, jo + 3, jo + 4, jo + 5, jo + 6};

    compute_equilibrium(linkage, opts, fixedVars);
    linkage.saveVisualizationGeometry("flat.msh");

    BENCHMARK_REPORT_NO_MESSAGES();

    return 0;
}
