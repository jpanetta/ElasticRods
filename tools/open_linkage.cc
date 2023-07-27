#include <iostream>
#include <fstream>
#include "../RodLinkage.hh"
#include "../restlen_solve.hh"
#include "../compute_equilibrium.hh"
#include "../open_linkage.hh"

int main(int argc, const char *argv[]) {
    if (argc != 7) {
        std::cout << "Usage: open_linkage flat_linkage.obj cross_section.json initial_deployment_angle beta gradtol step_size" << std::endl;
        exit(-1);
    }

    const std::string linkage_path(argv[1]),
                cross_section_path(argv[2]);
    Real deployedActuationAngle = std::stod(argv[3]);
    Real beta = std::stod(argv[4]);
    Real gradtol = std::stod(argv[5]);
    Real step_size = std::stod(argv[6]);

    RodLinkage linkage(linkage_path);

    BENCHMARK_RESET();
    RodMaterial mat;
    if (cross_section_path.substr(cross_section_path.size() - 4) == "json") {
        mat.set(*CrossSection::load(cross_section_path), RodMaterial::StiffAxis::D1, false);
    }
    else {
        mat.setContour(20000, 0.3, cross_section_path, 1.0, RodMaterial::StiffAxis::D1);
    }

    linkage.setMaterial(mat);

    std::cout << "Opening linkage with " << linkage.numSegments() << " segments, " << linkage.numJoints() << " joints, and " << linkage.numDoF() << " DoF." << std::endl;

    std::cout << "Solving for rest lengths" << std::endl;
    restlen_solve(linkage);

    ////////////////////////////////////////////////////////////////////////////
    // Computed deployed equilibrium under full actuation
    ////////////////////////////////////////////////////////////////////////////
    // Constrain global rigid motion by fixing the position, orientation of the centermost joint
    const size_t jdo = linkage.dofOffsetForJoint(linkage.centralJoint());
    std::vector<size_t> rigidMotionFixedVars = { jdo + 0, jdo + 1, jdo + 2, jdo + 3, jdo + 4, jdo + 5 };

    // Compute undeployed equilibrium (preserving initial average actuation angle).
    NewtonOptimizerOptions eopts;
    eopts.beta = beta;
    eopts.gradTol = gradtol;
    eopts.verbose = 10;
    eopts.niter = 50;

    open_linkage(linkage, deployedActuationAngle, eopts, linkage.centralJoint(), step_size);
    BENCHMARK_REPORT_NO_MESSAGES();

    linkage.saveVisualizationGeometry("deployed.msh");

    return 0;
}
