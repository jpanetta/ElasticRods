#include <iostream>
#include <fstream>
#include <MeshFEM/GlobalBenchmark.hh>

#include "RodLinkage.hh"
#include "compute_equilibrium.hh"
#include "VectorOperations.hh"

// Generate random number in the range [-1, 1]
Real randUniform() { return 2 * (rand() / double(RAND_MAX)) - 1.0; }

int main(int argc, const char * argv[]) {
    if (argc != 6) {
        std::cout << "usage: " << argv[0] << " in_mesh.msh cross_section.json driving_joint_idx fullAngle numSteps" << std::endl;
        exit(-1);
    }
    const std::string &linkageGraph = argv[1];
    std::cout.precision(19);

    const size_t driving_joint_idx = std::stoi(argv[3]);
    const Real fullAngle = std::stod(argv[4]);
    const size_t numSteps = std::stod(argv[5]);

    RodLinkage linkage(linkageGraph, 10);
    // TODO: determine realistic shear modulus.
    RodMaterial mat(*CrossSection::load(argv[2]));

    // The linkage is driven by opening a central joint (i.e., rotating one of the
    // edge vectors around the normal joint).
    auto &joint = linkage.joint(driving_joint_idx);
    joint.setConstrained(true);

    linkage.setMaterial(mat);
    std::cout << linkage.joints().size() << " joints" << std::endl;

    linkage.saveVisualizationGeometry("pre_rest_len.msh");

    // Determine the rest lengths by fixing the positions of the joint centers
    // and running a constrained optimization with low stretching stiffness.
    auto jointPosVars = linkage.jointPositionDoFIndices();
    linkage.setStretchingStiffness(1e-5);

    std::cout << "Rest length solve" << std::endl;
    compute_equilibrium(linkage, 1000, true, jointPosVars);

    linkage.saveVisualizationGeometry("new_lens.msh");
    linkage.setMaterial(mat);

    std::cout << "Equilibrium solve" << std::endl;
    size_t newton_its = compute_equilibrium(linkage, 1000, true, std::vector<size_t>(), false);

    linkage.saveVisualizationGeometry("post_rest_len.msh");

    Vector3D n = joint.e_A().cross(joint.e_B()).normalized();
    // If the edge vectors point in opposite directions, we actually want to open
    // the angle from B to A (closing the angle from A to B).
    Real angleSign = std::copysign(1.0, joint.e_A().dot(joint.e_B()));

    const Real thetaStep = angleSign * fullAngle / numSteps;

    Vector3D nSinHalfThetaStep = n * sin(0.5 * thetaStep);
    Real cosHalfThetaStep = cos(0.5 * thetaStep);

    linkage.saveVisualizationGeometry("open_it_0.msh");

    Real zPerturbationEpsilon = 1e-7;

    std::ofstream dataFile("open_iterates.txt");
    dataFile.precision(19);

    auto reportIterate = [&](size_t it) {
        dataFile << std::abs(it * thetaStep)
                 << '\t' << newton_its
                 << '\t' << linkage.maxStrain()
                 << '\t' << linkage.energy()
                 << '\t' << linkage.energyStretch()
                 << '\t' << linkage.energyBend()
                 << '\t' << linkage.energyTwist()
                 << '\n';
    };

    reportIterate(0);

    for (size_t it = 1; it <= numSteps; ++it) {
        joint.set_e_B(rotatedVector( nSinHalfThetaStep, cosHalfThetaStep, joint.e_B()));
        joint.set_e_A(rotatedVector(-nSinHalfThetaStep, cosHalfThetaStep, joint.e_A()));
        joint.setConstrained(true); // set the current edge vectors as the constrained orientations

        // Apply a random perturbation to the joint z positions to try to break symmetry.
        auto dofs = linkage.getDoFs();
        assert(jointPosVars.size() % 3 == 0);
        for (size_t i = 0; i < jointPosVars.size() / 3; ++i)
            dofs[jointPosVars[3 * i + 2]] += zPerturbationEpsilon * randUniform();
        linkage.setDoFs(dofs);

        std::cout << "Solving for equilibrium " << it << std::endl;
        newton_its = compute_equilibrium(linkage, 40, true, std::vector<size_t>(), it == 1);
        reportIterate(it);

        linkage.saveVisualizationGeometry("open_it_" + std::to_string(it) + ".msh");
    }

    // auto dofs = linkage.getDoFs();
    // assert(jointPosVars.size() % 3 == 0);
    // for (size_t i = 0; i < jointPosVars.size() / 3; ++i)
    //     dofs.at(jointPosVars[3 * i + 2]) += 1e-3 * randUniform();
    // linkage.setDoFs(dofs);

    compute_equilibrium(linkage, 1000, true);
    linkage.saveVisualizationGeometry("final_equilibrium.msh");

    BENCHMARK_REPORT_NO_MESSAGES();

    return 0;
}
