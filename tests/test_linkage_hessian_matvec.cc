#include <iostream>
#include <fstream>
#include "../RodLinkage.hh"
#if 0
#include "compute_equilibrium.hh"
#endif
#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/GlobalBenchmark.hh>

#include "../CrossSectionMesh.hh"
#include "../open_linkage.hh"
#include <MeshFEM/GaussQuadrature.hh>

int main(int argc, const char * argv[]) {
    if (argc != 3) {
        std::cout << "usage: " << argv[0] << " linkage.msh cross_section.json" << std::endl;
        exit(-1);
    }
    const std::string &linkageGraph = argv[1];

    // const size_t np = 1;
    // tbb::task_scheduler_init init(np);

    std::cout.precision(19);

    RodLinkage linkage(linkageGraph, 10);

    RodMaterial mat;
    const std::string &cross_section_path = argv[2];
    if (cross_section_path.substr(cross_section_path.size() - 4) == "json") {
        mat.set(*CrossSection::load(cross_section_path), RodMaterial::StiffAxis::D1, false);
    }
    else {
        mat.setContour(20000, 0.3, cross_section_path, 1.0, RodMaterial::StiffAxis::D1);
    }
    linkage.setMaterial(mat);

    NewtonOptimizerOptions eopts;
    eopts.beta = 1;
    eopts.gradTol = 1e-2;
    eopts.verbose = 0;
    eopts.niter = 10;
    // open_linkage(linkage, 0.8, eopts, linkage.centralJoint());

    linkage.saveVisualizationGeometry("test.msh");

    Eigen::VectorXd v = Eigen::VectorXd::Random(linkage.numExtendedDoFPSRL());
    v += Eigen::VectorXd::Ones(linkage.numExtendedDoFPSRL());
    // v.tail(v.size() - linkage.dofOffsetForJoint(0)).setZero();
    // v.head(linkage.numDoF()).setZero();
    // v.tail(linkage.numSegments()).setZero();

    auto test = [&]() {
        linkage.setDoFs(linkage.getDoFs()); // clear cache...
        {
            std::cout << "Bergou 2010:" << std::endl;
            auto Hv_direct = linkage.applyHessianPerSegmentRestlen(v);
            auto Hsp = linkage.hessianPerSegmentRestlenSparsityPattern();
            linkage.hessianPerSegmentRestlen(Hsp, RodLinkage::EnergyType::Full);
            auto Hv_sparse_matvec = Hsp.apply(v);
            std::cout << (Hv_direct - Hv_sparse_matvec).norm() / (Hv_sparse_matvec).norm() << std::endl;
        }

        linkage.setBendingEnergyType(ElasticRod::BendingEnergyType::Bergou2008);

        linkage.setDoFs(linkage.getDoFs()); // clear cache...
        {
            std::cout << "Bergou 2008:" << std::endl;
            auto Hv_direct = linkage.applyHessianPerSegmentRestlen(v);
            auto Hsp = linkage.hessianPerSegmentRestlenSparsityPattern();
            linkage.hessianPerSegmentRestlen(Hsp, RodLinkage::EnergyType::Full);
            auto Hv_sparse_matvec = Hsp.apply(v);
            std::cout << (Hv_direct - Hv_sparse_matvec).norm() / (Hv_sparse_matvec).norm() << std::endl;
            // Also test hessian against FD approx in deployed config!
        }
    };

    test();

    std::cout << std::endl << "Post perturbation: " << std::endl;
    const size_t ndofs = linkage.numDoF();

    Eigen::VectorXd perturb = 1e-2 * Eigen::VectorXd::Random(ndofs);
    // Perturb the joint rotations more significantly
    for (size_t ji = 0; ji < linkage.numJoints(); ++ji)
        perturb.segment<3>(linkage.dofOffsetForJoint(ji) + 3) = 0.3 * Eigen::Vector3d::Random();

    linkage.setDoFs(linkage.getDoFs() + perturb);

    linkage.saveVisualizationGeometry("test_perturbed.msh");

    test();

    std::cout << std::endl << "Post rotation parametrization update: " << std::endl;
    linkage.updateRotationParametrizations();
    test();

    return 0;
}
