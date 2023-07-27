#include <iostream>
#include <fstream>
#include "../RodLinkage.hh"
#include "../LinkageTerminalEdgeSensitivity.hh"
#if 0
#include "compute_equilibrium.hh"
#endif
#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/GlobalBenchmark.hh>

#include "../CrossSectionMesh.hh"
#include <MeshFEM/GaussQuadrature.hh>
#include <MeshFEM/MSHFieldWriter.hh>

// Generate random number in the range [-1, 1]
Real randUniform() { return 2 * (rand() / double(RAND_MAX)) - 1.0; }

Eigen::VectorXd getDofPerturbation(size_t nvars, Real epsilon = 1.0) {
    Eigen::VectorXd perturbation(nvars);
    BENCHMARK_START_TIMER("Gen Perturbation");
    for (size_t i = 0; i < nvars; ++i)
        perturbation[i] = epsilon * randUniform();
    BENCHMARK_STOP_TIMER("Gen Perturbation");
    return perturbation;
}

void fdGradientTest(RodLinkage &linkage, Real epsilon, Eigen::VectorXd perturbation = Eigen::VectorXd()) {
    const size_t nvars = linkage.numExtendedDoFPSRL() /*, njoints = linkage.numJoints()*/;
    if (size_t(perturbation.size()) != nvars)
        perturbation = getDofPerturbation(nvars, 1.0);
    perturbation *= epsilon;
    // // Only perturb the rest lengths...
    // perturbation.segment(0, linkage.numDoF()).setZero();

    auto dofs = linkage.getExtendedDoFsPSRL();

    // linkage.updateSourceFrame();
    auto gradEnergy = linkage.gradientPerSegmentRestlen(false, RodLinkage::EnergyType::Full);
    std::cout << "gradient norm: " << gradEnergy.norm() << std::endl;

    auto perturbed_dofs = dofs;
    Real analytic_diff_energy = gradEnergy.dot(perturbation);

    linkage.setExtendedDoFsPSRL(dofs + perturbation);
    Real perturbed_energy_plus = linkage.energy();

    linkage.setExtendedDoFsPSRL(dofs - perturbation);
    Real perturbed_energy_minus = linkage.energy();

    Real centered_diff_energy = 0.5 * (perturbed_energy_plus - perturbed_energy_minus);

    std::cout << "Centered diff energy: " << centered_diff_energy << std::endl;
    std::cout << "Analytic diff energy: " << analytic_diff_energy << std::endl;
}

void fdHessianTest(RodLinkage &linkage, Real epsilon) {
    const size_t nvars = linkage.numExtendedDoFPSRL();
    auto perturbation = getDofPerturbation(nvars, epsilon);

    // // Only perturb the rest lengths...
    // perturbation.segment(0, linkage.numDoF()).setZero();
    // Only perturb the deformed configuration
    // perturbation.tail(linkage.numSegments()).setZero();

    auto dofs = linkage.getExtendedDoFsPSRL();

    linkage.updateSourceFrame();
    auto H = linkage.hessianPerSegmentRestlenSparsityPattern();
    linkage.hessianPerSegmentRestlen(H, RodLinkage::EnergyType::Full);

    {
        static size_t count = 0;
        H.dump("hessian_" + std::to_string(count++));
    }

    auto gradEnergy = linkage.gradientPerSegmentRestlen(true, RodLinkage::EnergyType::Full);

    Eigen::VectorXd  analytic_diff_grad_energy = H.apply(perturbation);

    auto perturbed_dofs = dofs;
    for (size_t i = 0; i < nvars; ++i)
        perturbed_dofs[i] += perturbation[i];

    linkage.setExtendedDoFsPSRL(perturbed_dofs);
    auto perturbed_energy_grad_plus = linkage.gradientPerSegmentRestlen(false, RodLinkage::EnergyType::Full);

    for (size_t i = 0; i < nvars; ++i)
        perturbed_dofs[i] -= 2 * perturbation[i];

    linkage.setExtendedDoFsPSRL(perturbed_dofs);
    auto perturbed_energy_grad_minus = linkage.gradientPerSegmentRestlen(false, RodLinkage::EnergyType::Full);

    Eigen::VectorXd centered_diff_grad_energy = 0.5 * (perturbed_energy_grad_plus - perturbed_energy_grad_minus);

    // std::cout << "Centered diff grad energy:" << centered_diff_grad_energy.transpose() << std::endl;
    // std::cout << "Analytic diff grad energy:" << analytic_diff_grad_energy.transpose() << std::endl;

    Real maxRelDiff = 0;
    size_t maxIdx = 0;
    for (size_t i = 0; i < nvars; ++i) {
        Real relDiff = std::abs((centered_diff_grad_energy[i] - analytic_diff_grad_energy[i]) / centered_diff_grad_energy[i]);
        if (relDiff > maxRelDiff) {
            maxIdx = i;
            maxRelDiff = relDiff;
        }
    }
    std::cout << "Hessian max rel error:\t" << maxRelDiff << std::endl;
    std::cout << "For values: \t" << centered_diff_grad_energy[maxIdx] << ", " << analytic_diff_grad_energy[maxIdx] << std::endl;
}

void testRestLenLaplacian(RodLinkage &linkage) {
    std::cout << std::endl;
    std::cout << "Rest length Laplacian energy:   " << linkage.restLengthLaplacianEnergy() << std::endl;

    auto g = linkage.restLengthLaplacianGradEnergy();
    Eigen::VectorXd rl = linkage.getRestLengths();
    std::cout << "Laplacian energy from gradient: " << 0.5 * rl.dot(g) << std::endl;

    // std::cout << "Rest length Laplacian gradient: " << g.transpose() << std::endl;
    // std::cout << "         Gradient from hessian: " << linkage.restLengthLaplacianHessEnergy().apply(rl).transpose() << std::endl;
    std::cout << "Laplacian energy from Hessian: " << 0.5 * rl.dot(linkage.restLengthLaplacianHessEnergy().apply(rl)) << std::endl;
}

void fdMassTest(RodLinkage &linkage, Real epsilon) {
    const size_t nvars = linkage.numDoF();
    auto perturbation = getDofPerturbation(nvars, epsilon);

    auto dofs = linkage.getDoFs();

    auto M = linkage.massMatrix(false);

    Real kineticEnergy = 0.5 * perturbation.dot(M.apply(perturbation));
    std::cout << "Kinetic energy from mass matrix: " << kineticEnergy << std::endl;

    auto linkagePlus = linkage;
    linkagePlus.setDoFs(dofs + perturbation);

    auto linkageMinus = linkage;
    linkageMinus.setDoFs(dofs - perturbation);

    // Finite difference kinetic energy
    Real fdKineticEnergy = 0;
    constexpr size_t K = 2;
    constexpr size_t Deg = 2;
    for (size_t si = 0; si < linkage.numSegments(); ++si) {
        const auto &r     = linkage.segment(si).rod;
        const auto &rPlus = linkagePlus.segment(si).rod;
        const auto &rMinus = linkageMinus.segment(si).rod;

        const size_t ne = r.numEdges();

        for (size_t j = 0; j < ne; ++j) {
            const auto &mat = r.material(j);
            Vector3D x_j_dot   = 0.5 * (rPlus.deformedPoint(j    ) - rMinus.deformedPoint(j    )),
                     x_jp1_dot = 0.5 * (rPlus.deformedPoint(j + 1) - rMinus.deformedPoint(j + 1));
            Vector3D d1_dot = 0.5 * (rPlus.deformedMaterialFrameD1(j) - rMinus.deformedMaterialFrameD1(j)),
                     d2_dot = 0.5 * (rPlus.deformedMaterialFrameD2(j) - rMinus.deformedMaterialFrameD2(j));
            const Real restLen = r.restLengths()[j];

            // Integrate the kinetic energy contribution over the cross-section.
            for (const auto &e : mat.crossSectionMesh().elements()) {
                // Linear interpolant evaluating the cross-section coordinate functions
                Interpolant<Real, K, 1> s, t;
                for (const auto &v : e.vertices()) {
                    s[v.localIndex()] = v.node()->p[0];
                    t[v.localIndex()] = v.node()->p[1];
                }
                fdKineticEnergy += Quadrature<1, Deg>::integrate([&](const EvalPt<1> &alpha) {
                    return Quadrature<K, Deg>::integrate([&] (const EvalPt<K> &p) {
                            return (std::get<0>(alpha) * x_j_dot + std::get<1>(alpha) * x_jp1_dot
                                    + s(p) * d1_dot + t(p) * d2_dot).squaredNorm(); },
                            e->volume());
                }, r.density(j) * restLen);
            }
        }
    }

    fdKineticEnergy *= 0.5;
    std::cout << "Kinetic energy from finite diff: " << fdKineticEnergy << std::endl;
}

void fdLinfVelocityTest(RodLinkage &linkage, Real epsilon) {
    const size_t nvars = linkage.numDoF();
    auto perturbation = getDofPerturbation(nvars, epsilon);

    auto dofs = linkage.getDoFs();

    auto linkagePlus = linkage;
    linkagePlus.setDoFs(dofs + perturbation);

    auto linkageMinus = linkage;
    linkageMinus.setDoFs(dofs - perturbation);

    linkage.updateSourceFrame();
    std::cout << "Analytic approx   max velocity: " << linkage.approxLinfVelocity(perturbation) << std::endl;

    Real fdMaxVelocity = 0;
    for (size_t si = 0; si < linkage.numSegments(); ++si) {
        const auto &rPlus = linkagePlus.segment(si).rod;
        const auto &rMinus = linkageMinus.segment(si).rod;

        std::vector<MeshIO::IOVertex > verticesPlus, verticesMinus;
        std::vector<MeshIO::IOElement> quads;
        rPlus.visualizationGeometry(verticesPlus, quads);
        rMinus.visualizationGeometry(verticesMinus, quads);

        for (size_t i = 0; i < verticesPlus.size(); ++i)
            fdMaxVelocity = std::max(fdMaxVelocity, 0.5 * (verticesPlus[i].point - verticesMinus[i].point).norm());
    }

    std::cout << "Finite difference max velocity: " << fdMaxVelocity << std::endl;
}

void fdTerminalEdgeTest(RodLinkage &linkage, size_t ji, size_t ABoffset, bool updatedSource, Real epsilon, bool testHessian = false) {
    auto dofs = linkage.getDoFs();
    // Only perturb the omega/alpha/len variables of joint "ji"
    const size_t jo = linkage.dofOffsetForJoint(ji);

    const auto &joint = linkage.joint(ji);
    size_t si = (ABoffset == 0) ? joint.segmentsA()[0] : joint.segmentsB()[0];

    RodLinkage::TerminalEdge edge;
    {
        bool isStart;
        double s_jA, s_jB;
        std::tie(s_jA, s_jB, isStart) = joint.terminalEdgeIdentification(si);
        edge = isStart ? RodLinkage::TerminalEdge::Start : RodLinkage::TerminalEdge::End;
    }


    auto sensitivity = linkage.getTerminalEdgeSensitivity(si, edge, updatedSource, testHessian);
    const size_t j = sensitivity.j;

    Eigen::Matrix<Real, 4, 6> fd_jacobian;
    std::vector<Eigen::Matrix<Real, 6, 6>> jacobian_plus, jacobian_minus;

    // Jacobian entries needed for centered finite difference Hessian approximation
    if (testHessian) { jacobian_plus.resize(4), jacobian_minus.resize(4); }

    for (size_t var = 0; var < 6; ++var) {
        const auto &r = linkage.segment(si).rod;
        auto perturbed_dofs = dofs;
        perturbed_dofs(jo + 3 + var) += epsilon;
        linkage.setDoFs(perturbed_dofs);
        Vector3D e_plus = r.deformedPoints()[j + 1] - r.deformedPoints()[j];
        Real theta_plus = r.theta(j);
        auto sensitivity_plus = linkage.getTerminalEdgeSensitivity(si, edge, false, false);
        perturbed_dofs(jo + 3 + var) -= 2 * epsilon;
        linkage.setDoFs(perturbed_dofs);
        Vector3D e_minus = r.deformedPoints()[j + 1] - r.deformedPoints()[j];
        Real theta_minus = r.theta(j);
        auto sensitivity_minus = linkage.getTerminalEdgeSensitivity(si, edge, false, false);

        fd_jacobian.block<3, 1>(0, var) = (    e_plus -     e_minus) / (2 * epsilon);
        fd_jacobian(3, var) =             (theta_plus - theta_minus) / (2 * epsilon);

        if (testHessian) {
            // std::cout << "sensitivity_plus  jacobian: " << sensitivity_plus .jacobian << std::endl;
            // std::cout << "sensitivity_minus jacobian: " << sensitivity_minus.jacobian << std::endl;
            for (size_t comp = 0; comp < 4; ++comp) {
                jacobian_plus [comp].col(var) = sensitivity_plus .jacobian.row(comp).transpose();
                jacobian_minus[comp].col(var) = sensitivity_minus.jacobian.row(comp).transpose();
            }
        }
    }

    auto analytic_jacobian = sensitivity.jacobian;
    analytic_jacobian.block<3, 6>(0, 0) *= sensitivity.s_jX;

    std::string test_type(updatedSource ? "updated source " : "");
    // std::cout << test_type << "fd_jacobian:\n" << fd_jacobian << std::endl;
    // std::cout << test_type << "analytic_jacobian:\n" << analytic_jacobian << std::endl;
    std::cout << test_type << "jacobian max rel error: " << (fd_jacobian - analytic_jacobian).cwiseAbs().maxCoeff() / fd_jacobian.norm() << std::endl;

    if (testHessian) {
        auto analytic_hessian = sensitivity.hessian;
        for (size_t comp = 0; testHessian && (comp < 4); ++comp) {
            auto fd_hessian_comp = ((jacobian_plus[comp] - jacobian_minus[comp]) / (2 * epsilon)).eval();
            int r = 0, c = 0;
            Real maxError = (fd_hessian_comp - analytic_hessian[comp]).cwiseAbs().maxCoeff(&r, &c);
            std::string test_type = " hessian of component " + std::to_string(comp);
            std::cout << "fd" << test_type << ":\n" << fd_hessian_comp << std::endl;
            std::cout << "analytic" << test_type << ":\n" << analytic_hessian[comp] << std::endl;
            std::cout << "max rel error for" << test_type << ": " << maxError / fd_hessian_comp.norm() << std::endl;
            std::cout << "for entry " << r << ", " << c << "; computed by differencing " << jacobian_plus[comp](r, c) << ", "  << jacobian_minus[comp](r, c) << ", " << std::endl;
        }
    }
}

int main(int argc, const char * argv[]) {
    if ((argc != 4) && (argc != 5) && (argc != 6)) {
        std::cout << "usage: " << argv[0] << " linkage.msh cross_section.json constrained_joint_idx [numprocs] [fd_eps]" << std::endl;
        exit(-1);
    }
    const std::string &linkageGraph = argv[1];

    Real fd_eps = 1e-7;
    if (argc >= 6) { fd_eps = std::stod(argv[5]); }

    std::cout.precision(19);

    BENCHMARK_START_TIMER_SECTION("Load and initialize");

    RodMaterial mat(*CrossSection::load(argv[2]), RodMaterial::StiffAxis::D1, true /* keep the cross-section mesh so that we can integrate over it in the mass matrix test */);
    const auto bendingEnergyType = RodLinkage::BEnergyType::Bergou2008;
    // const auto bendingEnergyType = RodLinkage::BEnergyType::Bergou2010;

    RodLinkage linkage(linkageGraph, 5);
    // RodLinkage linkage(linkageGraph, 10);
    // RodLinkage linkage(linkageGraph, 3);
    linkage.setMaterial(mat);
    linkage.setBendingEnergyType(bendingEnergyType);
    // linkage.writeRodDebugData("post_construction_rod_data.msh");

    auto dofs = linkage.getDoFs();
    linkage.setDoFs(dofs);
    // linkage.writeRodDebugData("dof_reset_rod_data.msh");

    const size_t nvars = linkage.numDoF();
    std::cout << "Number of DoFs: " << nvars << std::endl;
    std::cout << "Number of extended DoFs: " << linkage.numExtendedDoFPSRL() << std::endl;

    linkage.setDoFs(dofs + getDofPerturbation(nvars, 1e-3));

    std::cout << "pre updateRotationParametrizations fd tests" << std::endl;
    fdHessianTest(linkage, fd_eps);
    fdGradientTest(linkage, fd_eps);
    linkage.updateSourceFrame();
    fdTerminalEdgeTest(linkage, 0, 0, true, fd_eps, true);
    linkage.updateSourceFrame();
    fdTerminalEdgeTest(linkage, 0, 1, true, fd_eps, true);
    std::cout << std::endl;

    auto pre_reset_dofs = linkage.getDoFs();
    Real pre_reset_energy = linkage.energy();
    linkage.updateRotationParametrizations();
    auto post_reset_dofs = linkage.getDoFs();
    std::cout << "Joint parametrization update changed linkage dofs by " << (post_reset_dofs - pre_reset_dofs).norm()
              << ", energy by " << std::abs(linkage.energy() - pre_reset_energy)
              << std::endl;
    std::cout << "post updateRotationParametrizations fd tests" << std::endl;
    fdHessianTest(linkage, fd_eps); // must come before fdGradientTest/fdTerminalEdgeTest--those will perturb the rotation variables!
    linkage.setDoFs(post_reset_dofs);
    fdGradientTest(linkage, fd_eps);
    linkage.setDoFs(post_reset_dofs);
    linkage.updateSourceFrame();
    fdTerminalEdgeTest(linkage, 0, 0, true, fd_eps, true);
    linkage.setDoFs(post_reset_dofs);
    linkage.updateSourceFrame();
    fdTerminalEdgeTest(linkage, 0, 1, true, fd_eps, true);
    linkage.setDoFs(post_reset_dofs);
    std::cout << std::endl;

    auto lenVars = linkage.lengthVars();
    Real minLen = std::numeric_limits<Real>::max();
    for (size_t var : lenVars)
        minLen = std::min(minLen, post_reset_dofs[var]);

    std::cout << "Min length var: " << minLen << std::endl;

    // fdGradientTest(linkage, fd_eps);
    // fdHessianTest(linkage, fd_eps);

    // testRestLenLaplacian(linkage);

    // {
    //     // Test gradient evaluation away from the source configuration
    //     auto dofs = linkage.getDoFs();
    //     for (Real &d : dofs)
    //         d += 1e-4 * randUniform();
    //     linkage.setDoFs(dofs);
    // }

    // linkage.writeRodDebugData("rod_debug.msh");
    // linkage.writeLinkageDebugData("linkage_debug.msh");
    linkage.saveVisualizationGeometry("linkage_geometry.msh");

    BENCHMARK_STOP_TIMER_SECTION("Load and initialize");

#if 1
    BENCHMARK_START_TIMER_SECTION("FD tests");
    std::cout << std::endl;
    // fdMassTest(linkage, 1e-8);
    // fdLinfVelocityTest(linkage, fd_eps);
    std::cout << std::endl;



    // linkage.updateSourceFrame();

    // fdTerminalEdgeTest(linkage, 0, 0, true, fd_eps, true);
    // fdTerminalEdgeTest(linkage, 0, 1, true, fd_eps, true);

    BENCHMARK_STOP_TIMER_SECTION("FD tests");

    {
        RodLinkage l2(linkageGraph);
        l2.setBendingEnergyType(bendingEnergyType);
        l2.setMaterial(mat);

        // auto dofs = l2.getDoFs();
        // l2.setDoFs(dofs);

        l2.updateSourceFrame();
        // l2.setExtendedDoFsPSRL(l2.getExtendedDoFsPSRL());
        l2.setExtendedDoFsPSRL(l2.getExtendedDoFsPSRL() + getDofPerturbation(l2.numExtendedDoFPSRL(), 1e-2));
        // l2.updateSourceFrame();
        // l2.updateRotationParametrizations();
        using EType = RodLinkage_T<ADReal>::EnergyType;
        RodLinkage_T<ADReal> ldiff(l2);
        ldiff.setBendingEnergyType(RodLinkage_T<ADReal>::BEnergyType(bendingEnergyType));
        std::cout << std::endl;
        std::cout << l2.energy() << std::endl;
        std::cout << ldiff.energy() << std::endl;
        std::cout << l2.gradient(false, RodLinkage::EnergyType::Full, true).norm() << std::endl;
        std::cout << ldiff.gradient(false, EType::Full, true).norm() << std::endl;

        srand(1);
        auto perturb = getDofPerturbation(l2.numExtendedDoFPSRL(), fd_eps);
        // perturb.head(ldiff.numDoF()).setZero();
        // perturb.tail(ldiff.numSegments()).setZero();

        auto ad_dofs = ldiff.getExtendedDoFsPSRL();
        for (int i = 0; i < perturb.size(); ++i)
            ad_dofs[i].derivatives()[0] = perturb(i);
        ldiff.setExtendedDoFsPSRL(ad_dofs);

        const EType et = EType::Stretch;

        // Test autodiff gradient
        std::cout << std::endl;
        std::cout << "AutoDiff energy full diff: " << ldiff.energy().derivatives() << std::endl;
        std::cout << "Analytic energy full diff: " << ldiff.gradientPerSegmentRestlen(false, EType::Full).dot(perturb) << std::endl;
        std::cout << "AutoDiff energy stre diff: " << ldiff.energyStretch().derivatives() << std::endl;
        std::cout << "Analytic energy stre diff: " << ldiff.gradientPerSegmentRestlen(false, EType::Stretch).dot(perturb) << std::endl;
        std::cout << "AutoDiff energy bend diff: " << ldiff.energyBend().derivatives() << std::endl;
        std::cout << "Analytic energy bend diff: " << ldiff.gradientPerSegmentRestlen(false, EType::Bend).dot(perturb) << std::endl;
        std::cout << "AutoDiff energy twst diff: " << ldiff.energyTwist().derivatives() << std::endl;
        std::cout << "Analytic energy twst diff: " << ldiff.gradientPerSegmentRestlen(false, EType::Twist).dot(perturb) << std::endl;

        // Test autodiff hessian
        ldiff.updateSourceFrame();
        auto H = ldiff.hessianPerSegmentRestlenSparsityPattern();
        ldiff.hessianPerSegmentRestlen(H, et);

        auto g = ldiff.gradientPerSegmentRestlen(false, et);
        for (int i = 0; i < g.size(); ++i) g[i].value() = g[i].derivatives()[0];
        std::cout << "AutoDiff grad diff rel error: " << (H.apply(Eigen::Matrix<ADReal, Eigen::Dynamic, 1>(perturb)) - g).norm() / g.norm() << std::endl;

        // Test autodiff of hessian
        auto Hplus = H, Hminus = H;
        Hplus.setZero(), Hminus.setZero();

        ldiff.setExtendedDoFsPSRL(ad_dofs + perturb);
        ldiff.updateSourceFrame();
        ldiff.hessianPerSegmentRestlen(Hplus, et);

        ldiff.setExtendedDoFsPSRL(ad_dofs - perturb);
        ldiff.updateSourceFrame();
        ldiff.hessianPerSegmentRestlen(Hminus, et);
        Hplus.addWithIdenticalSparsity(Hminus, -1.0);
        Hplus.scale(1.0 / 2.0);

        auto H_ad = H;
        for (auto &x : H_ad.Ax) x.value() = x.derivatives()[0];

        auto Hdiff_fd = Eigen::Map<Eigen::Matrix<ADReal, Eigen::Dynamic, 1>>(Hplus.Ax.data(), Hplus.Ax.size());
        auto Hdiff_ad = Eigen::Map<Eigen::Matrix<ADReal, Eigen::Dynamic, 1>>(H_ad.Ax.data(), H_ad.Ax.size());
        
        // std::cout << std::endl;
        // std::cout << Hdiff_fd.transpose() << std::endl;
        // std::cout << Hdiff_ad.transpose() << std::endl;
        // std::cout << std::endl;

        std::cout << "FD vs AD Hessian derivative rel error: " << (Hdiff_ad - Hdiff_fd).norm() / Hdiff_ad.norm() << std::endl;
    }

    // Test Hessian matvec
    std::cout << std::endl;
    {
        RodLinkage l2(linkageGraph);
        l2.setBendingEnergyType(bendingEnergyType);
        l2.setMaterial(mat);
        srand(1);
        auto perturb = getDofPerturbation(l2.numExtendedDoF(), fd_eps);

        auto Hv = l2.applyHessian(perturb, true);
        auto HvMatrixImpl = l2.hessian(ElasticRod::EnergyType::Full, true).apply(perturb);
        std::cout << "Hessian matvec rel error: " << (Hv - HvMatrixImpl).norm() / HvMatrixImpl.norm() << std::endl;

        auto perturbPSRL = getDofPerturbation(l2.numExtendedDoFPSRL(), fd_eps);
        auto HvPSRL = l2.applyHessianPerSegmentRestlen(perturbPSRL);

        auto HPSRL = l2.hessianPerSegmentRestlenSparsityPattern();
        l2.hessianPerSegmentRestlen(HPSRL);
        auto HvPSRLMatrixImpl = HPSRL.apply(perturbPSRL);
        std::cout << "PSRL Hessian matvec rel error: " << (HvPSRL - HvPSRLMatrixImpl).norm() / HvPSRLMatrixImpl.norm() << std::endl;
    }

    {
        std::vector<std::vector<size_t>> polylinesA, polylinesB;
        std::vector<Point3D> points;
        std::vector<Vector3D> normals;
        std::vector<Real> stresses;
        linkage.florinVisualizationGeometry(polylinesA, polylinesB, points, normals, stresses);

        std::vector<MeshIO::IOVertex > vertices(points.size());
        VectorField<Real, 3> normalField(points.size());
        ScalarField<Real> stressField(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            vertices[i].point = points[i];
            normalField(i) = normals[i];
            stressField[i] = stresses[i];
        }

        std::vector<MeshIO::IOElement> elements;
        for (const auto &polyline : polylinesA) {
            for (size_t j = 0; j < polyline.size() - 1; ++j)
                elements.emplace_back(polyline[j], polyline[j + 1]);
        }
        for (const auto &polyline : polylinesB) {
            for (size_t j = 0; j < polyline.size() - 1; ++j)
                elements.emplace_back(polyline[j], polyline[j + 1]);
        }

        MSHFieldWriter writer("debug_vis_geom.msh", vertices, elements);
        writer.addField("normals", normalField, DomainType::PER_NODE);
        writer.addField("stresses", stressField, DomainType::PER_NODE);
    }

    return 0;
#endif

#if 0
    try {
        compute_equilibrium(linkage, 1000, true, std::vector<size_t>(), false);
    }
    catch (const std::runtime_error &e) {
        std::cout << "Error during equilibrium solve: " << e.what() << std::endl;
    }
#endif

    BENCHMARK_REPORT_NO_MESSAGES();

    linkage.saveVisualizationGeometry("equilibrium_config.msh");

    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> tris;
    std::vector<size_t> originJoint;
    linkage.triangulation(vertices, tris, originJoint);
    MeshIO::save("triangulation.msh", vertices, tris);
    // MeshIO::save("triangulation.msh", vertices, tris, MeshIO::FMT_GUESS, MeshIO::MESH_QUAD);

    return 0;
}
