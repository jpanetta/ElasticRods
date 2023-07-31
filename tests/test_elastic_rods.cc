#include <iostream>
#include "../ElasticRod.hh"
#include "../SparseMatrixOps.hh"
#include <MeshFEM/MeshIO.hh>
#include <map>
#include "../CrossSectionMesh.hh"
#include <MeshFEM/GaussQuadrature.hh>
#include "../compute_equilibrium.hh"
#include "../knitro_solver.hh"
#include <MeshFEM/AutomaticDifferentiation.hh>

// Generate random number in the range [-1, 1]
Real randUniform() { return 2 * (rand() / double(RAND_MAX)) - 1.0; }

void generatePerturbations(const size_t nv, Real pos_epsilon, Real theta_epsilon, Real restLen_epsilon,
                           std::vector<Vector3D> &centerlinePerturbation,
                           std::vector<Real> &thetaPerturbation,
                           std::vector<Real> &restLenPerturbation) {
    const size_t ne = nv - 1;
    centerlinePerturbation.resize(nv);
    thetaPerturbation.resize(ne);
    restLenPerturbation.resize(ne);
    for (auto &v : centerlinePerturbation)
        v = pos_epsilon * Vector3D(randUniform(), randUniform(), randUniform());
    for (size_t j = 0; j < ne; ++j) {
        thetaPerturbation[j] = theta_epsilon * randUniform();
        restLenPerturbation[j] = restLen_epsilon * randUniform();
    }
}

Eigen::VectorXd getDoFPerturbation(const std::vector<Vector3D> &centerlinePerturbation,
                                   const std::vector<Real> &thetaPerturbation,
                                   const std::vector<Real> &restLenPerturbation = std::vector<Real>()) {
    Eigen::VectorXd dofPerturbation(3 * centerlinePerturbation.size() + thetaPerturbation.size() + restLenPerturbation.size());
    size_t offset = 0;
    for (size_t i = 0; i < centerlinePerturbation.size(); ++i) {
        dofPerturbation[offset++] = centerlinePerturbation[i][0];
        dofPerturbation[offset++] = centerlinePerturbation[i][1];
        dofPerturbation[offset++] = centerlinePerturbation[i][2];
    }
    for (Real val :   thetaPerturbation) dofPerturbation[offset++] = val;
    for (Real val : restLenPerturbation) dofPerturbation[offset++] = val;

    return dofPerturbation;
}

void fdHessianTest(ElasticRod &e, Real pos_epsilon, Real theta_epsilon, Real restLen_epsilon = 0) {
    const size_t nv = e.numVertices(), ne = e.numEdges();

    std::vector<Vector3D> centerlinePerturbation;
    std::vector<Real>     thetaPerturbation, restLenPerturbation;
    generatePerturbations(nv, pos_epsilon, theta_epsilon, restLen_epsilon, centerlinePerturbation, thetaPerturbation, restLenPerturbation);
    auto dofPerturbation = getDoFPerturbation(centerlinePerturbation, thetaPerturbation, restLenPerturbation);

    auto points = e.deformedPoints();
    auto thetas = e.thetas();
    auto rlens  = e.restLengths();

    for (size_t i = 0; i < nv; ++i) points[i] += centerlinePerturbation[i];
    for (size_t j = 0; j < ne; ++j) thetas[j] +=      thetaPerturbation[j];
    for (size_t j = 0; j < ne; ++j)  rlens[j] +=    restLenPerturbation[j];

    e.updateSourceFrame();
    auto Hsp = e.hessianSparsityPattern(true);
    auto Hfull    = Hsp; e.hessian(Hfull   , ElasticRod::EnergyType::Full,    true);
    auto Hstretch = Hsp; e.hessian(Hstretch, ElasticRod::EnergyType::Stretch, true);
    auto Htwist   = Hsp; e.hessian(Htwist  , ElasticRod::EnergyType::Twist  , true);
    auto Hbend    = Hsp; e.hessian(Hbend   , ElasticRod::EnergyType::Bend   , true);

    e.setDeformedConfiguration(points, thetas);
    e.setRestLengths(rlens);
    auto gFullPlus    = e.gradEnergy       (false, true);
    auto gStretchPlus = e.gradEnergyStretch(       true);
    auto gBendPlus    = e.gradEnergyBend   (false, true);
    auto gTwistPlus   = e.gradEnergyTwist  (false, true);

    for (size_t i = 0; i < nv; ++i) points[i] -= 2 * centerlinePerturbation[i];
    for (size_t j = 0; j < ne; ++j) thetas[j] -= 2 *      thetaPerturbation[j];
    for (size_t j = 0; j < ne; ++j)  rlens[j] -= 2 *    restLenPerturbation[j];

    e.setDeformedConfiguration(points, thetas);
    e.setRestLengths(rlens);
    auto gFullMinus    = e.gradEnergy       (false, true);
    auto gStretchMinus = e.gradEnergyStretch(       true);
    auto gBendMinus    = e.gradEnergyBend   (false, true);
    auto gTwistMinus   = e.gradEnergyTwist  (false, true);

    auto reportDiff = [](const std::string &name, const Eigen::VectorXd &analytic_diff_grad,
                         const ElasticRod::Gradient &gPlus, const ElasticRod::Gradient &gMinus) {
        std::cout << "Analytic/finite diff grad " << name << ":\n";
        std::cout << analytic_diff_grad.transpose() << std::endl;

        auto finite_diff_grad = (0.5 * (gPlus - gMinus)).eval();
        std::cout << finite_diff_grad.transpose() << std::endl;

        Real error = (analytic_diff_grad - finite_diff_grad).norm();
        std::cout << "rel error: " << error / finite_diff_grad.norm() << std::endl;
    };

    reportDiff("full"   , Hfull   .apply(dofPerturbation), gFullPlus   , gFullMinus   );
    reportDiff("stretch", Hstretch.apply(dofPerturbation), gStretchPlus, gStretchMinus);
    reportDiff("bend"   , Hbend   .apply(dofPerturbation), gBendPlus   , gBendMinus   );
    reportDiff("twist"  , Htwist  .apply(dofPerturbation), gTwistPlus  , gTwistMinus  );
}

void fdGradientTest(ElasticRod &e, Real pos_epsilon, Real theta_epsilon, Real restLen_epsilon = 0) {
    const size_t nv = e.numVertices(), ne = e.numEdges();
    std::vector<Vector3D> centerlinePerturbation;
    std::vector<Real>     thetaPerturbation, restLenPerturbation;
    generatePerturbations(nv, pos_epsilon, theta_epsilon, restLen_epsilon, centerlinePerturbation, thetaPerturbation, restLenPerturbation);

    Vector3D energies(e.energyStretch(), e.energyBend(), e.energyTwist());
    auto gradStretch = e.gradEnergyStretch(true),
            gradBend = e.gradEnergyBend(false, true),
           gradTwist = e.gradEnergyTwist(false, true);

    Vector3D analytic_diff_energies(Vector3D::Zero());
    for (size_t i = 0; i < nv; ++i) {
        analytic_diff_energies += Vector3D(gradStretch.gradPos(i).dot(centerlinePerturbation[i]),
                                           gradBend   .gradPos(i).dot(centerlinePerturbation[i]),
                                           gradTwist  .gradPos(i).dot(centerlinePerturbation[i]));
    }

    for (size_t j = 0; j < ne; ++j) {
        analytic_diff_energies += Vector3D(gradStretch.gradTheta  (j) *   thetaPerturbation[j],
                                           gradBend   .gradTheta  (j) *   thetaPerturbation[j],
                                           gradTwist  .gradTheta  (j) *   thetaPerturbation[j])
                               +  Vector3D(gradStretch.gradRestLen(j) * restLenPerturbation[j],
                                           gradBend   .gradRestLen(j) * restLenPerturbation[j],
                                           gradTwist  .gradRestLen(j) * restLenPerturbation[j]);
    }

    auto points = e.deformedPoints();
    auto thetas = e.thetas();
    auto restLens = e.restLengths();

    for (size_t i = 0; i < nv; ++i)
        points[i] += centerlinePerturbation[i];
    for (size_t j = 0; j < ne; ++j) {
        thetas[j] += thetaPerturbation[j];
        restLens[j] += restLenPerturbation[j];
    }
    e.setDeformedConfiguration(points, thetas);
    e.setRestLengths(restLens);
    Vector3D plusPerturbedEnergies(e.energyStretch(), e.energyBend(), e.energyTwist());

    for (size_t i = 0; i < nv; ++i)
        points[i] -= 2 * centerlinePerturbation[i];
    for (size_t j = 0; j < ne; ++j) {
        thetas[j] -= 2 * thetaPerturbation[j];
        restLens[j] -= 2 * restLenPerturbation[j];
    }
    e.setDeformedConfiguration(points, thetas);
    e.setRestLengths(restLens);
    Vector3D minusPerturbedEnergies(e.energyStretch(), e.energyBend(), e.energyTwist());

    Vector3D centered_diff_energies = (plusPerturbedEnergies - minusPerturbedEnergies) / 2;

    std::cout << "Centered diff energies: " << centered_diff_energies.transpose() << std::endl;
    std::cout << "Analytic diff energies: " << analytic_diff_energies.transpose() << std::endl;
}

void fdMassTest(ElasticRod &r, Real pos_epsilon, Real theta_epsilon) {
    const size_t nv = r.numVertices(), ne = r.numEdges();
    std::vector<Vector3D> centerlinePerturbation;
    std::vector<Real>     thetaPerturbation, restLenPerturbation;
    generatePerturbations(nv, pos_epsilon, theta_epsilon, 0, centerlinePerturbation, thetaPerturbation, restLenPerturbation);
    auto dofPerturbation = getDoFPerturbation(centerlinePerturbation, thetaPerturbation);

    r.updateSourceFrame();
    auto M = r.massMatrix();
    Real kineticEnergy = 0.5 * dofPerturbation.dot(M.apply(dofPerturbation));
    std::cout << "Kinetic energy from mass matrix: " << kineticEnergy << std::endl;

    auto points = r.deformedPoints();
    auto thetas = r.thetas();
    for (size_t i = 0; i < nv; ++i) points[i] += centerlinePerturbation[i];
    for (size_t j = 0; j < ne; ++j) thetas[j] += thetaPerturbation[j];
    ElasticRod rPlus = r;
    rPlus.setDeformedConfiguration(points, thetas);
    for (size_t i = 0; i < nv; ++i) points[i] -= 2 * centerlinePerturbation[i];
    for (size_t j = 0; j < ne; ++j) thetas[j] -= 2 * thetaPerturbation[j];
    ElasticRod rMinus = r;
    rMinus.setDeformedConfiguration(points, thetas);

    // Finite difference kinetic energy
    Real fdKineticEnergy = 0;
    constexpr size_t K = 2;
    constexpr size_t Deg = 2;
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
            }, restLen);
        }
    }

    fdKineticEnergy *= 0.5;
    std::cout << "Kinetic energy from finite diff: " << fdKineticEnergy << std::endl;
}

void fdLinfVelocityTest(ElasticRod &r, Real pos_epsilon, Real theta_epsilon) {
    const size_t nv = r.numVertices(), ne = r.numEdges();
    std::vector<Vector3D> centerlinePerturbation;
    std::vector<Real>     thetaPerturbation, restLenPerturbation;
    generatePerturbations(nv, pos_epsilon, theta_epsilon, 0, centerlinePerturbation, thetaPerturbation, restLenPerturbation);
    auto dofPerturbation = getDoFPerturbation(centerlinePerturbation, thetaPerturbation);

    r.updateSourceFrame();
    std::cout << "Analytic approx     max velocity: " << r.approxLinfVelocity(dofPerturbation) << std::endl;

    auto points = r.deformedPoints();
    auto thetas = r.thetas();
    for (size_t i = 0; i < nv; ++i) points[i] += centerlinePerturbation[i];
    for (size_t j = 0; j < ne; ++j) thetas[j] += thetaPerturbation[j];
    ElasticRod rPlus = r;
    rPlus.setDeformedConfiguration(points, thetas);
    std::vector<MeshIO::IOVertex > verticesPlus, verticesMinus;
    std::vector<MeshIO::IOElement> quads;
    rPlus.visualizationGeometry(verticesPlus, quads);

    for (size_t i = 0; i < nv; ++i) points[i] -= 2 * centerlinePerturbation[i];
    for (size_t j = 0; j < ne; ++j) thetas[j] -= 2 * thetaPerturbation[j];
    ElasticRod rMinus = r;
    rMinus.setDeformedConfiguration(points, thetas);
    rMinus.visualizationGeometry(verticesMinus, quads);

    Real fdMaxVelocity = 0;
    for (size_t i = 0; i < verticesPlus.size(); ++i)
        fdMaxVelocity = std::max(fdMaxVelocity, 0.5 * (verticesPlus[i].point - verticesMinus[i].point).norm());
    std::cout << "Finite difference max velocity:" << fdMaxVelocity << std::endl;
}

void testRestLenLaplacian(ElasticRod &r) {
    std::cout << std::endl;
    std::cout << "Rest length Laplacian energy:   " << r.restLengthLaplacianEnergy() << std::endl;
    
    auto g = r.restLengthLaplacianGradEnergy();
    Eigen::VectorXd rl = Eigen::Map<const Eigen::VectorXd>(r.restLengths().data(), r.numEdges());
    std::cout << "Laplacian energy from gradient: " << 0.5 * rl.dot(g) << std::endl;
    
    std::cout << "Rest length Laplacian gradient: " << g.transpose() << std::endl;
    std::cout << "         Gradient from hessian: " << r.restLengthLaplacianHessEnergy().apply(rl).transpose() << std::endl;
}

int main(int argc, const char * argv[]) {

    std::cout.precision(19);

    std::vector<Point3D> config_1{
            Point3D(0, 0, 0),
            Point3D(1, 0, 0),
            Point3D(2.1, 0.0, 0),
            Point3D(3, 0.0, 0)
    };

    std::vector<Point3D> config_2{
            Point3D(0, 0, 0),
            Point3D(1, 0, 0),
            Point3D(1 + 0.6 * sqrt(0.5),  0.6 * sqrt(0.5), 0),
            Point3D(2 + 0.6 * sqrt(0.5),  0.6 * sqrt(0.5), 0)
    };

    ElasticRod e(config_1);
    // std::cout << e.numVertices() << " vertices, " << e.numEdges() << " edges." << std::endl;
    e.setDeformedConfiguration(config_2, std::vector<Real>{0, M_PI / 6.0, M_PI / 6.0});
    // e.setDeformedConfiguration(config_2, std::vector<Real>{0, 0});

    RodMaterial mat;
    Real a = 0.1;
    Real b = 0.05;
    mat.set("ellipse", 200, 0.3, { a, b }, RodMaterial::StiffAxis::D1, true /* keep cross-section mesh to permit integration in fdMassTest */);
    // mat.setEllipse(200, 0.3, a, b);
    // mat.set("+", 200, 0.3, { 0.05, 0.05, 0.001, 0.001 }, true /* keep cross-section mesh to permit integration in fdMassTest */);
    e.setMaterial(mat);

    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> quads;
    e.visualizationGeometry(vertices, quads);

    MeshIO::save("geometry.msh", vertices, quads, MeshIO::FMT_GUESS, MeshIO::MESH_QUAD);

    e.writeDebugData("debug.msh");

    auto pts = e.deformedPoints();
    auto ths = e.thetas();
    auto rls = e.restLengths();
    std::vector<Vector3D> pperturb;
    std::vector<Real>     tperturb, rlperturb;
    generatePerturbations(config_2.size(), 0.3, M_PI / 4, 0.1, pperturb, tperturb, rlperturb);
    for (size_t i = 0; i < pts.size(); ++i) pts[i] += pperturb[i];
    for (size_t j = 0; j < ths.size(); ++j) ths[j] += tperturb[j];
    for (size_t j = 0; j < ths.size(); ++j) rls[j] += rlperturb[j];
    e.setDeformedConfiguration(pts, ths);
    e.setRestLengths(rls);

    e.setBendingEnergyType(ElasticRod::BendingEnergyType::Bergou2008);
    fdGradientTest(e, 1e-6, 1e-6, 1e-6);

    pts = e.deformedPoints();
    ths = e.thetas();
    generatePerturbations(config_2.size(), 0.1, M_PI / 4, 0.0, pperturb, tperturb, rls);
    for (size_t i = 0; i < pts.size(); ++i) pts[i] += pperturb[i];
    for (size_t j = 0; j < ths.size(); ++j) ths[j] += tperturb[j];
    e.setDeformedConfiguration(pts, ths);
    e.updateSourceFrame();

    // std::cout << "Testing perturbed configuration" << std::endl;
    // fdGradientTest(e, 1e-5, 1e-5);

    // Try to minimize twisting energy wrt theta.
    auto L = e.hessThetaEnergyTwist();
    auto g = e.gradEnergyTwist();

    std::cout << "Pre twist min grad:" << std::endl;
    std::cout << g.transpose() << std::endl;

    std::vector<Real> rhs(ths.size());
    for (size_t j = 0; j < ths.size(); ++j)
        rhs[j] = -g.gradTheta(j);
    L.fixVariable(0, 0);
    L.fixVariable(ths.size() - 1, 0);
    auto thetaStep = L.solve(rhs);

    for (size_t j = 0; j < ths.size(); ++j)
        ths[j] += thetaStep[j];
    e.pushDeformedConfiguration(pts, ths);

    auto g2 = e.gradEnergyTwist();
    std::cout << "post twist min grad:" << std::endl;
    std::cout << g2.transpose() << std::endl;

#if 0
    {
        SPSDSystem<Real> Lsparse(triplet_matrix_from_tridiagonal(L));
        std::vector<size_t> fixedVars      = {  0, ths.size() - 1};
        std::vector<Real>   fixedVarValues = {0.0, 0.0};
        fixedVars.pop_back(), fixedVarValues.pop_back();
        Lsparse.fixVariables(fixedVars, fixedVarValues);
        auto thetaStep2 = Lsparse.solve(rhs);
        std::cout << "Theta step diffs:" << std::endl;
        for (size_t i = 0; i < thetaStep2.size(); ++i) {
            Real diff = thetaStep[i] - thetaStep2[i];
            std::cout << '\t' << diff << std::endl;
        }
    }
#endif
    e.popDeformedConfiguration();

    fdHessianTest(e, 1e-6, 1e-6, 1e-6);

    fdMassTest(e, 1e-5, 1e-5);
    fdLinfVelocityTest(e, 1e-5, 1e-5);

    // Test Hessian matvec
    std::cout << std::endl;
    {
        std::vector<Vector3D> centerlinePerturbation;
        std::vector<Real>     thetaPerturbation, restLenPerturbation;
        generatePerturbations(e.numVertices(), 1e-5, 1e-5, 1e-5, centerlinePerturbation, thetaPerturbation, restLenPerturbation);
        auto dofPerturbation = getDoFPerturbation(centerlinePerturbation, thetaPerturbation, restLenPerturbation);

        auto Hv = dofPerturbation;
        Hv.setZero();
        e.applyHessEnergy(dofPerturbation, Hv, true);
        auto HvMatrixImpl = e.hessian(ElasticRod::EnergyType::Full, true).apply(dofPerturbation);
        std::cout << "Hessian matvec rel error: " << (Hv - HvMatrixImpl).norm() / HvMatrixImpl.norm() << std::endl;
    }
    return 0;

    // Test autodiff
    std::cout << std::endl;
    {
        ElasticRod_T<ADReal> rdiff(e);
        rdiff.updateSourceFrame();
        std::cout << std::endl;
        std::cout << e.energy() << std::endl;
        std::cout << rdiff.energy() << std::endl;
        std::cout << e.gradEnergy().norm() << std::endl;
        std::cout << rdiff.gradEnergy().norm() << std::endl;

        std::vector<Vector3D> centerlinePerturbation;
        std::vector<Real>     thetaPerturbation, restLenPerturbation;
        generatePerturbations(e.numVertices(), 1e-5, 1e-5, 1e-5, centerlinePerturbation, thetaPerturbation, restLenPerturbation);
        auto dofPerturbation = getDoFPerturbation(centerlinePerturbation, thetaPerturbation, restLenPerturbation);
        auto ad_dofs = rdiff.getExtendedDoFs();
        assert(ad_dofs.size() == dofPerturbation.size());
        for (int i = 0; i < ad_dofs.size(); ++i)
            ad_dofs[i].derivatives() << dofPerturbation(i);

        // Test autodiff gradient
        rdiff.setExtendedDoFs(ad_dofs);
        std::cout << "AutoDiff energy full diff: " << rdiff.energy().derivatives() << std::endl;
        std::cout << "Analytic energy full diff: " << rdiff.gradEnergy(false, true).dot(dofPerturbation) << std::endl;
        std::cout << "AutoDiff energy bend diff: " << rdiff.energyBend().derivatives() << std::endl;
        std::cout << "Analytic energy bend diff: " << rdiff.gradEnergyBend(false, true).dot(dofPerturbation) << std::endl;
        std::cout << "AutoDiff energy twst diff: " << rdiff.energyTwist().derivatives() << std::endl;
        std::cout << "Analytic energy twst diff: " << rdiff.gradEnergyTwist(false, true).dot(dofPerturbation) << std::endl;

        // Test autodiff hessian
        auto H = rdiff.hessianSparsityPattern(true);
        rdiff.hessian(H, ElasticRod_T<ADReal>::EnergyType::Full, true);

        auto g = rdiff.gradEnergy(false, true);
        std::cout << "AutoDiff grad diff:";
        for (int i = 0; i < g.size(); ++i)
            std::cout << " " << g[i].derivatives();
        std::cout << std::endl;
        std::cout << "Analytic grad diff: " << H.apply(Eigen::Matrix<ADReal, Eigen::Dynamic, 1>(dofPerturbation)).transpose() << std::endl;

        Real eps = 0.1; // multiplier for an already small perturbation...
        if (argc > 1) eps = std::stod(argv[1]);
        // auto fdtest_quantity = [&](auto f, const std::string &name) {
        //     rdiff.setExtendedDoFs(ad_dofs);
        //     auto adQuantity = f();
        //     Eigen::VectorXd adDerivative(adQuantity.size());
        //     for (int i = 0; i < adQuantity.size(); ++i)
        //         adDerivative[i] = adQuantity[i].derivatives()[0];

        //     rdiff.setExtendedDoFs(ad_dofs + eps * dofPerturbation);
        //     auto plus = f();
        //     rdiff.setExtendedDoFs(ad_dofs - eps * dofPerturbation);
        //     auto minus = f();
        //     std::cout << "FD " << name << ": " << ((plus - minus) / (2 * eps)).transpose() << std::endl;
        //     std::cout << "AD " << name << ": " << adDerivative.transpose() << std::endl;
        // };

        auto Hplus = H, Hminus = H;
        Hplus.setZero(), Hminus.setZero();

        rdiff.setExtendedDoFs(ad_dofs + eps * dofPerturbation);
        auto gplus = rdiff.gradEnergy(false, true);
        // auto refTwistPlus = rdiff.deformedConfiguration().referenceTwist[1];
        rdiff.hessian(Hplus, ElasticRod_T<ADReal>::EnergyType::Full, true);

        rdiff.setExtendedDoFs(ad_dofs - eps * dofPerturbation);
        auto gminus = rdiff.gradEnergy(false, true);
        // auto refTwistMinus = rdiff.deformedConfiguration().referenceTwist[1];
        rdiff.hessian(Hminus, ElasticRod_T<ADReal>::EnergyType::Full, true);

        // Test autodiff of hessian
        Hplus.addWithIdenticalSparsity(Hminus, -1.0);
        Hplus.scale(1.0 / (2 * eps));

        auto H_ad = H;
        for (auto &x : H_ad.Ax) { x.value() = x.derivatives()[0]; std::cout << " " << x; }
        std::cout << std::endl;
        for (auto &x : Hplus.Ax) std::cout << " " << x;
        std::cout << std::endl;

        auto Hdiff_fd = Eigen::Map<Eigen::Matrix<ADReal, Eigen::Dynamic, 1>>(Hplus.Ax.data(), Hplus.Ax.size());
        auto Hdiff_ad = Eigen::Map<Eigen::Matrix<ADReal, Eigen::Dynamic, 1>>(H_ad.Ax.data(), H_ad.Ax.size());
        
        std::cout << "FD vs AD Hessian derivative rel error: " << (Hdiff_ad - Hdiff_fd).norm() / Hdiff_ad.norm() << std::endl;
#if 0
        std::cout << "fd grad diff: " << ((gplus - gminus) / (2 * eps)).transpose() << std::endl;

        fdtest_quantity([&]() { return rdiff.deformedConfiguration().referenceDirectors[0].d1; }, "referenceDirectors[0].d1");
        fdtest_quantity([&]() { return rdiff.deformedConfiguration().referenceDirectors[1].d1; }, "referenceDirectors[1].d1");
        fdtest_quantity([&]() { return rdiff.deformedConfiguration().referenceDirectors[0].d2; }, "referenceDirectors[0].d2");
        fdtest_quantity([&]() { return rdiff.deformedConfiguration().referenceDirectors[1].d2; }, "referenceDirectors[1].d2");
        fdtest_quantity([&]() { return rdiff.deformedConfiguration().materialFrame[0].d1; }, "materialFrame[0].d1");
        fdtest_quantity([&]() { return rdiff.deformedConfiguration().materialFrame[1].d1; }, "materialFrame[1].d1");
        fdtest_quantity([&]() { return rdiff.deformedConfiguration().materialFrame[0].d2; }, "materialFrame[0].d2");
        fdtest_quantity([&]() { return rdiff.deformedConfiguration().materialFrame[1].d2; }, "materialFrame[1].d2");
        fdtest_quantity([&]() { return rdiff.deformedConfiguration().sourceMaterialFrame[0].d1; }, "sourceMaterialFrame[0].d1");
        fdtest_quantity([&]() { return rdiff.deformedConfiguration().sourceMaterialFrame[1].d1; }, "sourceMaterialFrame[1].d1");
        fdtest_quantity([&]() { return rdiff.deformedConfiguration().sourceMaterialFrame[0].d2; }, "sourceMaterialFrame[0].d2");
        fdtest_quantity([&]() { return rdiff.deformedConfiguration().sourceMaterialFrame[1].d2; }, "sourceMaterialFrame[1].d2");
        fdtest_quantity([&]() { return rdiff.deformedConfiguration().tangent[0]; }, "tangent[0]");
#endif

        BENCHMARK_RESET();
        BENCHMARK_START_TIMER("Autodiff Hessian Eval");
        for (size_t i = 0; i < 1000; ++i) {
            auto Hsp = rdiff.hessianSparsityPattern(true);
            rdiff.hessian(Hsp, ElasticRod_T<ADReal>::EnergyType::Full, true);
        }
        BENCHMARK_STOP_TIMER("Autodiff Hessian Eval");
        BENCHMARK_START_TIMER("Plain Hessian Eval");
        for (size_t i = 0; i < 1000; ++i) {
            auto Hsp = e.hessianSparsityPattern(true);
            e.hessian(Hsp, ElasticRod::EnergyType::Full, true);
        }
        BENCHMARK_STOP_TIMER("Plain Hessian Eval");

        BENCHMARK_REPORT_NO_MESSAGES();
    }

    return 0;

    // Test that the diagonal lumped mass matrix actually holds the
    // row sums of the full mass matrix.
    {
        auto M = e.massMatrix();
        Eigen::VectorXd rowSums = Eigen::VectorXd::Zero(M.m);
        for (const auto &t : M.nz) {
            rowSums[t.i] += t.v;
            if (t.j != t.i) rowSums[t.j] += t.v;
        }
        auto lumpedM = e.lumpedMassMatrix();
        std::cout << lumpedM.transpose() << std::endl;
        std::cout << rowSums.transpose() << std::endl;
        std::cout << ((lumpedM - rowSums).array() / rowSums.array()).transpose() << std::endl;
    }

    testRestLenLaplacian(e);

    std::cout << std::endl;
    std::cout << std::endl;

    // Try running Newton's method...
    const size_t niter = 500;
    const size_t nbacktrack_iter = 10;
    const size_t ndofs = e.numDoF();

    const std::vector<size_t> fixedVars = {  0,  1,  2,
                                             9,
                                            10,
                                            11,
                                            // 12,
                                            14 };

#if HAS_KNITRO
    knitro_compute_equilibrium(e, niter, fixedVars);
#endif

    return 0;

    NewtonOptimizerOptions opts;
    opts.niter = niter;
    opts.verbose = true;
    compute_equilibrium(e, opts, fixedVars);

    BENCHMARK_REPORT_NO_MESSAGES();

    return 0;

    std::cout << 0
              << '\t' << e.energy()
              << '\t' << g.norm()
              << std::endl;
    vertices.clear(), quads.clear();
    e.visualizationGeometry(vertices, quads);
    MeshIO::save("config_" + std::to_string(0) + ".msh", vertices, quads, MeshIO::FMT_GUESS, MeshIO::MESH_QUAD);

    for (size_t it = 1; it <= niter; ++it) {
        e.updateSourceFrame();
        auto g = e.gradEnergy(true);
        SPSDSystem<Real> H(e.hessian());
        {
            std::vector<Real> fixedVarValues(fixedVars.size(), 0.0);
            H.fixVariables(fixedVars, fixedVarValues);
        }
        Eigen::VectorXd step;
        H.solve(-g, step);

        Real directionalDerivative = step.dot(g);
        // If the Newton search direction is not a descent direction
        // (non-positive Hessian), switch to steepest descent.
        if (directionalDerivative >= 0) {
            std::cout << "Using steepest descent" << std::endl;
            step = -g;
            directionalDerivative = -g.norm();
        }

        // Simple backtracking line search to ensure a sufficient decrease
        Real currEnergy = e.energy();
        Real alpha = 1.0;
        auto dofs = e.getDoFs();
        auto steppedDoFs = dofs;
        const Real c_1 = 1e-5;
        size_t bit;
        for (bit = 0; bit < nbacktrack_iter; ++bit) {
            for (size_t i = 0; i < ndofs; ++i)
                steppedDoFs[i] = dofs[i] + alpha * step[i];
            e.setDoFs(steppedDoFs);
            Real steppedEnergy = e.energy();

            if  (steppedEnergy - currEnergy <= c_1 * alpha * directionalDerivative)
                break;
            alpha *= 0.5;
        }
        if (bit == nbacktrack_iter) break;

        e.setDoFs(steppedDoFs);

        vertices.clear(), quads.clear();
        e.visualizationGeometry(vertices, quads);
        MeshIO::save("config_" + std::to_string(it) + ".msh", vertices, quads, MeshIO::FMT_GUESS, MeshIO::MESH_QUAD);

        std::cout << it
                  << '\t' << e.energy()
                  << '\t' << e.energyStretch()
                  << '\t' << e.energyBend()
                  << '\t' << e.energyTwist()
                  << '\t' << g.norm()
                  << '\t' << alpha
                  << std::endl;
    }

    vertices.clear(), quads.clear();
    e.visualizationGeometry(vertices, quads);
    MeshIO::save("equilibrium_config.msh", vertices, quads, MeshIO::FMT_GUESS, MeshIO::MESH_QUAD);
    e.writeDebugData("forces.msh");

    return 0;
}
