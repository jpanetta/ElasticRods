#include <iostream>
#include <fstream>
#include "../RodLinkage.hh"
#include "../LinkageOptimization.hh"
#include "../LinkageTerminalEdgeSensitivity.hh"
#include "../cg_solver.hh"
#include "../open_linkage.hh"
#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/GlobalBenchmark.hh>

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

int main(int argc, const char * argv[]) {
    if ((argc != 3) && (argc != 4)) {
        std::cout << "usage: " << argv[0] << " linkage.msh cross_section.json [fd_eps]" << std::endl;
        exit(-1);
    }
    const std::string &linkageGraph = argv[1];

    Real fd_eps = 1e-8;
    if (argc >= 4) { fd_eps = std::stod(argv[3]); }

    std::cout.precision(19);

    RodMaterial mat(*CrossSection::load(argv[2]), RodMaterial::StiffAxis::D1, true /* keep the cross-section mesh so that we can integrate over it in the mass matrix test */);

    RodLinkage l2d(linkageGraph);
    RodLinkage l3d(l2d);

    l2d.setMaterial(mat);
    l3d.setMaterial(mat);

    NewtonOptimizerOptions eopts;
    eopts.gradTol = 1e-14;
    eopts.verbose = 1;
    eopts.niter = 50;

    // open_linkage(l3d, 0.9488, eopts, l3d.centralJoint());
    open_linkage(l3d, 1.5, eopts, l3d.centralJoint());

    LinkageOptimization lopt(l2d, l3d, eopts, std::make_unique<LOMinAngleConstraint>());
    lopt.target_surface_fitter.setTargetJointPosVsTargetSurfaceTradeoff(l3d, 1.0);
    // LinkageOptimization lopt(l2d, l3d, eopts);
    lopt.gamma = 0.5;
    lopt.beta = 10000.0;
    lopt.prediction_order = 0;

    lopt.getMinAngleConstraint().s = 1e-2;


    // Test smooth minimum delta gradient
// TODO: reenable after we determine if this is what is making the
    {
        auto &flat = lopt.getLinesearchFlatLinkage();
        auto extract_alpha_components = [&](const Eigen::VectorXd &v) {
            Eigen::VectorXd result(flat.numJoints());
            for (size_t ji = 0; ji < flat.numJoints(); ++ji)
                result[ji] = v[flat.dofOffsetForJoint(ji) + 6];
            return result;
        };
        auto &alpha_min = lopt.getMinAngleConstraint();
        auto dofPerturbation = getDofPerturbation(flat.numDoF());
        auto centerDoFs = flat.getDoFs();
        flat.setDoFs(centerDoFs + fd_eps * dofPerturbation);
        auto grad_alphamin_plus = extract_alpha_components(alpha_min.grad(flat));
        flat.setDoFs(centerDoFs - fd_eps * dofPerturbation);
        auto grad_alphamin_minus = extract_alpha_components(alpha_min.grad(flat));
        flat.setDoFs(centerDoFs);
        auto delta_grad_alphamin_analytic = extract_alpha_components(alpha_min.delta_grad(flat, dofPerturbation));
        auto delta_grad_alphamin_fd       = ((grad_alphamin_plus - grad_alphamin_minus) / (2 * fd_eps)).eval();
        std::cout << "fd       delta grad alphamin: " << delta_grad_alphamin_fd      .head(5).transpose() << " ..." << std::endl;
        std::cout << "analytic delta grad alphamin: " << delta_grad_alphamin_analytic.head(5).transpose() << " ..." << std::endl;
        std::cout << "relerror delta grad alphamin: " << (delta_grad_alphamin_analytic - delta_grad_alphamin_fd).norm() / delta_grad_alphamin_fd.norm() << std::endl;
    }

    // lopt.constructTargetSurface();
    std::cout << "Constructed target surface" << std::endl;

#if 1
    std::cout << "2D average joint angle: " << l2d.getAverageJointAngle() << std::endl;
    std::cout << "2D min     joint angle: " << l2d.getMinJointAngle() << std::endl;
    std::cout << "2D smin    joint angle: " << lopt.getMinAngleConstraint().alpha_min(l2d) << std::endl;
#endif

    // Finite difference gradient tests
    auto params = l2d.getDesignParameters();

    auto grad_J                = lopt.gradp_J(params);
    auto grad_target           = lopt.gradp_J_target();
    auto grad_angle_constraint = lopt.gradp_angle_constraint(params);
    auto grad_c                = lopt.gradp_c(params);
    auto param_perturbation = getDofPerturbation(lopt.numParams());
    {
        Real J_plus         = lopt.J               (params + fd_eps * param_perturbation),
             J_target_plus  = lopt.J_target        (params + fd_eps * param_perturbation),
             anglec_plus    = lopt.angle_constraint(params + fd_eps * param_perturbation),
             c_plus         = lopt.c               (params + fd_eps * param_perturbation);

        Real J_minus        = lopt.J               (params - fd_eps * param_perturbation),
             J_target_minus = lopt.J_target        (params - fd_eps * param_perturbation),
             anglec_minus   = lopt.angle_constraint(params - fd_eps * param_perturbation),
             c_minus        = lopt.c               (params - fd_eps * param_perturbation);

        std::cout << "fd       delta J: " << (J_plus - J_minus) / (2 * fd_eps) << std::endl;
        std::cout << "analytic delta J: " << grad_J.dot(param_perturbation) << std::endl;

        std::cout << "fd       delta J_target: " << (J_target_plus - J_target_minus) / (2 * fd_eps) << std::endl;
        std::cout << "analytic delta J_target: " << grad_target.dot(param_perturbation) << std::endl;

        std::cout << "fd       delta angle constraint: " << (anglec_plus - anglec_minus) / (2 * fd_eps) << std::endl;
        std::cout << "analytic delta angle constraint: " << grad_angle_constraint.dot(param_perturbation) << std::endl;

        std::cout << "fd       delta c: " << (c_plus - c_minus) / (2 * fd_eps) << std::endl;
        std::cout << "analytic delta c: " << grad_c.dot(param_perturbation) << std::endl;
    }

    std::cout << "Evaluate plus linkage" << std::endl;
    auto grad_plus    = lopt.gradp_J               (params + fd_eps * param_perturbation);
    auto grad_ac_plus = lopt.gradp_angle_constraint(params + fd_eps * param_perturbation);
    auto grad_c_plus  = lopt.gradp_c               (params + fd_eps * param_perturbation);
    auto w_plus   = lopt.get_w_x();
    auto s_plus   = lopt.get_s_x();
    auto y_plus   = lopt.get_y();
    auto x_plus   = lopt.getLinesearchDeployedLinkage().getDoFs();
    auto x2d_plus = lopt.getLinesearchFlatLinkage().getDoFs();

    std::cout << "Evaluate minus linkage" << std::endl;
    auto grad_minus    = lopt.gradp_J               (params - fd_eps * param_perturbation);
    auto grad_ac_minus = lopt.gradp_angle_constraint(params - fd_eps * param_perturbation);
    auto grad_c_minus  = lopt.gradp_c               (params - fd_eps * param_perturbation);

    auto w_minus   = lopt.get_w_x();
    auto s_minus   = lopt.get_s_x();
    auto y_minus   = lopt.get_y();
    auto x_minus   = lopt.getLinesearchDeployedLinkage().getDoFs();
    auto x2d_minus = lopt.getLinesearchFlatLinkage().getDoFs();

    std::cout << "running apply_hess_J..." << std::endl;
    auto delta_gradp_J                = lopt.apply_hess_J               (params, param_perturbation);
    auto delta_gradp_angle_constraint = lopt.apply_hess_angle_constraint(params, param_perturbation);
    auto delta_gradp_c                = lopt.apply_hess_c               (params, param_perturbation);

    auto fd_report = [&](const std::string &name, auto fd_result, auto an_result) {
        std::cout << std::endl;
        std::cout << "fd       " << name << ": " << fd_result.segment(0, 5).transpose() << "..." << std::endl;
        std::cout << "analytic " << name << ": " << an_result.segment(0, 5).transpose() << "..." << std::endl;
        std::cout << "fd       " << name << " norm: " << fd_result.norm() << std::endl;
        std::cout << "analytic " << name << " norm: " << an_result.norm() << std::endl;
        std::cout << name << " rel error: " << (fd_result - an_result).norm() / an_result.norm() << std::endl;

        int idx;
        Real err = (fd_result - an_result).cwiseAbs().maxCoeff(&idx);
        std::cout << "greatest abs error " << err << " at entry " << idx << ": "
                  << fd_result[idx] << " vs " << an_result[idx] << std::endl;
        std::cout << fd_result.segment(idx - 5, 10).transpose() << std::endl;
        std::cout << an_result.segment(idx - 5, 10).transpose() << std::endl;
    };

#if 1
    // Warning: if updateRotationParametrizations() is called, the rotation
    // components of delta x will appear incorrect. But this is just because
    // the rotation variables are being reset to zero before the finite
    // difference evaluations; the full Hessian of J wrt p will still be
    // correct with this discrepancy.
    // Nevertheless, it may be useful for debugging to disable the calls
    // updateRotationParametrizations() (in both LinkageOptimization::m_forceEquilibriumUpdate
    // and EquilibriumProblem::m_iterationCallback) and enable the following test:
    fd_report("delta x", ((x_plus - x_minus) / (2 * fd_eps)).eval(), lopt.get_delta_x3d());
    fd_report("delta w", ((w_plus - w_minus) / (2 * fd_eps)).eval(), lopt.get_delta_w_x());
    fd_report("delta s", ((s_plus - s_minus) / (2 * fd_eps)).eval(), lopt.get_delta_s_x());
    fd_report("delta y", ((y_plus - y_minus) / (2 * fd_eps)).eval(), lopt.get_delta_y());

    fd_report("delta x_2d", ((x2d_plus - x2d_minus) / (2 * fd_eps)).eval(), lopt.get_delta_x2d());

    fd_report("delta gradp J", ((grad_plus - grad_minus) / (2 * fd_eps)).eval(), delta_gradp_J);

    fd_report("delta gradp angle constraint", ((grad_ac_plus - grad_ac_minus) / (2 * fd_eps)).eval(), delta_gradp_angle_constraint);
    fd_report("delta gradp c", ((grad_c_plus - grad_c_minus) / (2 * fd_eps)).eval(), delta_gradp_c);

    BENCHMARK_REPORT_NO_MESSAGES();
#endif

    BENCHMARK_RESET();
    BENCHMARK_START_TIMER_SECTION("CG iterations");

    // lopt.getLinesearchDeployedLinkage().updateRotationParametrizations();
    // lopt.getLinesearchFlatLinkage().updateRotationParametrizations();
    std::cout << "running CG iterations for Newton step:" << std::endl;
    using VecX = Eigen::VectorXd;
    VecX step = VecX::Zero(params.size());
    for (size_t i = 0; i < 10; ++i) {
        // Benchmark Hessian matvec using the cg solver with a disabled termination check.
        cg_solver<true>([&](const VecX &v) { return lopt.apply_hess_J(params, v); },
                        lopt.gradp_J(params),
                        step,
                        [&](size_t /* k */, const VecX &/* r */) { /* std::cout << "residual " << k << " norm: " << r.norm() << ", " << " step norm: " << step.norm() << std::endl; */ },
                        100, 0.0);
    }

    BENCHMARK_STOP_TIMER_SECTION("CG iterations");

    BENCHMARK_REPORT_NO_MESSAGES();
#if 0
    cg_solver([&](const VecX &v) { return lopt.apply_hess_J(params, lopt.apply_hess_J(params, v)); },
              lopt.apply_hess_J(params, lopt.gradp_J(params)),
              step,
              [&](size_t k, const VecX &r) { std::cout << "residual " << k << " norm: " << r.norm() << ", " << " step norm: " << step.norm() << ", true residual norm: " << (lopt.gradp_J(params) - lopt.apply_hess_J(params, step)).norm() << std::endl; },
              50, 0.0);
#endif

#if 0
    auto perturbation_dir = getDofPerturbation(l2d.numDesignParams());

    l3d.updateSourceFrame();
    l3d.updateRotationParametrizations();
    // l3d.setExtendedDoFsPSRL(l3d.getExtendedDoFsPSRL() + getDofPerturbation(l3d.numExtendedDoFPSRL(), 1e-300));
    // l3d.setExtendedDoFsPSRL(l3d.getExtendedDoFsPSRL() + getDofPerturbation(l3d.numExtendedDoFPSRL(), 1e-4));
    // l3d.updateSourceFrame();
    l3d.updateRotationParametrizations();
    std::cout << "Constructing autodiff linkage" << std::endl << std::endl;
    RodLinkage_T<ADReal> l3d_diff(l3d);

    srand(1);
    auto perturb = getDofPerturbation(l3d.numExtendedDoFPSRL());
    auto ad_dofs = l3d_diff.getExtendedDoFsPSRL();
    for (int i = 0; i < perturb.size(); ++i)
        ad_dofs[i].derivatives()[0] = perturb(i);
    std::cout << "Injecting derivatives" << std::endl << std::endl;
    l3d_diff.setExtendedDoFsPSRL(ad_dofs);

    using EType = RodLinkage_T<ADReal>::EnergyType;
    std::cout << l3d.energy() << std::endl;
    std::cout << l3d_diff.energy() << std::endl;
    std::cout << l3d.gradient(false, RodLinkage::EnergyType::Full, true).norm() << std::endl;
    std::cout << l3d_diff.gradient(false, EType::Full, true).norm() << std::endl;
    std::cout << std::endl;

#if 0
    auto reportDerivatives = [&](const std::string &name, const auto &vec) {
        std::cout << "AutoDiff " << name << ":";
        for (int i = 0; i < vec.size(); ++i)
            std::cout << "\t" << vec[i].derivatives();
        std::cout << std::endl;
    };

    reportDerivatives("joint e_A", l3d_diff.joint(0).e_A());
    reportDerivatives("joint e_B", l3d_diff.joint(0).e_B());
    reportDerivatives("joint omega", l3d_diff.joint(0).omega());
    reportDerivatives("joint normal", l3d_diff.joint(0).normal());
    reportDerivatives("joint s_t_A", l3d_diff.joint(0).source_t_A());
    reportDerivatives("joint s_t_B", l3d_diff.joint(0).source_t_B());
    reportDerivatives("joint source normal", l3d_diff.joint(0).source_normal());
    reportDerivatives("joint pos", l3d_diff.joint(0).pos());
    std::cout << "joint len a:\t" << l3d_diff.joint(0).len_A().derivatives() << std::endl;
#endif

    std::cout << "AutoDiff energy full diff: " << l3d_diff.energy().derivatives() << std::endl;
    std::cout << "Analytic energy full diff: " << l3d_diff.gradientPerSegmentRestlen(false, EType::Full).dot(perturb) << std::endl;
    std::cout << "AutoDiff energy bend diff: " << l3d_diff.energyBend().derivatives() << std::endl;
    std::cout << "Analytic energy bend diff: " << l3d_diff.gradientPerSegmentRestlen(false, EType::Bend).dot(perturb) << std::endl;
    std::cout << "AutoDiff energy twst diff: " << l3d_diff.energyTwist().derivatives() << std::endl;
    std::cout << "Analytic energy twst diff: " << l3d_diff.gradientPerSegmentRestlen(false, EType::Twist).dot(perturb) << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;

    auto H = l3d_diff.hessianPerSegmentRestlenSparsityPattern();
    const auto et = EType::Stretch;
    l3d_diff.hessianPerSegmentRestlen(H, et);
    auto g = l3d_diff.gradientPerSegmentRestlen(false, et);
    for (int i = 0; i < g.size(); ++i) g[i].value() = g[i].derivatives()[0];
    auto g_analytic = H.apply(Eigen::Matrix<ADReal, Eigen::Dynamic, 1>(perturb));
    std::cout << "AutoDiff grad diff rel error: " << (g_analytic - g).norm() / g.norm() << std::endl;
    {
        size_t ji = 0;
        size_t ABoffset = 0;
        size_t si = (ABoffset == 0) ? l3d_diff.joint(ji).segmentsA()[0] : l3d_diff.joint(ji).segmentsB()[0];
        auto sensitivity = l3d_diff.getTerminalEdgeSensitivity(ji, si, false, true);
        Eigen::Matrix<Real, 4, 6> delta_jacobian, analytic_delta_jacobian;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 6; ++c)
                delta_jacobian(r, c) = sensitivity->jacobian(r, c).derivatives()[0];
        std::cout << "autodiff delta_jacobian:" << std::endl;
        std::cout << delta_jacobian << std::endl;
        std::cout << std::endl;

        analytic_delta_jacobian.setZero();
        for (int comp = 0; comp < 4; ++comp)
            for (int var = 0; var < 6; ++var)
                for (int pvar = 0; pvar < 6; ++pvar)
                    analytic_delta_jacobian(comp, var) += perturb[l3d_diff.dofOffsetForJoint(ji) + 3 + pvar] * sensitivity->hessian[comp](var, pvar).value();

        std::cout << "analytic delta_jacobian:" << std::endl;
        std::cout << analytic_delta_jacobian << std::endl;
        std::cout << std::endl;

        std::cout << "difference:" << std::endl;
        std::cout << delta_jacobian - analytic_delta_jacobian << std::endl;
        std::cout << std::endl;

        auto Hplus = H, Hminus = H;

        Hplus.setZero(), Hminus.setZero();
        l3d_diff.setExtendedDoFsPSRL(ad_dofs + fd_eps * perturb);
        l3d_diff.updateSourceFrame();
        l3d_diff.hessianPerSegmentRestlen(Hplus, et);

        auto sensitivity_plus  = l3d_diff.getTerminalEdgeSensitivity(ji, si, true, true);

        l3d_diff.setExtendedDoFsPSRL(ad_dofs - fd_eps * perturb);
        l3d_diff.hessianPerSegmentRestlen(Hminus, et);
        l3d_diff.updateSourceFrame();
        auto sensitivity_minus = l3d_diff.getTerminalEdgeSensitivity(ji, si, true, true);
        std::cout << std::endl;
        const size_t comp = 3;
#if 1
        std::cout << (sensitivity_plus->hessian[comp] - sensitivity_minus->hessian[comp]) / (2 * fd_eps) << std::endl;
        std::cout << std::endl;
        Eigen::Matrix<double, 6, 6> delta_hessian;
        for (int r = 0; r < 6; ++r)
            for (int c = 0; c < 6; ++c)
                delta_hessian(r, c) = sensitivity->hessian[comp](r, c).derivatives()[0];

        std::cout << delta_hessian << std::endl;

        std::cout << std::endl;
        std::cout << (sensitivity_plus->hessian[comp] - sensitivity_minus->hessian[comp]) / (2 * fd_eps) - delta_hessian << std::endl;
        std::cout << std::endl;
#endif
#if 0
        // Note: this comparison shows a discrepancy for the gradient of theta (last row of the jacobian)
        // This is because the analytic/autodiff sensitivity was computed assuming the source frame is
        // held fixed (false was passed for updatedSource), while the finite
        // difference was computed by updating the source frame. Changing either of these calculations
        // to be consistent resolves the discrepancy.
        std::cout << "fd joint jacobian" << std::endl;
        std::cout << (sensitivity_plus->jacobian - sensitivity_minus->jacobian) / (2 * fd_eps) << std::endl;
        std::cout << std::endl;

        std::cout << "autodiff joint jacobian" << std::endl;
        std::cout << delta_jacobian << std::endl;

        std::cout << "error" << std::endl;
        std::cout << std::endl;
        std::cout << (sensitivity_plus->jacobian - sensitivity_minus->jacobian) / (2 * fd_eps) - delta_jacobian << std::endl;
        std::cout << std::endl;
#endif

        auto H_ad = H;
        for (auto &x : H_ad.Ax) x.value() = x.derivatives()[0];

        Hplus.addWithIdenticalSparsity(Hminus, -1.0);
        Hplus.scale(1.0 / (2.0 * fd_eps));
        auto Hdiff_fd = Eigen::Map<Eigen::Matrix<ADReal, Eigen::Dynamic, 1>>(Hplus.Ax.data(), Hplus.Ax.size());
        auto Hdiff_ad = Eigen::Map<Eigen::Matrix<ADReal, Eigen::Dynamic, 1>>(H_ad.Ax.data(), H_ad.Ax.size());

        std::cout << "FD vs AD Hessian derivative rel error: " << (Hdiff_ad - Hdiff_fd).norm() / Hdiff_ad.norm() << std::endl;

        using TI = decltype(H_ad.begin());
        int idx;
        (Hdiff_ad - Hdiff_fd).cwiseAbs().maxCoeff(&idx);
        TI it(H_ad, idx);

        std::cout << "fd value at " << it.get_i() << ", " << it.get_j() << ": " << Hdiff_fd[idx] << std::endl;
        std::cout << "ad value at " << it.get_i() << ", " << it.get_j() << ": " << it.get_val() << std::endl;
    }

    // auto g = l3d_diff.gradient(false);
    // for (int i = 0; i < g.size(); ++i)
    //     std::cout << "\t" << g[i].derivatives()[0];
    // std::cout << std::endl;
#endif
    // Test the 2nd order Taylor expansion of x along a perturbation direction.
    // Note: the updateRotationParametrizations() calls must be disabled for this to work...
    // (otherwise the rotation variables being set to zero will corrupt the finite difference comparison)
    if (lopt.prediction_order == 2) {
        // Eval away from, then back at params will force an equilibrium update, recomputing delta_x and delta_delta_x
        std::cout << "Forcing re-eval" << std::endl;
        lopt.J(params);
        auto x_3d = lopt.getLinesearchDeployedLinkage().getDoFs();
        auto x_2d = lopt.getLinesearchFlatLinkage().getDoFs();
        lopt.commitLinesearchLinkage();

        lopt.J(params + fd_eps * param_perturbation); // side_effect: compute analytic deltas for the parameter perturbation at params
        std::cout << "Getting delta state" << std::endl;
        const auto delta_x3d = lopt.get_delta_x3d();
        const auto delta_x2d = lopt.get_delta_x2d();
        const auto delta_delta_x3d = lopt.get_delta_delta_x3d();
        const auto delta_delta_x2d = lopt.get_delta_delta_x2d();
        const auto x3d_second_order = lopt.get_second_order_x3d();
        const auto x2d_second_order = lopt.get_second_order_x2d();
        Real error_3d = (x3d_second_order - (x_3d + delta_x3d + 0.5 * delta_delta_x3d)).norm(),
             error_2d = (x2d_second_order - (x_2d + delta_x2d + 0.5 * delta_delta_x2d)).norm();
        std::cout << "error_3d: " << error_3d << std::endl;
        std::cout << "error_2d: " << error_2d << std::endl;
        if (error_3d > 1e-14) throw std::runtime_error("second order prediction error: " + std::to_string(error_3d));
        if (error_2d > 1e-14) throw std::runtime_error("second order prediction error: " + std::to_string(error_2d));

        lopt.J(params + fd_eps * param_perturbation);
        auto x_3d_plus = lopt.getLinesearchDeployedLinkage().getDoFs();
        auto x_2d_plus = lopt.getLinesearchFlatLinkage().getDoFs();
        lopt.commitLinesearchLinkage();
        lopt.J(params + 2 * fd_eps * param_perturbation); // side_effect: compute analytic deltas for perturbation (fd_eps * param_perturbation) at (params + fd_eps * param_perturbation)
        auto delta_x_3d_plus = lopt.get_delta_x3d();
        auto delta_x_2d_plus = lopt.get_delta_x2d();

        lopt.J(params - fd_eps * param_perturbation);
        auto x_3d_minus = lopt.getLinesearchDeployedLinkage().getDoFs();
        auto x_2d_minus = lopt.getLinesearchFlatLinkage().getDoFs();
        lopt.commitLinesearchLinkage();
        lopt.J(params); // side_effect: compute analytic deltas for perturbation (fd_eps * param_perturbation) at (params + fd_eps * param_perturbation)
        auto delta_x_3d_minus = lopt.get_delta_x3d();
        auto delta_x_2d_minus = lopt.get_delta_x2d();

        // Note: analytic deltas were computed for a single step distance of fd_eps
        fd_report("delta x3d", ((x_3d_plus - x_3d_minus) / (2 * fd_eps)).eval(), (delta_x3d / fd_eps).eval());
        fd_report("delta x2d", ((x_2d_plus - x_2d_minus) / (2 * fd_eps)).eval(), (delta_x2d / fd_eps).eval());
        fd_report("delta delta x3d", ((delta_x_3d_plus - delta_x_3d_minus) / (2 * fd_eps)).eval(), (delta_delta_x3d / fd_eps).eval());
        fd_report("delta delta x2d", ((delta_x_2d_plus - delta_x_2d_minus) / (2 * fd_eps)).eval(), (delta_delta_x2d / fd_eps).eval());

        fd_report("plus delta x3d",  Eigen::VectorXd::Zero(delta_x_3d_plus.size()), (delta_x_3d_plus / fd_eps).eval());
        fd_report("plus delta x2d",  Eigen::VectorXd::Zero(delta_x_3d_plus.size()), (delta_x_2d_plus / fd_eps).eval());
        fd_report("minus delta x3d", Eigen::VectorXd::Zero(delta_x_3d_plus.size()), (delta_x_3d_minus / fd_eps).eval());
        fd_report("minus delta x2d", Eigen::VectorXd::Zero(delta_x_3d_plus.size()), (delta_x_2d_minus / fd_eps).eval());

        auto x3d_first_order  = x_3d + delta_x3d;
        lopt.J(params + fd_eps * param_perturbation);
        auto x3d_actual = lopt.getLinesearchDeployedLinkage().getDoFs();
        std::cout << "3D 0th order error: " << (x_3d             - x3d_actual).norm() / x3d_actual.norm() << std::endl;
        std::cout << "3D 1st order error: " << (x3d_first_order  - x3d_actual).norm() / x3d_actual.norm() << std::endl;
        std::cout << "3D 2nd order error: " << (x3d_second_order - x3d_actual).norm() / x3d_actual.norm() << std::endl;

        auto x2d_first_order  = x_2d + delta_x2d;
        auto x2d_actual = lopt.getLinesearchFlatLinkage().getDoFs();
        std::cout << "2D 0th order error: " << (x_2d             - x2d_actual).norm() / x2d_actual.norm() << std::endl;
        std::cout << "2D 1st order error: " << (x2d_first_order  - x2d_actual).norm() / x2d_actual.norm() << std::endl;
        std::cout << "2D 2nd order error: " << (x2d_second_order - x2d_actual).norm() / x2d_actual.norm() << std::endl;
    }

    return 0;
}
