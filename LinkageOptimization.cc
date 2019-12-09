#include "LinkageOptimization.hh"

LinkageOptimization::LinkageOptimization(RodLinkage &flat, RodLinkage &deployed, const NewtonOptimizerOptions &eopts, std::unique_ptr<LOMinAngleConstraint> &&minAngleConstraint, int pinJoint, bool allowFlatActuation)
    : m_equilibrium_options(eopts), m_numParams(flat.numDesignParams()),
      m_E0(deployed.energy()), m_l0(BBox<Point3D>(deployed.deformedPoints()).dimensions().norm()),
      m_flat(flat), m_deployed(deployed), m_linesearch_flat(flat), m_linesearch_deployed(deployed),
      m_minAngleConstraint(std::move(minAngleConstraint)),
      m_allowFlatActuation(allowFlatActuation)
{
    std::runtime_error mismatch("Linkage mismatch");
    if (m_numParams != deployed.numDesignParams())                                    throw mismatch;
    if ((deployed.getDesignParameters() - flat.getDesignParameters()).norm() > 1e-16) throw mismatch;
    m_alpha_tgt = deployed.getAverageJointAngle();

    // Unless the user specifies otherwise, use the current deployed linkage joint positions as the target
    target_surface_fitter.joint_pos_tgt = deployed.jointPositions();
    constructTargetSurface();

    // Trade off between fitting to the individual joint targets and the target surface.
    target_surface_fitter.setTargetJointPosVsTargetSurfaceTradeoff(deployed, 0.01);

    // Constrain the position and orientation of the centermost joint to prevent global rigid motion.
    if (pinJoint != -1) {
        m_rm_constrained_joint = pinJoint;
        if (m_rm_constrained_joint >= flat.numJoints()) throw std::runtime_error("Manually specified pinJoint is out of bounds");
    }
    else {
        m_rm_constrained_joint = flat.centralJoint();
    }
    const size_t jdo = flat.dofOffsetForJoint(m_rm_constrained_joint);
    for (size_t i = 0; i < 6; ++i) m_rigidMotionFixedVars.push_back(jdo + i);

    m_flat_optimizer     = get_equilibrium_optimizer(m_linesearch_flat, TARGET_ANGLE_NONE, m_rigidMotionFixedVars);
    m_deployed_optimizer = get_equilibrium_optimizer(m_linesearch_deployed,   m_alpha_tgt, m_rigidMotionFixedVars);

    m_flat_optimizer    ->options = m_equilibrium_options;
    m_deployed_optimizer->options = m_equilibrium_options;

    // Ensure we start at an equilibrium (using the passed equilibrium solver options)
    m_forceEquilibriumUpdate();
    m_updateMinAngleConstraintActuation();

    commitLinesearchLinkage();
}

void LinkageOptimization::m_forceEquilibriumUpdate() {
    // Update the flat/deployed equilibria
    m_equilibriumSolveSuccessful = true;
    try {
        if (m_equilibrium_options.verbose)
            std::cout << "Flat equilibrium solve" << std::endl;
        auto cr_flat = getFlatOptimizer().optimize();

        if (m_equilibrium_options.verbose)
            std::cout << "Deployed equilibrium solve" << std::endl;
        auto cr_deploy = getDeployedOptimizer().optimize();
        // if (!cr_flat.success || !cr_deploy.success) throw std::runtime_error("Equilibrium solve did not converge");
    }
    catch (const std::runtime_error &e) {
        std::cout << "Equilibrium solve failed: " << e.what() << std::endl;
        m_equilibriumSolveSuccessful = false;
        return; // subsequent update_factorizations will fail if we caught a Tau runaway...
    }

    // We will be evaluating the Hessian/using the simplified gradient expressions:
    m_linesearch_flat    .updateSourceFrame();
    m_linesearch_flat    .updateRotationParametrizations();
    m_linesearch_deployed.updateSourceFrame();
    m_linesearch_deployed.updateRotationParametrizations();

    // Use the final equilibria's Hessians for sensitivity analysis, not the second-to-last iterates'
    try {
        getFlatOptimizer()    .update_factorizations();
        getDeployedOptimizer().update_factorizations();
    }
    catch (const std::runtime_error &e) {
        std::cout << "Equilibrium solve failed: " << e.what() << std::endl;
        m_equilibriumSolveSuccessful = false;
        return;
    }

    // The cached adjoint state is invalidated whenever the equilibrium is updated...
    m_adjointStateIsCurrent      = false;
    m_autodiffLinkagesAreCurrent = false;

    m_updateClosestPoints();
}

bool LinkageOptimization::m_updateEquilibria(const Eigen::Ref<const Eigen::VectorXd> &newParams) {
    // Real dist = (m_linesearch_flat.getDesignParameters() - newParams).norm();
    // if (dist != 0) std::cout << "m_updateEquilibria at dist " << (m_linesearch_flat.getDesignParameters() - newParams).norm() << std::endl;
    if ((m_linesearch_flat.getDesignParameters() - newParams).norm() < 1e-16) return false;

    m_linesearch_deployed.set(m_deployed);
    m_linesearch_flat    .set(m_flat);

    const Eigen::VectorXd currParams = m_flat.getDesignParameters();
    Eigen::VectorXd delta_p = newParams - currParams;

    if (delta_p.norm() == 0) { // returning to linesearch start; no prediction/Hessian factorization necessary
        m_forceEquilibriumUpdate();
        return true;
        // The following caching is not working for some reason...
        // TODO: debug
        // auto &opt_2D = getFlatOptimizer();
        // auto &opt_3D = getDeployedOptimizer();

        // if (!(opt_2D.solver.hasStashedFactorization() && opt_3D.solver.hasStashedFactorization()))
        //     throw std::runtime_error("Factorization was not stashed... was commitLinesearchLinkage() called?");

        // // Copy the stashed factorization into the current one (preserving the stash)
        // opt_2D.solver.swapStashedFactorization();
        // opt_3D.solver.swapStashedFactorization();
        // opt_2D.solver.stashFactorization();
        // opt_3D.solver.stashFactorization();

        // // The cached adjoint state is invalidated whenever the equilibrium is updated...
        // m_adjointStateIsCurrent      = false;
        // m_autodiffLinkagesAreCurrent = false;

        // m_updateClosestPoints();
        // 
        // return true;
    }

    // Apply the new design parameters and measure the energy with the 0^th order prediction
    // (i.e. at the current equilibrium).
    // We will only replace this equilibrium if the higher order predictions achieve a lower energy.
    m_linesearch_deployed.setDesignParameters(newParams);
    m_linesearch_flat    .setDesignParameters(newParams);
    Real bestEnergy3d = m_linesearch_deployed.energy(),
         bestEnergy2d = m_linesearch_flat    .energy();

    Eigen::VectorXd curr_x3d = m_deployed.getDoFs(),
                    curr_x2d = m_flat    .getDoFs();
    Eigen::VectorXd best_x3d = curr_x3d,
                    best_x2d = curr_x2d;

    if (prediction_order > 0) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("Predict equilibrium");
        // Return to using the Hessian for the last committed linkage
        // (i.e. for the equilibrium stored in m_flat and m_deployed).
        auto &opt_2D = getFlatOptimizer();
        auto &opt_3D = getDeployedOptimizer();
        if (!(opt_2D.solver.hasStashedFactorization() && opt_3D.solver.hasStashedFactorization()))
            throw std::runtime_error("Factorization was not stashed... was commitLinesearchLinkage() called?");
        opt_2D.solver.swapStashedFactorization();
        opt_3D.solver.swapStashedFactorization();

        {
            // Solve for equilibrium perturbation corresponding to delta_p:
            //      [H_3D a][delta x     ] = [-d2E/dxdp delta_p]
            //      [a^T  0][delta lambda]   [        0        ]
            //                               \_________________/
            //                                        b
            const size_t np = numParams(), nd = m_flat.numDoF();
            VecX_T<Real> neg_deltap_padded(nd + np);
            neg_deltap_padded.setZero();
            neg_deltap_padded.tail(np) = -delta_p;

            // Computing -d2E/dxdp delta_p can skip the *-x and restlen-* blocks
            HessianComputationMask mask_dxdp;
            mask_dxdp.dof_in      = false;
            mask_dxdp.restlen_out = false;

            m_delta_x3d = opt_3D.extractFullSolution(opt_3D.kkt_solver(opt_3D.solver, opt_3D.removeFixedEntries(m_deployed.applyHessianPerSegmentRestlen(neg_deltap_padded, mask_dxdp).head(nd))));
            Eigen::VectorXd b_reduced = opt_2D.removeFixedEntries(m_flat.applyHessianPerSegmentRestlen(neg_deltap_padded, mask_dxdp).head(nd));
            if (opt_2D.get_problem().hasLEQConstraint()) m_delta_x2d = opt_2D.extractFullSolution(opt_2D.kkt_solver(opt_2D.solver, b_reduced));
            else                                         m_delta_x2d = opt_2D.extractFullSolution(opt_2D.solver.solve(b_reduced));

            // Evaluate the energy at the 1st order-predicted equilibrium
            {
                auto first_order_x3d = (curr_x3d + m_delta_x3d).eval(),
                     first_order_x2d = (curr_x2d + m_delta_x2d).eval();
                m_linesearch_deployed.setDoFs(first_order_x3d);
                m_linesearch_flat    .setDoFs(first_order_x2d);
                Real energy1stOrder3d = m_linesearch_deployed.energy(),
                     energy1stOrder2d = m_linesearch_flat    .energy();
                if (energy1stOrder3d < bestEnergy3d) { std::cout << " used first order prediction, energy reduction " << bestEnergy3d - energy1stOrder3d << std::endl; bestEnergy3d = energy1stOrder3d; best_x3d = first_order_x3d; } else { m_linesearch_deployed.setDoFs(best_x3d); }
                if (energy1stOrder2d < bestEnergy2d) { std::cout << " used first order prediction, energy reduction " << bestEnergy2d - energy1stOrder2d << std::endl; bestEnergy2d = energy1stOrder2d; best_x2d = first_order_x2d; } else { m_linesearch_flat    .setDoFs(best_x2d); }
            }
            
            if (prediction_order > 1) {
                // TODO: also stash autodiff linkages for committed linkages?
                // Solve for perturbation of equilibrium perturbation corresponding to delta_p:
                //      [H_3D a][delta_p^T d2x/dp^2 delta_p] = -[d3E/dx3 delta_x + d3E/dx2dp delta_p 0][delta_x     ] + [-(d3E/dxdpdx delta_x + d3E/dxdpdp delta_p) delta_p] = -[d3E/dx3 delta_x + d3E/dx2dp delta_p    d3E/dxdpdx delta_x + d3E/dxdpdp delta_p    0][delta_x     ]
                //      [a^T  0][delta_p^T d2l/dp^2 delta_p]    [0                                   0][delta_lambda] + [                       0                          ]    [                0                                         0                       0][delta_p     ]
                //                                                                                                                                                                                                                                                   [delta_lambda]
                m_diff_linkage_deployed.set(m_deployed);
                m_diff_linkage_flat    .set(m_flat);

                Eigen::VectorXd neg_d3E_delta_x3d, neg_d3E_delta_x2d;
                {
                    // inject design parameter perturbation.
                    VecX_T<ADReal> ad_p = currParams;
                    for (size_t i = 0; i < np; ++i) ad_p[i].derivatives()[0] = delta_p[i];
                    m_diff_linkage_deployed.setDesignParameters(ad_p);
                    m_diff_linkage_flat    .setDesignParameters(ad_p);

                    // inject equilibrium perturbation
                    VecX_T<ADReal> ad_x_3d = curr_x3d;
                    VecX_T<ADReal> ad_x_2d = curr_x2d;
                    for (int i = 0; i < ad_x_3d.size(); ++i) ad_x_3d[i].derivatives()[0] = m_delta_x3d[i];
                    for (int i = 0; i < ad_x_2d.size(); ++i) ad_x_2d[i].derivatives()[0] = m_delta_x2d[i];
                    m_diff_linkage_deployed.setDoFs(ad_x_3d);
                    m_diff_linkage_flat    .setDoFs(ad_x_2d);

                    VecX_T<Real> delta_edof_3d(nd + np);
                    VecX_T<Real> delta_edof_2d(nd + np);
                    delta_edof_3d.head(nd) = m_delta_x3d;
                    delta_edof_2d.head(nd) = m_delta_x2d;
                    delta_edof_3d.tail(np) = delta_p;
                    delta_edof_2d.tail(np) = delta_p;

                    neg_d3E_delta_x3d = -extractDirectionalDerivative(m_diff_linkage_deployed.applyHessianPerSegmentRestlen(delta_edof_3d)).head(nd);
                    neg_d3E_delta_x2d = -extractDirectionalDerivative(m_diff_linkage_flat    .applyHessianPerSegmentRestlen(delta_edof_2d)).head(nd);
                }

                m_delta_delta_x3d = opt_3D.extractFullSolution(opt_3D.kkt_solver(opt_3D.solver, opt_3D.removeFixedEntries(neg_d3E_delta_x3d)));
                b_reduced = opt_2D.removeFixedEntries(neg_d3E_delta_x2d);
                if (opt_2D.get_problem().hasLEQConstraint()) m_delta_delta_x2d = opt_2D.extractFullSolution(opt_2D.kkt_solver(opt_2D.solver, b_reduced));
                else                                         m_delta_delta_x2d = opt_2D.extractFullSolution(opt_2D.solver.solve(b_reduced));

                // Evaluate the energy at the 2nd order-predicted equilibrium, roll back to previous best if energy is higher.
                {
                    m_second_order_x3d = (curr_x3d + m_delta_x3d + 0.5 * m_delta_delta_x3d).eval(),
                    m_second_order_x2d = (curr_x2d + m_delta_x2d + 0.5 * m_delta_delta_x2d).eval();
                    m_linesearch_deployed.setDoFs(m_second_order_x3d);
                    m_linesearch_flat    .setDoFs(m_second_order_x2d);
                    Real energy2ndOrder3d = m_linesearch_deployed.energy(),
                         energy2ndOrder2d = m_linesearch_flat    .energy();
                    if (energy2ndOrder3d < bestEnergy3d) { std::cout << " used second order prediction, energy reduction " << bestEnergy3d - energy2ndOrder3d << std::endl; bestEnergy3d = energy2ndOrder3d; best_x3d = m_second_order_x3d;} else { m_linesearch_deployed.setDoFs(best_x3d); }
                    if (energy2ndOrder2d < bestEnergy2d) { std::cout << " used second order prediction, energy reduction " << bestEnergy2d - energy2ndOrder2d << std::endl; bestEnergy2d = energy2ndOrder2d; best_x2d = m_second_order_x2d;} else { m_linesearch_flat    .setDoFs(best_x2d); }
                }
            }
        }

        // Return to using the primary factorization, storing the committed
        // linkages' factorizations back in the stash for later use.
        opt_2D.solver.swapStashedFactorization();
        opt_3D.solver.swapStashedFactorization();
    }

    m_forceEquilibriumUpdate();

    return true;
}

void LinkageOptimization::m_updateMinAngleConstraintActuation() {
    if (!m_minAngleConstraint || !m_allowFlatActuation) return;

    getFlatOptimizer().optimize(); // We need to update the flat equilibrium to determine if the minimum angle constraint is in the working set

    // Add/remove the minimum angle constraint to the working set.
    if (m_minAngleConstraint->shouldRelease(m_linesearch_flat, getFlatOptimizer())) {
        m_minAngleConstraint->inWorkingSet = false;
    }
    else if (m_minAngleConstraint->violated(m_linesearch_flat)) {
        m_minAngleConstraint->inWorkingSet = true;
        Real alpha_bar_0 = m_linesearch_flat.getAverageJointAngle();
        m_minAngleConstraint->actuationAngle = alpha_bar_0;
        // Construct the actuated flat equilibrium solver if it hasn't been.
        if (!m_flat_optimizer_actuated) {
            m_flat_optimizer_actuated = get_equilibrium_optimizer(m_linesearch_flat, alpha_bar_0, m_rigidMotionFixedVars);
            m_flat_optimizer_actuated->options = m_equilibrium_options;
            m_flat_optimizer_actuated->optimize();
            m_linesearch_flat.updateSourceFrame();
            m_linesearch_flat.updateRotationParametrizations();
            m_flat_optimizer_actuated->update_factorizations();
        }
    }

    // If the minimum angle is in the working set, solve for the actuation angle such
    // that the bound is satisifed.
    m_minAngleConstraint->enforce(m_linesearch_flat, getFlatOptimizer());
}

// Update the adjoint state vectors "w", "y", and "s"
bool LinkageOptimization::m_updateAdjointState(const Eigen::Ref<const Eigen::VectorXd> &params) {
    m_updateEquilibria(params);
    if (m_adjointStateIsCurrent) return false;
    std::cout << "Updating adjoint state" << std::endl;

    // Solve the adjoint problems needed to efficiently evaluate the gradient.
    const auto &prob3D = getDeployedOptimizer().get_problem();

    // Note: if the Hessian modification failed (tau runaway), the adjoint state
    // solves will fail. To keep the solver from giving up entirely, we simply
    // set the adjoint state to 0 in these cases. Presumably this only happens
    // at bad iterates that will be discarded anyway.
    try {
        // Adjoint solve for the flatness constraint on the closed linkage:
        // H_2D y = 2 S_z^T S_z x_2D      or      [H_2D a][y_x     ] = [2 S_z^T S_z x_2D]
        //                                        [a^T  0][y_lambda]   [         0      ]
        // Depending on whether the closed linkage is actuated.
        {
            auto &opt = getFlatOptimizer();
            Eigen::VectorXd b_reduced = opt.removeFixedEntries(m_apply_S_z_transpose(2 * m_apply_S_z(m_linesearch_flat.getDoFs())));
            if (opt.get_problem().hasLEQConstraint()) m_y = opt.extractFullSolution(opt.kkt_solver(opt.solver, b_reduced));
            else                                      m_y = opt.extractFullSolution(opt.solver.solve(b_reduced));
        }

        // Adjoint solve for the target fitting objective on the deployed linkage
        {
            if (!prob3D.hasLEQConstraint()) throw std::runtime_error("The deployed linkage must have a linear equality constraint applied!");
            m_w_x = target_surface_fitter.adjoint_solve(m_linesearch_deployed, getDeployedOptimizer());
        }

        // Adjoint solve for the minimum opening angle constraint
        // [H_2D a][s_x     ] = [d alpha_min / d_x]
        // [a^T  0][s_lambda]   [        0        ]
        if (m_minAngleConstraint) {
            auto &opt = getFlatOptimizer();
            Eigen::VectorXd Hinv_b_reduced;
            opt.solver.solve(opt.removeFixedEntries(m_minAngleConstraint->grad(m_linesearch_flat)), Hinv_b_reduced);
            if (opt.get_problem().hasLEQConstraint()) m_s_x = opt.extractFullSolution(opt.kkt_solver.solve(Hinv_b_reduced));
            else                                      m_s_x = opt.extractFullSolution(Hinv_b_reduced);
        }
    }
    catch (...) {
        m_y.setZero();
        m_w_x.setZero();
    }

    m_adjointStateIsCurrent = true;

    return true;
}

Eigen::VectorXd LinkageOptimization::gradp_J(const Eigen::Ref<const Eigen::VectorXd> &params) {
    m_updateAdjointState(params);

    return (      gamma  / m_E0) * m_linesearch_flat    .grad_design_parameters(true)
        + ((1.0 - gamma) / m_E0) * m_linesearch_deployed.grad_design_parameters(true)
        + (beta / (m_l0 * m_l0)) * gradp_J_target(params);
}

Eigen::VectorXd LinkageOptimization::gradp_J_target(const Eigen::Ref<const Eigen::VectorXd> &params) {
    m_updateAdjointState(params);

    HessianComputationMask mask;
    mask.dof_out = false;
    mask.restlen_in = false;

    Eigen::VectorXd w_padded(m_linesearch_flat.numDoF() + numParams());
    w_padded.head(m_w_x.size()) = m_w_x;
    w_padded.tail(numParams()).setZero();
    return -m_linesearch_deployed.applyHessianPerSegmentRestlen(w_padded, mask).tail(numParams());
}

Eigen::VectorXd LinkageOptimization::gradp_c(const Eigen::Ref<const Eigen::VectorXd> &params) {
    m_updateAdjointState(params);

    HessianComputationMask mask;
    mask.dof_out = false;
    mask.restlen_in = false;

    Eigen::VectorXd y_padded(m_linesearch_flat.numDoF() + numParams());
    y_padded.head(m_y.size()) = m_y;
    y_padded.tail(numParams()).setZero();
    return -m_linesearch_flat.applyHessianPerSegmentRestlen(y_padded, mask).tail(numParams());
}

Eigen::VectorXd LinkageOptimization::gradp_angle_constraint(const Eigen::Ref<const Eigen::VectorXd> &params) {
    if (!m_minAngleConstraint) throw std::runtime_error("No minimum angle constraint is applied.");
    m_updateAdjointState(params);

    HessianComputationMask mask;
    mask.dof_out = false;
    mask.restlen_in = false;

    Eigen::VectorXd s_padded(m_linesearch_flat.numDoF() + numParams());
    s_padded.head(m_s_x.size()) = m_s_x;
    s_padded.tail(numParams()).setZero();
    return -m_linesearch_flat.applyHessianPerSegmentRestlen(s_padded, mask).tail(numParams());
}

Eigen::VectorXd LinkageOptimization::apply_hess(const Eigen::Ref<const Eigen::VectorXd> &params,
                                                const Eigen::Ref<const Eigen::VectorXd> &delta_p,
                                                Real coeff_J, Real coeff_c, Real coeff_angle_constraint) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("apply_hess_J");
    BENCHMARK_START_TIMER_SECTION("Preamble");
    const size_t np = numParams(), nd = m_linesearch_flat.numDoF();
    if (size_t( params.size()) != np) throw std::runtime_error("Incorrect parameter vector size");
    if (size_t(delta_p.size()) != np) throw std::runtime_error("Incorrect delta parameter vector size");
    m_updateAdjointState(params);

    if (!m_autodiffLinkagesAreCurrent) {
        BENCHMARK_SCOPED_TIMER_SECTION timer2("Update autodiff linkages");
        m_diff_linkage_deployed.set(m_linesearch_deployed);
        m_diff_linkage_flat    .set(m_linesearch_flat);
        m_autodiffLinkagesAreCurrent = true;
    }

    auto &opt  = getDeployedOptimizer();
    const auto &prob3D = opt.get_problem();
    if (!prob3D.hasLEQConstraint()) throw std::runtime_error("The deployed linkage must have a linear equality constraint applied!");
    auto &H_3D = opt.solver;

    BENCHMARK_STOP_TIMER_SECTION("Preamble");

    VecX_T<Real> neg_deltap_padded(nd + np);
    neg_deltap_padded.head(nd).setZero();
    neg_deltap_padded.tail(np) = -delta_p;

    // Computing -d2E/dxdp delta_p can skip the *-x and restlen-* blocks
    HessianComputationMask mask_dxdp, mask_dxpdx;
    mask_dxdp.dof_in      = false;
    mask_dxdp.restlen_out = false;
    mask_dxpdx.restlen_in = false;

    // Note: if the Hessian modification failed (tau runaway), the delta forward/adjoint state
    // solves will fail. To keep the solver from giving up entirely, we simply
    // set the failed quantities to 0 in these cases. Presumably this only happens
    // at bad iterates that will be discarded anyway.
    // Solve for closed state perturbation
    try {
        // H_2D delta_x = [-d2E/dxdp delta_p]   or   [H_2D a][delta_x     ] = [-d2E/dxdp delta_p]
        //                \_________________/        [a^T  0][delta_lambda]   [        0        ]
        //                         b                                          \_________________/
        //                                                                             b
        // Depending on whether the closed linkage is actuated.
        BENCHMARK_SCOPED_TIMER_SECTION timer2("solve delta x2d");
        auto &opt_2D = getFlatOptimizer();

        Eigen::VectorXd b_reduced = opt_2D.removeFixedEntries(m_linesearch_flat.applyHessianPerSegmentRestlen(neg_deltap_padded, mask_dxdp).head(nd));
        if (opt_2D.get_problem().hasLEQConstraint()) m_delta_x2d = opt_2D.extractFullSolution(opt_2D.kkt_solver(opt_2D.solver, b_reduced));
        else                                         m_delta_x2d = opt_2D.extractFullSolution(opt_2D.solver.solve(b_reduced));
    }
    catch (...) { m_delta_x2d.setZero(); }

    VecX_T<Real> d3E_s, d3E_y;
    try {
        // Solve for deployed state perturbation
        // [H_3D a][delta x     ] = [-d2E/dxdp delta_p]
        // [a^T  0][delta lambda]   [        0        ]
        //                          \_________________/
        //                                   b
        {
            BENCHMARK_SCOPED_TIMER_SECTION timer2("solve delta x3d");
            VecX_T<Real> b = m_linesearch_deployed.applyHessianPerSegmentRestlen(neg_deltap_padded, mask_dxdp).head(nd);
            m_delta_x3d = opt.extractFullSolution(opt.kkt_solver(H_3D, opt.removeFixedEntries(b)));
        }

        // Solve for deployed adjoint state perturbation
        BENCHMARK_START_TIMER_SECTION("getDoFs and inject state");
        VecX_T<ADReal> ad_p = params;
        for (size_t i = 0; i < np; ++i) ad_p[i].derivatives()[0] = delta_p[i];
        m_diff_linkage_deployed.setDesignParameters(ad_p); // inject design parameter perturbation.

        bool need_2d_autodiff = (coeff_c != 0.0) || (coeff_angle_constraint != 0.0);
        if (need_2d_autodiff) m_diff_linkage_flat.setDesignParameters(ad_p);

        auto ad_x_3d = m_diff_linkage_deployed.getDoFs();
        auto ad_x_2d = m_diff_linkage_flat    .getDoFs();

        auto inject_delta_state_3d = [&](VecX_T<Real> delta) {
            for (int i = 0; i < ad_x_3d.size(); ++i) ad_x_3d[i].derivatives()[0] = delta[i];
            m_diff_linkage_deployed.setDoFs(ad_x_3d);
        };

        auto inject_delta_state_2d = [&](VecX_T<Real> delta) {
            for (int i = 0; i < ad_x_2d.size(); ++i) ad_x_2d[i].derivatives()[0] = delta[i];
            m_diff_linkage_flat.setDoFs(ad_x_2d);
        };
        inject_delta_state_3d(m_delta_x3d);
        if (need_2d_autodiff)
            inject_delta_state_2d(m_delta_x2d);
        BENCHMARK_STOP_TIMER_SECTION("getDoFs and inject state");

        // [H_3D a][delta w_x     ] = [W delta_x + W_surf (I - dp_dx) delta_x ] - [d3E/dx dx dx delta_x + d3E/dx dx dp delta_p] w
        // [a^T  0][delta w_lambda]   [               0                       ]   [                     0                     ]
        //                            \________________________________________________________________________________________/
        //                                                                      b
        if (coeff_J != 0.0) {
            BENCHMARK_SCOPED_TIMER_SECTION timer2("solve delta w x");
            BENCHMARK_START_TIMER_SECTION("Hw");
            VecX_T<ADReal> w_padded(nd + np);
            w_padded.head(nd) = m_w_x;
            w_padded.tail(np).setZero();
            // Note: we need the "p" rows of d3E_w for evaluating the full Hessian matvec expressions below...
            m_d3E_w = extractDirectionalDerivative(m_diff_linkage_deployed.applyHessianPerSegmentRestlen(w_padded, mask_dxpdx));
            BENCHMARK_STOP_TIMER_SECTION("Hw");

            BENCHMARK_START_TIMER_SECTION("KKT_solve");
            m_delta_w_x = target_surface_fitter.delta_adjoint_solve(m_linesearch_deployed, opt, m_delta_x3d, m_d3E_w);
            BENCHMARK_STOP_TIMER_SECTION("KKT_solve");
        }

        // [H_2D a][delta s_x     ] = [d^2 alpha_min /d_x^2 delta_x] - [d3E/dx dx dx delta_x + d3E/dx dx dp delta_p] s
        // [a^T  0][delta s_lambda]   [               0            ]   [                     0                     ]
        //                            \______________________________________________________________________________/
        //                                                               b
        if (m_minAngleConstraint && (coeff_angle_constraint != 0.0)) {
            auto &opt_2D = getFlatOptimizer();

            BENCHMARK_SCOPED_TIMER_SECTION timer2("solve delta s x");
            BENCHMARK_START_TIMER_SECTION("Hs");
            VecX_T<ADReal> s_padded(nd + np);
            s_padded.head(nd) = m_s_x;
            s_padded.tail(np).setZero();
            // Note: we need the "p" rows of d3E_s for evaluating the full angle constraint Hessian matvec expression below...
            d3E_s = extractDirectionalDerivative(m_diff_linkage_flat.applyHessianPerSegmentRestlen(s_padded, mask_dxpdx));
            BENCHMARK_STOP_TIMER_SECTION("Hs");

            BENCHMARK_START_TIMER_SECTION("KKT_solve");
            auto b = (m_minAngleConstraint->delta_grad(m_linesearch_flat, m_delta_x2d) - d3E_s.head(nd)).eval();
            if (opt_2D.get_problem().hasLEQConstraint()) m_delta_s_x = opt_2D.extractFullSolution(opt_2D.kkt_solver(opt_2D.solver, opt_2D.removeFixedEntries(b)));
            else                                         m_delta_s_x = opt_2D.extractFullSolution(opt_2D.solver.solve(opt_2D.removeFixedEntries(b)));
            BENCHMARK_STOP_TIMER_SECTION("KKT_solve");
        }

        // H_2D delta y = 2 S_z^T S_z delta x_2D - delta H_2D y      or      [H_2D a][delta y_x     ] = [2 S_z^T S_z delta x_2D] - [delta H_2D] y
        //                                                                   [a^T  0][delta y_lambda]   [         0            ]   [          ]
        // depending on whether the closed linkage is actuated,
        // where delta H_2D = d3E/dx dx dx delta_x + d3E/dx dx dp delta_p.
        if (coeff_c != 0.0) {
            auto &opt_2D = getFlatOptimizer();
            BENCHMARK_SCOPED_TIMER_SECTION timer2("solve delta y");
            BENCHMARK_START_TIMER_SECTION("Hy");
            VecX_T<ADReal> y_padded(nd + np);
            y_padded.head(nd) = m_y;
            y_padded.tail(np).setZero();
            // Note: we need the "p" rows of d3E_y for evaluating the full Hessian matvec expressions below...
            d3E_y = extractDirectionalDerivative(m_diff_linkage_flat.applyHessianPerSegmentRestlen(y_padded, mask_dxpdx));

            BENCHMARK_STOP_TIMER_SECTION("Hy");

            BENCHMARK_START_TIMER_SECTION("KKT_solve");
            auto b = (m_apply_S_z_transpose(2 * m_apply_S_z(m_delta_x2d)) - d3E_y.head(nd)).eval();
            if (opt_2D.get_problem().hasLEQConstraint()) m_delta_y = opt_2D.extractFullSolution(opt_2D.kkt_solver(opt_2D.solver, opt_2D.removeFixedEntries(b)));
            else                                         m_delta_y = opt_2D.extractFullSolution(opt_2D.solver.solve(opt_2D.removeFixedEntries(b)));
            BENCHMARK_STOP_TIMER_SECTION("KKT_solve");
        }
    }
    catch (...) {
        m_delta_x3d = VecX_T<Real>::Zero(nd     );
        m_delta_w_x = VecX_T<Real>::Zero(nd     );
        m_delta_s_x = VecX_T<Real>::Zero(nd     );
        m_delta_y   = VecX_T<Real>::Zero(nd     );
        m_d3E_w     = VecX_T<Real>::Zero(nd + np);
        d3E_s       = VecX_T<Real>::Zero(nd + np);
    }

    VecX_T<Real> result;
    result.setZero(np);
    // Accumulate the J hessian matvec
    {
        BENCHMARK_SCOPED_TIMER_SECTION timer3("evaluate hessian matvec");

        if (coeff_J != 0.0) {
            VecX_T<Real> delta_edofs(nd + np);
            delta_edofs.head(nd) = m_delta_x2d;
            delta_edofs.tail(np) = delta_p;
            HessianComputationMask mask;
            mask.dof_out = false;

            result = (gamma / m_E0) * m_linesearch_flat.applyHessianPerSegmentRestlen(delta_edofs, mask).tail(np);

            delta_edofs.head(nd) = m_delta_x3d;
            delta_edofs *= ((1.0 - gamma) / m_E0);
            delta_edofs.head(nd) -= (beta / (m_l0 * m_l0)) * m_delta_w_x;

            result += m_linesearch_deployed.applyHessianPerSegmentRestlen(delta_edofs, mask).tail(np)
                   - (beta / (m_l0 * m_l0)) * m_d3E_w.tail(np);
            result *= coeff_J;
        }
        if (coeff_c != 0) {
            HessianComputationMask mask;
            mask.dof_out = false;
            VecX_T<Real> delta_edofs(nd + np);
            delta_edofs.head(nd) = -m_delta_y;
            delta_edofs.tail(np).setZero();
            result += coeff_c * (m_linesearch_flat.applyHessianPerSegmentRestlen(delta_edofs, mask).tail(np) - d3E_y.tail(np));
        }
        if (coeff_angle_constraint != 0) {
            HessianComputationMask mask;
            mask.dof_out = false;
            VecX_T<Real> delta_edofs(nd + np);
            delta_edofs.head(nd) = -m_delta_s_x;
            delta_edofs.tail(np).setZero();
            result += coeff_angle_constraint * (m_linesearch_flat.applyHessianPerSegmentRestlen(delta_edofs, mask).tail(np) - d3E_s.tail(np));
        }
    }

    return result;
}
