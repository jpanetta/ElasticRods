////////////////////////////////////////////////////////////////////////////////
// LinkageOptimization.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Evaluates objective, constraints, and gradients for solving the following
//  optimal design problem for rod linkages:
//      min_p J(p)
//      s.t. c(p) = 0
//      J(p) =      gamma  / E_0 E(x_2D(p), p) +
//             (1 - gamma) / E_0 E(x_3D(p), p) +
//             beta / (2 l_0^2) ||x_3D(p) - x_tgt||_W^2
//      c(p) = || S_z x_2D(p) ||^2,
//
//      where x_2D is the equilibrium configuration of the closed linkage,
//            x_3D is the equilibrium configuration of the opened linkage,
//            x_tgt are the user-specified target positions for each joint
//            gamma, beta, W are weights
//            S_z selects the z component of each joint
//  See writeup/LinkageOptimization.pdf for more information.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  11/23/2018 16:13:21
////////////////////////////////////////////////////////////////////////////////
#ifndef LINKAGEOPTIMIZATION_HH
#define LINKAGEOPTIMIZATION_HH

#include "RodLinkage.hh"
#include <MeshFEM/Geometry.hh>
#include "compute_equilibrium.hh"
#include "LOMinAngleConstraint.hh"
#include "TargetSurfaceFitter.hh"

struct LinkageOptimization {
    // allowFlatActuation: whether we allow the application of average-angle actuation to enforce the minimum angle constraint at the beginning of optimization.
    LinkageOptimization(RodLinkage &flat, RodLinkage &deployed, const NewtonOptimizerOptions &eopts = NewtonOptimizerOptions(), std::unique_ptr<LOMinAngleConstraint> &&minAngleConstraint = std::unique_ptr<LOMinAngleConstraint>(), int pinJoint = -1, bool allowFlatActuation = true);

    // Evaluate at a new set of parameters and commit this change to the flat/deployed linkages (which
    // are used as a starting point for solving the line search equilibrium)
    void newPt(const Eigen::VectorXd &params) {
        std::cout << "newPt at dist " << (m_flat.getDesignParameters() - params).norm() << std::endl;
        m_updateAdjointState(params); // update the adjoint state as well as the equilibrium, since we'll probably be needing gradients at this point.
        commitLinesearchLinkage();
    }

    size_t numParams() const { return m_numParams; }
    const Eigen::VectorXd &params() const { return m_flat.getDesignParameters(); }

    Real J()        { return J(params()); }
    Real J_target() { return J_target(params()); }
    Eigen::VectorXd gradp_J()        { return gradp_J(params()); }
    Eigen::VectorXd gradp_J_target() { return gradp_J_target(params()); }
    Real c()        { return c(params()); }

    Real J(const Eigen::Ref<const Eigen::VectorXd> &params) {
        std::cout << "eval at dist " << (m_flat.getDesignParameters() - params).norm() << std::endl;
        std::cout << "eval at linesearch dist " << (m_linesearch_flat.getDesignParameters() - params).norm() << std::endl;
        m_updateEquilibria(params);

        if (!m_equilibriumSolveSuccessful) return std::numeric_limits<Real>::max();

        Real val = (       gamma  / m_E0) * m_linesearch_flat.energy()
                 + ((1.0 - gamma) / m_E0) * m_linesearch_deployed.energy()
                 + (beta / (m_l0 * m_l0)) * J_target(params);
        std::cout << "value " << val << std::endl;
        return val;
    }

    Real J_target(const Eigen::Ref<const Eigen::VectorXd> &params) {
        m_updateEquilibria(params);
        return target_surface_fitter.objective(m_linesearch_deployed);
    }

    Real c(const Eigen::Ref<const Eigen::VectorXd> &params) {
        m_updateEquilibria(params);
        return m_apply_S_z(m_linesearch_flat.getDoFs()).squaredNorm();
    }

    Real angle_constraint(const Eigen::Ref<const Eigen::VectorXd> &params) {
        if (!m_minAngleConstraint) throw std::runtime_error("No minimum angle constraint is applied.");
        m_updateEquilibria(params);
        return m_minAngleConstraint->eval(m_linesearch_flat);
    }

    Eigen::VectorXd gradp_J               (const Eigen::Ref<const Eigen::VectorXd> &params);
    Eigen::VectorXd gradp_J_target        (const Eigen::Ref<const Eigen::VectorXd> &params);
    Eigen::VectorXd gradp_c               (const Eigen::Ref<const Eigen::VectorXd> &params);
    Eigen::VectorXd gradp_angle_constraint(const Eigen::Ref<const Eigen::VectorXd> &params);

    // Hessian matvec: H delta_p
    Eigen::VectorXd apply_hess_J               (const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p) { return apply_hess(params, delta_p, 1.0, 0.0, 0.0); }
    Eigen::VectorXd apply_hess_c               (const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p) { return apply_hess(params, delta_p, 0.0, 1.0, 0.0); }
    Eigen::VectorXd apply_hess_angle_constraint(const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p) { return apply_hess(params, delta_p, 0.0, 0.0, 1.0); }

    Eigen::VectorXd apply_hess(const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, Real coeff_J, Real coeff_c, Real coeff_angle_constraint);

    // Access adjoint state for debugging
    Eigen::VectorXd get_w_x() const { return m_w_x; }
    Eigen::VectorXd get_y()   const { return m_y; }
    Eigen::VectorXd get_s_x() const { return m_s_x; }

    Eigen::VectorXd get_delta_x3d() const { return m_delta_x3d; }
    Eigen::VectorXd get_delta_x2d() const { return m_delta_x2d; }
    Eigen::VectorXd get_delta_w_x() const { return m_delta_w_x; }
    Eigen::VectorXd get_delta_s_x() const { return m_delta_s_x; }
    Eigen::VectorXd get_delta_y  () const { return m_delta_y; }

    Eigen::VectorXd get_delta_delta_x3d() const { return m_delta_delta_x3d; }
    Eigen::VectorXd get_delta_delta_x2d() const { return m_delta_delta_x2d; }
    Eigen::VectorXd get_second_order_x3d() const { return m_second_order_x3d; }
    Eigen::VectorXd get_second_order_x2d() const { return m_second_order_x2d; }

    RodLinkage &getLinesearchFlatLinkage()     { return m_linesearch_flat; }
    RodLinkage &getLinesearchDeployedLinkage() { return m_linesearch_deployed; }

    // Change the deployed linkage's opening angle by "alpha", resolving for equilibrium.
    // Side effect: commits the linesearch linkage (like calling newPt)
    void setTargetAngle(Real alpha) {
        m_alpha_tgt = alpha;
        m_deployed_optimizer->get_problem().setLEQConstraintRHS(alpha);

        m_linesearch_flat    .set(m_flat    );
        m_linesearch_deployed.set(m_deployed);

        m_forceEquilibriumUpdate();
        commitLinesearchLinkage();
    }

    void setAllowFlatActuation(bool allow) {
        m_allowFlatActuation = allow;
        m_updateMinAngleConstraintActuation();
        m_forceEquilibriumUpdate();
        commitLinesearchLinkage();
    }

    void commitLinesearchLinkage() {
        m_flat    .set(m_linesearch_flat);
        m_deployed.set(m_linesearch_deployed);
        // Stash the current factorizations to be reused at each step of the linesearch
        // to predict the equilibrium at the new design parameters.
        getFlatOptimizer()    .solver.stashFactorization();
        getDeployedOptimizer().solver.stashFactorization();
    }

    Real getTargetAngle() const { return m_alpha_tgt; }

    // Get the index of the joint whose orientation is constrained to pin
    // down the linkage's rigid motion.
    size_t getRigidMotionConstrainedJoint() const { return m_rm_constrained_joint; }

    // Construct/update a target surface for the surface fitting term by
    // inferring a smooth surface from the current joint positions.
    // Also update the closest point projections.
    void constructTargetSurface(size_t loop_subdivisions = 0) {
        target_surface_fitter.constructTargetSurface(m_linesearch_deployed, loop_subdivisions);
        invalidateAdjointState();
    }

    void setEquilibriumOptions(const NewtonOptimizerOptions &eopts) {
        getDeployedOptimizer().options = eopts;
        getFlatOptimizer    ().options = eopts;
    }

    NewtonOptimizerOptions getEquilibriumOptions() const {
        return m_flat_optimizer->options;
    }

    NewtonOptimizer &getDeployedOptimizer() { return *m_deployed_optimizer; }
    NewtonOptimizer &getFlatOptimizer() {
        if (m_minAngleConstraint && m_allowFlatActuation && m_minAngleConstraint->inWorkingSet) {
            if (!m_flat_optimizer_actuated) throw std::runtime_error("Actuated flat linkage solver doesn't exist.");
            return *m_flat_optimizer_actuated;
        }
        return *m_flat_optimizer;
    }

    const LOMinAngleConstraint &getMinAngleConstraint() const {
        if (m_minAngleConstraint) return *m_minAngleConstraint;
        throw std::runtime_error("No min angle constraint has been applied.");
    }

    LOMinAngleConstraint &getMinAngleConstraint() {
        if (m_minAngleConstraint) return *m_minAngleConstraint;
        throw std::runtime_error("No min angle constraint has been applied.");
    }

    // When the fitting weights change the adjoint state must be recompouted.
    // Let the user manually inform us of this change.
    void invalidateAdjointState() { m_adjointStateIsCurrent = false; }

    // Write the full, dense Hessians of J and angle_constraint to a file.
    void dumpHessians(const std::string &hess_J_path, const std::string &hess_ac_path, Real fd_eps = 1e-5) {
        // Eigen::MatrixXd hess_J (m_numParams, m_numParams),
        //                 hess_ac(m_numParams, m_numParams);

        auto curr_params = params();
        auto grad_J = gradp_J(curr_params);

        // std::cout << "Evaluating full Hessians" << std::endl;
        // for (size_t p = 0; p < m_numParams; ++p) {
        //     hess_J .col(p) = apply_hess_J (curr_params, Eigen::VectorXd::Unit(m_numParams, p));
        //     hess_ac.col(p) = apply_hess_angle_constraint(curr_params, Eigen::VectorXd::Unit(m_numParams, p));
        // }

        // std::ofstream hess_J_file(hess_J_path);
        // hess_J_file.precision(19);
        // hess_J_file << hess_J << std::endl;

        // std::ofstream hess_ac_file(hess_ac_path);
        // hess_ac_file.precision(19);
        // hess_ac_file << hess_ac << std::endl;

        size_t nperturbs = 3;
        Eigen::VectorXd relerror_fd_diff_grad_p_J(nperturbs),
                        relerror_delta_Hw(nperturbs),
                        relerror_delta_w(nperturbs),
                        relerror_delta_w_rhs(nperturbs),
                        relerror_delta_x(nperturbs),
                        relerror_delta_J(nperturbs);
        Eigen::VectorXd matvec_relerror_fd_diff_grad_p_J(nperturbs);
        Eigen::VectorXd grad_J_relerror(nperturbs);

        auto w = m_w_x;
        auto H = m_linesearch_deployed.hessian();
        auto Hw = H.apply(w);
        // Real w_lambda = m_w_lambda;

        for (size_t i = 0; i < nperturbs; ++i) {
            Eigen::VectorXd perturb = Eigen::VectorXd::Random(m_numParams);

            apply_hess_J(curr_params, perturb);
            auto delta_w = m_delta_w_x;
            auto H_delta_w = m_linesearch_deployed.applyHessian(delta_w);

            Real Jplus = J(curr_params + fd_eps * perturb);
            auto gradp_J_plus = gradp_J(curr_params + fd_eps * perturb);
            auto w_plus = m_w_x;
            auto x_plus = m_linesearch_deployed.getDoFs();
            auto Hw_plus = m_linesearch_deployed.applyHessian(w);
            auto H_plus_w_plus = m_linesearch_deployed.applyHessian(w_plus);
            auto w_rhs_plus = m_w_rhs;
            auto H_plus = m_linesearch_deployed.hessian();
            // Real w_lambda_plus = m_w_lambda;

            {
                Eigen::VectorXd v = Eigen::VectorXd::Random(m_linesearch_deployed.numDoF());
                auto my_H = m_linesearch_deployed.hessianSparsityPattern(false);
                m_linesearch_deployed.hessian(my_H);
                auto matvec_one = my_H.apply(v);
                auto matvec_two = m_linesearch_deployed.applyHessian(v);
                std::cout << "matvec error: " << (matvec_one - matvec_two).norm() / matvec_one.norm() << std::endl;

                v = w_plus;
                matvec_one = my_H.apply(v);
                matvec_two = m_linesearch_deployed.applyHessian(v);
                auto matvec_three = m_linesearch_deployed.applyHessian(v);
                std::cout << "w_plus matec error: " << (matvec_one - matvec_two).norm() / matvec_one.norm() << std::endl;

                std::cout << "w_plus.norm(): " << w_plus.norm() << std::endl;
                std::cout << "H w_plus.norm() 1: " << matvec_one.norm() << std::endl;
                std::cout << "H w_plus.norm() 2: " << matvec_two.norm() << std::endl;
                std::cout << "H w_plus.norm() 3: " << matvec_three.norm() << std::endl;

                std::ofstream out_file_w("w_plus.txt");
                out_file_w.precision(16);
                out_file_w << w_plus << std::endl;

                std::ofstream out_file("matvec_one.txt");
                out_file.precision(16);
                out_file << matvec_one << std::endl;

                std::ofstream out_file2("matvec_two.txt");
                out_file2.precision(16);
                out_file2 << matvec_two << std::endl;

                std::ofstream out_file3("matvec_three.txt");
                out_file3.precision(16);
                out_file3 << matvec_two << std::endl;
            }

            Real Jminus = J(curr_params - fd_eps * perturb);
            auto gradp_J_minus = gradp_J(curr_params - fd_eps * perturb);
            auto w_minus = m_w_x;
            auto x_minus = m_linesearch_deployed.getDoFs();
            auto Hw_minus = m_linesearch_deployed.applyHessian(w);
            auto H_minus_w_minus = m_linesearch_deployed.applyHessian(w_minus);
            auto w_rhs_minus = m_w_rhs;
            // Real w_lambda_minus = m_w_lambda;

            Real fd_J = (Jplus - Jminus) / (2 * fd_eps);
            relerror_delta_J[i] = std::abs((grad_J.dot(perturb) - fd_J) / fd_J);

            Eigen::VectorXd fd_diff_grad_p_J = (gradp_J_plus - gradp_J_minus) / (2 * fd_eps);
            Eigen::VectorXd fd_delta_w = (w_plus - w_minus) / (2 * fd_eps);
            Eigen::VectorXd fd_delta_x = (x_plus - x_minus) / (2 * fd_eps);
            Eigen::VectorXd fd_delta_Hw = (Hw_plus - Hw_minus) / (2 * fd_eps);
            // Eigen::VectorXd fd_delta_w_rhs = (w_rhs_plus - w_rhs_minus) / (2 * fd_eps);
            // Real fd_delta_lambda = (w_lambda_plus - w_lambda_minus) / (2 * fd_eps);

#if 0
            auto fd_H_delta_w = H.apply(fd_delta_w);
            Eigen::VectorXd soln_error = ((Hw + fd_eps * fd_delta_Hw + fd_eps * fd_H_delta_w) - w_rhs_plus) + opt.extractFullSolution(opt.kkt_solver.a * (w_lambda + fd_eps * fd_delta_lambda));
            std::cout << "||Hw + delta_H w + H delta w + a (lambda + delta lambda) - b||: " << opt.removeFixedEntries(soln_error).norm() << std::endl;
            std::cout << soln_error.head(8).transpose() << std::endl;
            std::cout << soln_error.segment<8>(m_linesearch_deployed.dofOffsetForJoint(0)).transpose() << std::endl;

            Eigen::VectorXd soln_error2 = (H_plus.apply(w_plus) + w_lambda_plus * opt.extractFullSolution(opt.kkt_solver.a) - w_rhs_plus);
            std::cout << "||Hw_plus + a lambda_plus - b_plus||: " << opt.removeFixedEntries(soln_error2).norm() << std::endl;

            Eigen::VectorXd soln_error3 = ((Hw + fd_eps * fd_delta_Hw + fd_eps * H.apply(delta_w)) - w_rhs_plus) + opt.extractFullSolution(opt.kkt_solver.a * (w_lambda + fd_eps * fd_delta_lambda));
            std::cout << "||Hw + delta_H w + H delta w + a (lambda + delta lambda) - b||: " << opt.removeFixedEntries(soln_error3).norm() << std::endl;

            Eigen::VectorXd soln_error4 = H_plus_w_plus - w_rhs_plus + opt.extractFullSolution(opt.kkt_solver.a * w_lambda_plus);
            std::cout << "||H_plus w_plus + a (lambda + delta lambda) - b||: " << opt.removeFixedEntries(soln_error4).norm() << std::endl;
#endif

            // Eigen::VectorXd fd_diff_grad_p_ac = (gradp_angle_constraint(curr_params + fd_eps * perturb) - gradp_angle_constraint(curr_params - fd_eps * perturb)) / (2 * fd_eps);
            // relerror_fd_diff_grad_p_ac[i] = (hess_ac * perturb - fd_diff_grad_p_ac).norm() / fd_diff_grad_p_ac.norm();

            // relerror_fd_diff_grad_p_J[i] = (hess_J  * perturb - fd_diff_grad_p_J ).norm() / fd_diff_grad_p_J .norm();

            matvec_relerror_fd_diff_grad_p_J[i] = (apply_hess_J(curr_params, perturb) - fd_diff_grad_p_J ).norm() / fd_diff_grad_p_J .norm();
            relerror_delta_x[i] = (m_delta_x3d - fd_delta_x).norm() / fd_delta_x.norm();
            relerror_delta_w[i] = (m_delta_w_x - fd_delta_w).norm() / fd_delta_w.norm();
            relerror_delta_Hw[i] = (m_d3E_w.head(w.size()) - fd_delta_Hw).norm() / fd_delta_Hw.norm();
            // relerror_delta_w_rhs[i] = (m_delta_w_rhs - fd_delta_w_rhs).norm() / fd_delta_w_rhs.norm();

            // // Eigen::VectorXd semi_analytic_rhs = fd_delta_w_rhs - fd_delta_Hw;
            // opt.update_factorizations();
            // Eigen::VectorXd semi_analytic_delta_w = opt.extractFullSolution(opt.kkt_solver(opt.solver, opt.removeFixedEntries(semi_analytic_rhs)));
            // relerror_semi_analytic_delta_w[i] = (semi_analytic_delta_w - fd_delta_w).norm() / fd_delta_w.norm();

            // {
            //     std::cout << "semi_analytic_delta_w: " << semi_analytic_delta_w.head(5).transpose() << std::endl;
            //     std::cout << "delta_w: " << m_delta_w_x.head(5).transpose() << std::endl;
            //     std::cout << "fd_delta_w: " << fd_delta_w.head(5).transpose() << std::endl;
            //     std::cout << std::endl;

            //     int idx;
            //     Real err = (semi_analytic_delta_w - fd_delta_w).cwiseAbs().maxCoeff(&idx);
            //     std::cout << "greatest abs error " << err << " at entry " << idx << ": "
            //               <<  semi_analytic_delta_w[idx] << " vs " << fd_delta_w[idx] << std::endl;
            //     std::cout <<  semi_analytic_delta_w.segment(idx - 5, 10).transpose() << std::endl;
            //     std::cout << m_delta_w_x.segment(idx - 5, 10).transpose() << std::endl;
            //     std::cout <<  fd_delta_w.segment(idx - 5, 10).transpose() << std::endl;
            //     std::cout << std::endl;
            // }
        }


        std::cout << "Wrote " << hess_J_path << ", " << hess_ac_path << std::endl;
        std::cout << "Rel errors in delta        J: " << relerror_delta_J .transpose() << std::endl;
        // std::cout << "Rel errors in hessian-vec  J: " << relerror_fd_diff_grad_p_J .transpose() << std::endl;
        std::cout << "Rel errors in matvec hessian-vec  J: " << matvec_relerror_fd_diff_grad_p_J .transpose() << std::endl;
        std::cout << "Rel errors in delta x: " << relerror_delta_x.transpose() << std::endl;
        std::cout << "Rel errors in delta w: " << relerror_delta_w.transpose() << std::endl;
        std::cout << "Rel errors in delta Hw: " << relerror_delta_Hw.transpose() << std::endl;
        // std::cout << "Rel errors in delta w rhs: " << relerror_delta_w_rhs.transpose() << std::endl;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Public member variables
    ////////////////////////////////////////////////////////////////////////////
    Real beta = 500000, gamma = 0.9;

    TargetSurfaceFitter target_surface_fitter;
    // Configure how the equilibrium at a perturbed set of parameters is predicted (using 0th, 1st, or 2nd order Taylor expansion)
    size_t prediction_order = 2;

private:
    void m_forceEquilibriumUpdate();
    // Return whether "params" are actually new...
    bool m_updateEquilibria(const Eigen::Ref<const Eigen::VectorXd> &params);

    // Update the closest point projections for each joint to the target surface.
    void m_updateClosestPoints() { target_surface_fitter.updateClosestPoints(m_linesearch_deployed); }

    // Check if the minimum angle constraint is active and if so, change the closed
    // configuration's actuation angle to satisfy the constraint.
    void m_updateMinAngleConstraintActuation();

    // Update the adjoint state vectors "w" and "y"
    bool m_updateAdjointState(const Eigen::Ref<const Eigen::VectorXd> &params);

    // Extract the z coordinates of the joints
    Eigen::VectorXd m_apply_S_z(const Eigen::Ref<const Eigen::VectorXd> &x) {
        Eigen::VectorXd result(m_flat.numJoints());
        for (size_t ji = 0; ji < m_flat.numJoints(); ++ji)
            result[ji] = x[m_flat.dofOffsetForJoint(ji) + 2];
        return result;
    }

    // Take a vector of per-joint z coordinates and place them in the
    // appropriate locations of the state vector.
    Eigen::VectorXd m_apply_S_z_transpose(const Eigen::Ref<const Eigen::VectorXd> &zcoords) {
        Eigen::VectorXd result = Eigen::VectorXd::Zero(m_flat.numDoF());
        for (size_t ji = 0; ji < m_flat.numJoints(); ++ji)
            result[m_flat.dofOffsetForJoint(ji) + 2] = zcoords[ji];
        return result;
    }

    // Apply the joint position weight matrix W to a compressed state vector that
    // contains only variables corresponding to joint positions.
    // Returns an uncompressed vector with an entry for each state variable.
    Eigen::VectorXd m_apply_W    (const Eigen::Ref<const Eigen::VectorXd> &x_joint_pos) { return m_unpackJointPositions(target_surface_fitter.W_diag_joint_pos.cwiseProduct(x_joint_pos)); }
    Eigen::VectorXd m_apply_Wsurf(const Eigen::Ref<const Eigen::VectorXd> &x_joint_pos) { return m_unpackJointPositions(target_surface_fitter.Wsurf_diag_joint_pos.cwiseProduct(x_joint_pos)); }

    // Extract a full state vector from a compressed version that only holds
    // variables corresponding to joint positions.
    Eigen::VectorXd m_unpackJointPositions(const Eigen::Ref<const Eigen::VectorXd> &x_joint_pos) {
        Eigen::VectorXd result = Eigen::VectorXd::Zero(m_deployed.numDoF());

        for (size_t ji = 0; ji < m_deployed.numJoints(); ++ji)
            result.segment<3>(m_deployed.dofOffsetForJoint(ji)) = x_joint_pos.segment<3>(3 * ji);
        return result;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Private member variables
    ////////////////////////////////////////////////////////////////////////////
    NewtonOptimizerOptions m_equilibrium_options;
    Eigen::VectorXd m_w_x, m_y, m_s_x; // adjoint state vectors
    Eigen::VectorXd m_delta_w_x, m_delta_x2d, m_delta_x3d, m_delta_s_x, m_delta_y; // variations of adjoint/forward state from the last call to apply_hess (for debugging)
    Eigen::VectorXd m_delta_delta_x2d, m_delta_delta_x3d;   // second variations of forward state from last call to m_updateEquilibrium (for debugging)
    Eigen::VectorXd m_second_order_x3d, m_second_order_x2d; // second-order predictions of the linkage's equilibrium (for debugging)
    Eigen::VectorXd m_d3E_w;
    Eigen::VectorXd m_w_rhs, m_delta_w_rhs;
    // Real m_w_lambda, m_delta_w_lambda;
    std::vector<size_t> m_rigidMotionFixedVars;
    const size_t m_numParams;
    size_t m_rm_constrained_joint; // index of joint whose rigid motion is constrained.
    const Real m_E0 = 1.0, m_l0 = 1.0;
    Real m_alpha_tgt = 0.0;
    RodLinkage &m_flat, &m_deployed;
    RodLinkage m_linesearch_flat, m_linesearch_deployed;
    std::unique_ptr<NewtonOptimizer> m_flat_optimizer, m_deployed_optimizer;

    std::unique_ptr<LOMinAngleConstraint> m_minAngleConstraint;
    std::unique_ptr<NewtonOptimizer> m_flat_optimizer_actuated;

    RodLinkage_T<ADReal> m_diff_linkage_flat, m_diff_linkage_deployed;

    bool m_allowFlatActuation = true; // whether we allow the application of average-angle actuation to enforce the minimum angle constraint at the beginning of optimization.
    bool m_adjointStateIsCurrent = false, m_autodiffLinkagesAreCurrent = false;
    bool m_equilibriumSolveSuccessful = false;
};

#endif /* end of include guard: LINKAGEOPTIMIZATION_HH */
