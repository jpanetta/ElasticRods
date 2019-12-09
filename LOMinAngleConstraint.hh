////////////////////////////////////////////////////////////////////////////////
// LOMinAngleConstraint.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Helper class for satisfying a minimum angle constraint by actuating a
//  linkage to an average opening angle.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  12/15/2018 15:36:07
////////////////////////////////////////////////////////////////////////////////
#ifndef LOMINANGLECONSTRAINT_HH
#define LOMINANGLECONSTRAINT_HH

struct LOMinAngleConstraint {
    LOMinAngleConstraint() { }
    LOMinAngleConstraint(Real eps_) : eps(eps_) { }

    Real eps = M_PI / 128;
    Real actuationAngle = 0;
    Real tol = 1e-7;
    Real s = 1e-2; // "smoothness" for KS function approximating the minimum joint angle; setting this to zero results in exact minimum

    // Smooth approximation to infinity
    Real alpha_min(const RodLinkage &l) const {
        Real exact_min = l.getMinJointAngle();
        if (s == 0) return exact_min; // avoid divide by zero UB in exact minimum case
        Real sum = 0;
        for (const auto &j : l.joints())
            sum += exp((exact_min - j.alpha()) / s); // should never overflow since the standard library guarantees exp(-infinity) = 0
        // Note: sum is always >= 1, so s * log(sum) is always >= 0.
        // This means the approximate minimum is "conservative."
        return exact_min - s * log(sum);
    }

    // Eval the constraint c(x) = alpha_min(x) - eps >= 0
    Real eval(const RodLinkage &l) const { return alpha_min(l) - eps; }

    // dc / dx = d/dx alpha_min(x)
    Eigen::VectorXd grad(const RodLinkage &l) const {
        Real exact_min = l.getMinJointAngle();

        Eigen::VectorXd result(l.numDoF());
        result.setZero();
        Real denominator = 0;
        const size_t nj = l.numJoints();
        for (size_t ji = 0; ji < nj; ++ji) {
            Real coeff = 0.0;
            if (s == 0.0) {
                // Avoid divide by zero in the exact minimum case.
                // In this case, the minimum angle is non-differentiable, but
                // we compute a subderivative that doesn't favor any particular joint.
                if (std::abs(exact_min - l.joint(ji).alpha()) < 1e-15)
                    coeff = 1.0;
            }
            else { coeff = exp((exact_min - l.joint(ji).alpha()) / s); }
            result[l.dofOffsetForJoint(ji) + 6] = coeff;
            denominator += coeff;
        }

        assert(denominator > 0);

        result /= denominator;

        return result;
    }

    Eigen::VectorXd delta_grad(const RodLinkage &l, const Eigen::VectorXd &delta_x) const {
        Real exact_min = l.getMinJointAngle();

        // Avoid divide by zero in the exact minimum case.
        if (s == 0.0) return Eigen::VectorXd::Zero(l.numDoF());

        const size_t nj = l.numJoints();
        Eigen::VectorXd result(l.numDoF()), exp_terms_j(nj);
        for (size_t ji = 0; ji < nj; ++ji)
            exp_terms_j[ji] = exp((exact_min - l.joint(ji).alpha()) / s);

        // Note: the following calculation takes O(nj^2). An O(nj) formula can
        // be derived easily, but it suffers from (worse) catastrophic
        // cancellation for small s. Since the number of joints is typically
        // small, this calculation should still be fast enough.
        // TODO: derive a more robust formula that can still be computed in linear time.
        result.setZero();
        for (size_t ji_a = 0; ji_a < nj; ++ji_a) {
            const size_t dof_offset_a = l.dofOffsetForJoint(ji_a) + 6;
            for (size_t ji_b = 0; ji_b < ji_a; ++ji_b) { // skips the diagonal contributions, which are zero (delta_j - delta_j).
                const size_t dof_offset_b = l.dofOffsetForJoint(ji_b) + 6;
                Real contrib = exp_terms_j[ji_a] * exp_terms_j[ji_b] * (delta_x[dof_offset_b] - delta_x[dof_offset_a]);
                result[dof_offset_a] += contrib;
                result[dof_offset_b] -= contrib;
            }
        }
        Real sum_exp_terms = exp_terms_j.sum();
        result /= s * sum_exp_terms * sum_exp_terms;

        return result;
    }

    // Assumes optimizer's factorization/kkt_solver is up-to-date.
    Eigen::VectorXd solve_dx_dalphabar(NewtonOptimizer &opt) const {
        //   [H_2d a][dx/dalpha_bar] = [0]
        //   [a^T  0][dl/dalpha_bar]   [1]
        return opt.extractFullSolution(opt.kkt_solver(opt.solver, Eigen::VectorXd::Zero(opt.get_problem().numReducedVars()), 1.0));
    }

    // If the constraint is in the working set, check if the Lagrange multiplier is negative;
    // in this case, we should release the constraint.
    bool shouldRelease(const RodLinkage &l, NewtonOptimizer &opt) const {
        if (!inWorkingSet) return false;

        auto dx_dalphabar = solve_dx_dalphabar(opt);
        Real dE_dalphabar = l.gradient(true).dot(dx_dalphabar);
        Real dalphamin_dalphabar = grad(l).dot(dx_dalphabar);

        // If varying alphabar changes E and alphamin in opposite directions, we can
        // decrease the energy by releasing the constraint.
        // If either E or alphamin is stationary with respect to alphabar, the constraint
        // is not doing anything and can be released.
        return (dE_dalphabar * dalphamin_dalphabar) <= 0;
    }

    // Solve the nonlinear equation c(x(actuationAngle)) == 0
    void enforce(RodLinkage &l, NewtonOptimizer &opt, const Real maxAngleStep = 0.02) {
        if (!inWorkingSet) return;
        auto &prob = opt.get_problem();

        Real residual = eval(l);
        // size_t it = 0;
        while (std::abs(residual) > tol) {
            auto dx_dalphabar = solve_dx_dalphabar(opt);
            Real dalphamin_dalphabar = grad(l).dot(dx_dalphabar);
#if 0
            {
                std::ofstream dx_dalphabar_file("dx_dalphabar_" + std::to_string(it) + ".txt");
                dx_dalphabar_file << dx_dalphabar << std::endl;
                l.saveVisualizationGeometry("flat_actuation_" + std::to_string(it++) + ".msh");
                std::cout << "LOMinAngleConstraint residual: " << residual << std::endl;
                std::cout << "dx_dalphabar.norm(): " << dx_dalphabar.norm() << std::endl;
            }
#endif
            Real delta_alphabar = -residual / dalphamin_dalphabar;
            actuationAngle += std::copysign(std::min(std::abs(delta_alphabar), maxAngleStep), delta_alphabar); // clamped Newton step

            prob.setLEQConstraintRHS(actuationAngle);
            opt.optimize();
            l.updateSourceFrame();
            l.updateRotationParametrizations();
            // TODO: linesearch?
            std::cout << "residual: " << residual << std::endl;

            residual = eval(l);
        }
        std::cout << "flat actuation angle: " << actuationAngle << std::endl;
    }

    bool violated(const RodLinkage &l) const { return eval(l) < 0; }
    bool inWorkingSet = false;
};


#endif /* end of include guard: LOMINANGLECONSTRAINT_HH */
