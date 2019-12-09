#ifndef ACTUATIONSPARSIFIER_HH
#define ACTUATIONSPARSIFIER_HH

#include "RodLinkage.hh"
#include "TargetSurfaceFitter.hh"

struct ActuationSparsifier {
    enum class Regularization  { L1, L0, LP };
    ActuationSparsifier(RodLinkage &deployedLinkage, NewtonOptimizer &optimizer, Regularization regularization = Regularization::L1)
        : linkage(deployedLinkage), linesearch_linkage(deployedLinkage),
          equilibriumOptimizer(get_equilibrium_optimizer(linesearch_linkage, TARGET_ANGLE_NONE, optimizer.get_problem().fixedVars())),
          equilibriumProblem(dynamic_cast<EquilibriumProblem<RodLinkage>&>(equilibriumOptimizer->get_problem())),
          m_ndof(deployedLinkage.numDoF()), m_numTorques(deployedLinkage.numJoints()),
          m_l0(BBox<Point3D>(deployedLinkage.deformedPoints()).dimensions().norm()),
          m_regularization(regularization)
    {
        equilibriumProblem.external_forces = dynamic_cast<EquilibriumProblem<RodLinkage>&>(optimizer.get_problem()).external_forces;

        target_surface_fitter.joint_pos_tgt = linkage.jointPositions();
        target_surface_fitter.constructTargetSurface(linkage);

        // Trade off between fitting to the individual joint targets and the target surface.
        target_surface_fitter.setTargetJointPosVsTargetSurfaceTradeoff(linkage, 1.0);

        auto vars = initialVarsFromElasticForces(equilibriumProblem.external_forces);
        std::cout << "Force vector reconstruction error: " << (forceVectorFromVars(vars) - equilibriumProblem.external_forces).norm() << std::endl;
        m_forceEquilibriumUpdate(vars);
        linkage.set(linesearch_linkage);
    }

    VecX_T<Real> initialVarsFromElasticForces(const VecX_T<Real> &allForces) const {
        VecX_T<Real> result(numVars()); // ensures slack variables "xi" are created in the Regularization::L0 case.
        result.setZero();               // set the "xi" variables to 0 (to satisfy complementarity constraints)
        if (m_regularization == Regularization::LP)
            result = extractTorques(allForces).array().pow(p); // We use the torques raised to the "p" power as variables.
        else result.head(m_numTorques) = extractTorques(allForces);
        return result;
    }

    // Evaluate at a new set of forces and commit this change to the linkages (which
    // are used as a starting point for solving the line search equilibrium)
    void newPt(const Eigen::VectorXd &vars) {
        m_updateAdjointState(vars); // update the adjoint state as well as the equilibrium, since we'll probably be needing gradients at this point.
        linkage.set(linesearch_linkage);

        size_t numActuated = 0;
        auto torques = extractTorques(linkage.gradient());
        for (int i = 0; i < torques.size(); ++i)
            if (torques[i] > 1e-6) ++numActuated;
        std::cout << "Actuated joints: " << numActuated << "/" << linkage.numJoints() << std::endl;
        std::cout << "L1 norm:" << vars.segment(0, m_numTorques).sum() << std::endl;
        std::cout << "Distance: " << target_surface_fitter.objective(linesearch_linkage) << std::endl;
    }

    Real eval(const Eigen::VectorXd &vars) {
        m_updateEquilibrium(vars);
        Real regTerm = 0.0;
        if (m_regularization == Regularization::L0) {
            // In both cases we add the term "sum_i (1.0 - xi_i)"
            // (These m_numTorques xi variables are simply correlated with different
            //  force variables by the complementarity constraints).
            regTerm = m_numTorques - vars.tail(m_numTorques).sum();
        }
        else if (m_regularization == Regularization::L1) {
            // Note: all variables are bounded to be positive, so their L1 norm is simply their sum.
            regTerm = vars.sum();  // sum of torques
        }
        else if (m_regularization == Regularization::LP) {
            regTerm =  vars.sum(); // the vars are already the torques raised to the "p" power. Since we're evaluating the Lp "norm" raised to the p value, the result is just the sum...
        }
        else assert(false);

        return target_surface_fitter.objective(linesearch_linkage) + eta * regTerm;
    }

    VecX_T<Real> grad(const Eigen::VectorXd &vars) {
        m_updateEquilibrium(vars);
        m_updateAdjointState(vars);

        VecX_T<Real> gradRegTerm;
        if (m_regularization == Regularization::L0) {
            // gradient of "eta * sum_i (1 - xi_i)"
            gradRegTerm.setZero(numVars());
            gradRegTerm.tail(m_numTorques) = Eigen::VectorXd::Constant(numTorques(), -eta);
        }
        else if (m_regularization == Regularization::L1) {
            gradRegTerm = Eigen::VectorXd::Constant(vars.size(), eta);
        }
        else if (m_regularization == Regularization::LP) {
            gradRegTerm = Eigen::VectorXd::Constant(vars.size(), eta);
        }
        else assert(false);

        return apply_d_force_d_vars_transpose(vars, m_w) + gradRegTerm;
    }

    VecX_T<Real> apply_hess(const Eigen::Ref<const Eigen::VectorXd> &vars,
                            const Eigen::Ref<const Eigen::VectorXd> &delta_vars) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("apply_hess");
        if (size_t(      vars.size()) != numVars()) throw std::runtime_error("Incorrect vars size");
        if (size_t(delta_vars.size()) != numVars()) throw std::runtime_error("Incorrect delta vars size");
        m_updateAdjointState(vars);

        if (!m_autodiffLinkageIsCurrent) {
            diff_linkage.set(linesearch_linkage);
            m_autodiffLinkageIsCurrent = true;
        }

        // solve for x perturbation due to changing forces.
        auto &opt = *equilibriumOptimizer;
        m_delta_x = opt.extractFullSolution(opt.solver.solve(opt.removeFixedEntries(apply_d_force_d_vars(vars, delta_vars))));

        // inject x perturbation
        auto ad_x = diff_linkage.getDoFs();
        for (int i = 0; i < ad_x.size(); ++i) ad_x[i].derivatives()[0] = m_delta_x[i];
        diff_linkage.setDoFs(ad_x);

        auto d3E_w = extractDirectionalDerivative(diff_linkage.applyHessian(m_w));
        m_delta_w = target_surface_fitter.delta_adjoint_solve(linesearch_linkage, opt, m_delta_x, d3E_w);

        VecX_T<Real> result = apply_d_force_d_vars_transpose(vars, m_delta_w) + apply_delta_d_force_d_vars_transpose(vars, delta_vars, m_w);
        // // Only the LP regularization term has a nonzero Hessian
        // if (m_regularization == Regularization::LP)
        //     result += Eigen::VectorXd::Constant(vars.size(), eta * (1.0 / p) * (1.0 / p - 1) * std::pow(vars.sum(), 1.0 / p - 2.0) * delta_vars.sum());
        return result;
    }

    VecX_T<Real> extractTorques(const VecX_T<Real> &allForces) const {
        if (size_t(allForces.size()) != m_ndof) throw std::runtime_error("Unexpected force vector size");
        const size_t nj = linkage.numJoints();
        VecX_T<Real> result(nj);
        result.setZero();
        for (size_t ji = 0; ji < nj; ++ji) {
            size_t torqueIdx = linkage.dofOffsetForJoint(ji) + 6;
            result[ji] = allForces[torqueIdx];
        }
        return result;
    }

    VecX_T<Real> forceVectorFromTorques(const Eigen::Ref<const VecX_T<Real>> &torques) const {
        if (size_t(torques.size()) != m_numTorques) throw std::runtime_error("Unexpected torques size");
        const size_t nj = linkage.numJoints();
        VecX_T<Real> result(m_ndof);
        result.setZero();

        for (size_t ji = 0; ji < nj; ++ji) {
            size_t torqueIdx = linkage.dofOffsetForJoint(ji) + 6;
            result[torqueIdx] = torques[ji];
        }

        return result;
    }

    VecX_T<Real> forceVectorFromVars(const VecX_T<Real> &vars) const {
        if (m_regularization == Regularization::LP)
            return forceVectorFromTorques(vars.array().pow(1.0 / p).matrix());
        return forceVectorFromTorques(vars.head(m_numTorques)); // strip off "xi" variables in the Regularization::L0 case
    }

    // d_force_d_vars:
    //  This is a selection of rows/cols of the identity matrix, which we
    //  refer to as the selection matrix "S"
    VecX_T<Real> apply_d_force_d_vars(const VecX_T<Real> &vars, const VecX_T<Real> &delta_vars) const {
        if (m_regularization == Regularization::LP)
            return forceVectorFromTorques((1.0 / p) * (vars.array().pow(1.0 / p - 1.0).matrix().cwiseProduct(delta_vars)));
        return forceVectorFromTorques(delta_vars.head(m_numTorques)); // strip off "xi" variables in the Regularization::L0 case
    }

    VecX_T<Real> apply_d_force_d_vars_transpose(const VecX_T<Real> &vars, const VecX_T<Real> &forces) const {
        if (m_regularization == Regularization::LP)
            return (1.0 / p) * (vars.array().pow(1.0 / p - 1.0).matrix().cwiseProduct(extractTorques(forces)));
        VecX_T<Real> result(numVars());
        result.setZero(); // zero out the entries corresponding to "xi" variables (to handle Regularization::L0 case)
        result.head(m_numTorques) = extractTorques(forces);

        return result;
    }

    VecX_T<Real> apply_delta_d_force_d_vars_transpose(const VecX_T<Real> &vars, const VecX_T<Real> &delta_vars, const VecX_T<Real> &forces) const {
        if (m_regularization == Regularization::LP)
            return ((1.0 / p) * (1.0 / p - 1.0)) * (vars.array().pow(1.0 / p - 2.0).matrix().cwiseProduct(delta_vars).cwiseProduct(extractTorques(forces)));
        return VecX_T<Real>::Zero(numVars());
    }

    size_t numVars()      const { return numForceVars() + numSlackVars(); }
    size_t numForceVars() const { return numTorques(); }
    size_t numSlackVars() const { return (m_regularization == Regularization::L0) ? m_numTorques : 0; }
    size_t numTorques()   const { return m_numTorques; }

    const VecX_T<Real>  get_x()       const { return linesearch_linkage.getDoFs(); }
    const VecX_T<Real> &get_w()       const { return m_w; }
    const VecX_T<Real> &get_delta_x() const { return m_delta_x; }
    const VecX_T<Real> &get_delta_w() const { return m_delta_w; }

    TargetSurfaceFitter target_surface_fitter;
    Real eta = 0.02; // weight for the sparsifying regularization term
    Real   p = 0.125;
    RodLinkage &linkage, linesearch_linkage;
    RodLinkage_T<ADReal> diff_linkage;

    std::unique_ptr<NewtonOptimizer> equilibriumOptimizer;
    EquilibriumProblem<RodLinkage>  &equilibriumProblem; // reference to equilibriumOptimizer's problem instance

    Eigen::VectorXd linesearchEvalPt;

    Regularization   regularization() const { return m_regularization; }

private:
    void m_forceEquilibriumUpdate(const Eigen::VectorXd &vars) {
        // std::cout << "vars.size(): " << vars.size() << std::endl;
        // std::cout << "numVars():" << numVars() << std::endl;
        // std::cout << "numForceVars():" << numForceVars() << std::endl;
        // std::cout << "numSlackVars():" << numSlackVars() << std::endl;
        if (size_t(vars.size()) != numVars()) throw std::runtime_error("Unexpected number of variables passed to equilibrium updater");
        linesearchEvalPt = vars;
        equilibriumProblem.external_forces = forceVectorFromVars(vars);
        equilibriumOptimizer->optimize();

        linesearch_linkage.updateSourceFrame();
        linesearch_linkage.updateRotationParametrizations();

        // Use the final equilibrium's Hessians for sensitivity analysis, not the second-to-last iterate's
        equilibriumOptimizer->update_factorizations();

        // The cached adjoint state is invalidated whenever the equilibrium is updated...
        m_adjointStateIsCurrent    = false;
        m_autodiffLinkageIsCurrent = false;

        target_surface_fitter.updateClosestPoints(linesearch_linkage);
    }

    bool m_updateEquilibrium(const Eigen::VectorXd &vars) {
        if ((linesearchEvalPt - vars).norm() < 1e-16) return false;

        linesearch_linkage.set(linkage);
        m_forceEquilibriumUpdate(vars);
        return true;
    }

    bool m_updateAdjointState(const Eigen::VectorXd &vars) {
        m_updateEquilibrium(vars);
        if (m_adjointStateIsCurrent) return false;

        m_w = target_surface_fitter.adjoint_solve(linesearch_linkage, *equilibriumOptimizer);

        m_adjointStateIsCurrent = true;
        return true;
    }

    Eigen::VectorXd m_w;
    Eigen::VectorXd m_delta_x, m_delta_w;
    bool m_adjointStateIsCurrent = false,
         m_autodiffLinkageIsCurrent = false;

    const size_t m_ndof;
    const size_t m_numTorques;
    const Real m_l0 = 1.0;

    // Options controlling the sparsification strategy
    Regularization  m_regularization;
};

#endif /* end of include guard: ACTUATIONSPARSIFIER_HH */
