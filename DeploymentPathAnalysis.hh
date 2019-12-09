#ifndef DEPLOYMENTPATHANALYSIS_HH
#define DEPLOYMENTPATHANALYSIS_HH

#include "compute_equilibrium.hh"

struct DeploymentEnergyIncrement {
    Real linearTerm, quadraticTerm;
    Real operator()(Real deltaDeployAmount) const {
        return (linearTerm + quadraticTerm * deltaDeployAmount) * deltaDeployAmount;
    }
};

struct DeploymentPathAnalysis {
    DeploymentPathAnalysis(NewtonOptimizer &opt) {
        m_calculate(opt);
    }

    DeploymentPathAnalysis(RodLinkage &linkage, const std::vector<size_t> &fixedVars) {
        auto opt = get_equilibrium_optimizer(linkage, linkage.getAverageJointAngle(), fixedVars);
        m_calculate(*opt);
    }

    Eigen::VectorXd deploymentStep, secondBestDeploymentStep;
    Real relativeStiffnessGap;
    DeploymentEnergyIncrement bestEnergyIncrement,
                              secondBestEnergyIncrement;
private:
    using LinkageEQP = AverageAngleConstrainedEquilibriumProblem<RodLinkage>;

    auto m_grad_deployment_amount(const LinkageEQP &eqp) {
        return eqp.LEQConstraintMatrix();
    }

    void m_calculate(NewtonOptimizer &opt) {
        using VXd = RodLinkage::VecX;
        using V2d = RodLinkage::Vec2;
        using M2d = Eigen::Matrix2d;
        const auto &eqp = dynamic_cast<const LinkageEQP &>(opt.get_problem());
        VXd a = m_grad_deployment_amount(eqp);

        auto H_reduced = eqp.hessian();
        auto g = eqp.gradient();
        H_reduced.rowColRemoval([&](SuiteSparse_long i) { return opt.isFixed[i]; });
        VXd g_reduced = opt.removeFixedEntries(g);
        VXd a_reduced = opt.removeFixedEntries(a);

        auto &solver = opt.solver;
        solver.updateFactorization(H_reduced); VXd Hinv_a = opt.extractFullSolution(solver.solve(a_reduced));
        // Compute displacement step producing (linearized) unit deployment increment
        Real a_Hinv_a = a.dot(Hinv_a);
        deploymentStep = Hinv_a / a_Hinv_a;

        auto M = eqp.metric();
        VXd Md1 = M.apply(deploymentStep);
        Real a_Hinv_Md1 = Hinv_a.dot(Md1);

        VXd Md1_reduced = opt.removeFixedEntries(Md1);
        VXd Hinv_Md1 = opt.extractFullSolution(solver.solve(Md1_reduced));
        Real Md1_Hinv_Md1 = Md1.dot(Hinv_Md1);

        M2d A;
        A << a_Hinv_a, a_Hinv_Md1,
             a_Hinv_Md1, Md1_Hinv_Md1;

        V2d neg_lagrange_multipliers = A.inverse().col(0);

        secondBestDeploymentStep = Hinv_a   * neg_lagrange_multipliers[0] +
                                   Hinv_Md1 * neg_lagrange_multipliers[1];

        auto H = eqp.hessian();
        bestEnergyIncrement.linearTerm    = deploymentStep.dot(g);
        bestEnergyIncrement.quadraticTerm = 0.5 * deploymentStep.dot(H.apply(deploymentStep));

        secondBestEnergyIncrement.linearTerm    = secondBestDeploymentStep.dot(g);
        secondBestEnergyIncrement.quadraticTerm = 0.5 * secondBestDeploymentStep.dot(H.apply(secondBestDeploymentStep));

        relativeStiffnessGap = 1.0 / (1.0 - (a_Hinv_Md1 * a_Hinv_Md1) / (a_Hinv_a * Md1_Hinv_Md1)) - 1.0;
    }
};

#endif /* end of include guard: DEPLOYMENTPATHANALYSIS_HH */
