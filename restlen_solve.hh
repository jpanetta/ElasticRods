#ifndef RESTLEN_SOLVE_HH
#define RESTLEN_SOLVE_HH

#include <vector>
#include <cmath>
#include <memory>

#include <MeshFEM/newton_optimizer/newton_optimizer.hh>
#include "compute_equilibrium.hh"

// Solve an equilibrium problem augemented with rest length variables.
template<typename Object>
struct RestLengthProblem : public NewtonProblem {
    RestLengthProblem(Object &obj)
        : object(obj), m_hessianSparsity(obj.restlenSolveHessianSparsityPattern()),
          m_characteristicLength(obj.characteristicLength())
    {
        const Real initMinRestLen = obj.initialMinRestLength(); 
        auto lengthVars = obj.restlenSolveLengthVars();
        m_boundConstraints.reserve(lengthVars.size());
        // Make sure length variables aren't shrunk down to zero/inverted
        for (size_t var : lengthVars)
            m_boundConstraints.emplace_back(var, 0.01 * initMinRestLen, BoundConstraint::Type::LOWER);
        setFixedVars(obj.restLenFixedVars());
    }

    virtual void setVars(const Eigen::VectorXd &vars) override { object.setRestlenSolveDoF(vars); }
    virtual const Eigen::VectorXd getVars() const override { return object.getRestlenSolveDoF(); }
    virtual size_t numVars() const override { return object.numRestlenSolveDof(); }

    virtual Real energy() const override {
        return object.energy();
    }

    virtual Eigen::VectorXd gradient(bool freshIterate = false) const override {
        return object.restlenSolveGradient(freshIterate, ElasticRod::EnergyType::Full);
    }

    virtual SuiteSparseMatrix hessianSparsityPattern() const override { return m_hessianSparsity; }

    virtual void writeIterateFiles(size_t it) const override { if (writeIterates) { ::writeIterateFiles(object, it); } }

    virtual void writeDebugFiles(const std::string &errorName) const override {
        // auto M = metric();
        auto H = object.hessian();
        H.rowColRemoval(fixedVars());
        H.reflectUpperTriangle();
        // M.rowColRemoval(fixedVars());
        // M.reflectUpperTriangle();
        // M.dumpBinary("debug_" + errorName + "_mass.mat");
        H.dumpBinary("debug_" + errorName + "_hessian.mat");
        objectSpecificDebugFiles(object, errorName);
        object.saveVisualizationGeometry("debug_" + errorName + "_geometry.msh");
    }


private:
    virtual void m_iterationCallback(size_t /* i */) override { object.updateSourceFrame(); }

    virtual void m_evalHessian(SuiteSparseMatrix &result, bool /* projectionMask */) const override {
        result.setZero();
        object.restlenSolveHessian(result, ElasticRod::EnergyType::Full);
    }

    virtual void m_evalMetric(SuiteSparseMatrix &result) const override {
        result.setZero();
        object.massMatrix(result);
        const size_t rlo = object.restLenOffset(), nrl = object.numRestlenSolveRestLengths();
        for (size_t j = 0; j < nrl; ++j) {
            result.addNZ(result.findDiagEntry(rlo + j), m_characteristicLength);
            // TODO: figure out a more sensible mass to use for rest length variables.
            // Initial mass of each segment?
        }
    }

    Object &object;
    mutable SuiteSparseMatrix m_hessianSparsity;
    Real m_characteristicLength = 1.0;
};

template<typename Object>
std::unique_ptr<RestLengthProblem<Object>> restlen_problem(Object &obj, const std::vector<size_t> &fixedVars = std::vector<size_t>()) {
    auto problem = std::make_unique<RestLengthProblem<Object>>(obj);

    // Also fix the variables specified by the user.
    problem->addFixedVariables(fixedVars);
    return problem;
}

template<typename Object>
std::unique_ptr<NewtonOptimizer> get_restlen_optimizer(Object &obj, const std::vector<size_t> &fixedVars = std::vector<size_t>()) {
    auto problem = restlen_problem(obj, fixedVars);
    return std::make_unique<NewtonOptimizer>(std::move(problem));
}

// Rest length solve with custom optimizer options.
template<typename Object>
ConvergenceReport restlen_solve(Object &obj, const NewtonOptimizerOptions &opts, const std::vector<size_t> &fixedVars = std::vector<size_t>()) {
    auto opt = get_restlen_optimizer(obj, fixedVars);
    opt->options = opts;
    return opt->optimize();
}

// Default options for rest length solve: use the identity metric.
template<typename Object>
ConvergenceReport restlen_solve(Object &obj, const std::vector<size_t> &fixedVars = std::vector<size_t>()) {
    NewtonOptimizerOptions opts;
    opts.useIdentityMetric = true;
    return restlen_solve(obj, opts, fixedVars);
}

#endif /* end of include guard: RESTLEN_SOLVE_HH */
