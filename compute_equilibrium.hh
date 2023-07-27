#ifndef COMPUTE_EQUILIBRIUM_HH
#define COMPUTE_EQUILIBRIUM_HH

#include <vector>
#include <cmath>
#include <memory>
#include <limits>
#include <MeshFEM/Geometry.hh>
#include <MeshFEM/newton_optimizer/newton_optimizer.hh>

template<class Object>
void writeIterateFiles(Object &obj, size_t i) {
    obj.saveVisualizationGeometry("newton_it_" + std::to_string(i) + ".msh");
    // linkage.writeRodDebugData("newton_it_" + std::to_string(i) + "_rod_data.msh");
    // linkage.writeLinkageDebugData("newton_it_" + std::to_string(i) + "_linkage_data.msh");
}

inline void writeIterateFiles(ElasticRod &rod, size_t i) {
    rod.saveVisualizationGeometry("newton_it_" + std::to_string(i) + ".msh");
}

template<class Object>
auto objectSpecificDebugFiles(Object &obj, const std::string &errorName) {
    obj.saveVisualizationGeometry("debug_" + errorName + "_geometry.msh");
    obj.        writeRodDebugData("debug_" + errorName + "_rod_data.msh");
    // obj.writeLinkageDebugData("debug_linkage_data.msh");
}

inline void objectSpecificDebugFiles(ElasticRod &rod, const std::string &errorName) { rod.saveVisualizationGeometry("debug_" + errorName + "_geometry.msh"); }

template<typename Object>
struct EquilibriumProblem : NewtonProblem {
    EquilibriumProblem(Object &obj) : object(obj), m_hessianSparsity(obj.hessianSparsityPattern()), m_characteristicLength(obj.characteristicLength()) {
        // Make sure joint edge lengths aren't shrunk down to zero/inverted
        const Real val = 0.01 * obj.initialMinRestLength();
        auto lengthVars = obj.lengthVars(false /* rest lengths are fixed */);
        m_boundConstraints.reserve(lengthVars.size());
        for (size_t var : lengthVars)
            m_boundConstraints.emplace_back(var, val, BoundConstraint::Type::LOWER);
    }

    virtual void setVars(const Eigen::VectorXd &vars) override { object.setDoFs(vars); }
    virtual const Eigen::VectorXd getVars() const override { return object.getDoFs(); }
    virtual size_t numVars() const override { return object.numDoF(); }

    virtual Real energy() const override { return object.energy() + externalPotentialEnergy(); }

    virtual Eigen::VectorXd gradient(bool freshIterate = false) const override {
        Eigen::VectorXd result = object.gradient(freshIterate);
        // Add in the gradient of the external potential energy.
        if (external_forces.size() == 0) return result;
        if (external_forces.size() != result.size()) throw std::runtime_error("Invalid external force vector");
        result -= external_forces;
        return result;
    }

    // Potential energy stored in the externally applied force field.
    Real externalPotentialEnergy() const {
        if (external_forces.size() == 0) return 0.0;
        auto x = object.getDoFs();
        if (external_forces.size() != x.size()) throw std::runtime_error("Invalid external force vector");
        return -external_forces.dot(x);
    }

    virtual SuiteSparseMatrix hessianSparsityPattern() const override { /* m_hessianSparsity.fill(1.0); */ return m_hessianSparsity; }

    // "Physical" distance of a step relative to some characteristic lengthscale of the problem.
    // (Useful for determining reasonable step lengths to take when the Newton step is not possible.)
    virtual Real characteristicDistance(const Eigen::VectorXd &d) const override {
        // std::cout << "object.approxLinfVelocity(d): " << object.approxLinfVelocity(d) << std::endl;
        // std::cout << "m_characteristicLength: " << m_characteristicLength << std::endl;
        return object.approxLinfVelocity(d) / m_characteristicLength;
    }

    virtual void writeIterateFiles(size_t it) const override { if (writeIterates) { ::writeIterateFiles(object, it); } }

    virtual void writeDebugFiles(const std::string &errorName) const override {
        auto M = object.massMatrix();
        auto H = object.hessian();
        H.rowColRemoval(fixedVars());
        H.reflectUpperTriangle();
        M.rowColRemoval(fixedVars());
        M.reflectUpperTriangle();
        M.dumpBinary("debug_" + errorName + "_mass.mat");
        H.dumpBinary("debug_" + errorName + "_hessian.mat");
        objectSpecificDebugFiles(object, errorName);
        object.saveVisualizationGeometry("debug_" + errorName + "_geometry.msh");
    }

    virtual void customIterateReport(ConvergenceReport &report) const override {
        std::map<std::string, Real> data = {{"energy_bend",    object.energyBend()},
                                            {"energy_stretch", object.energyStretch()},
                                            {"energy_twist",   object.energyTwist()}};
        report.addCustomData(data);
    }

    // The external generalized forces acting on each degree of freedom in the linkage.
    // For position variables, these are true forces, while for other degrees of freedom
    // these act as a one-form computing the work done by a perturbation to the
    // degrees of freedom.
    // When this vector is empty, no forces are applied.
    // These forces can be used to apply gravity, custom actuation torques, or any
    // other loading scenario.
    Eigen::VectorXd external_forces;

protected:
    virtual void m_iterationCallback(size_t /* i */) override { object.updateSourceFrame(); object.updateRotationParametrizations(); }
    virtual void m_evalHessian(SuiteSparseMatrix &result, bool /* projectionMask */) const override {
        result.setZero();
        object.hessian(result);
    }
    virtual void m_evalMetric(SuiteSparseMatrix &result) const override {
        result.setZero();
        object.massMatrix(result, /* updated source; evaluated at the same time as the Hessian */ true, /* useLumped = */ true);
    }

    Object &object;
    mutable SuiteSparseMatrix m_hessianSparsity;
    Real m_characteristicLength = 1.0;
};

// Version of the equilibrium problem with a constraint imposing an average
// joint opening angle across the entire linkage.
template<typename Object>
struct AverageAngleConstrainedEquilibriumProblem : public EquilibriumProblem<Object> {
    using Base = EquilibriumProblem<Object>;
    using Base::Base;
    virtual void setLEQConstraintRHS(Real targetAngle) override { if (!hasLEQConstraint()) throw std::runtime_error("No constraint to configure."); m_targetAngle = targetAngle; }
    virtual bool hasLEQConstraint() const override { return true; }
    virtual Eigen::VectorXd LEQConstraintMatrix() const override {
        Eigen::VectorXd result(Eigen::VectorXd::Zero(this->numVars()));
        const auto jointAngles = this->object.jointAngleDoFIndices();
        Real val = 1.0 / jointAngles.size();
        for (size_t var : jointAngles)
            result[var] = val;
        return result;
    }

    virtual Real LEQConstraintRHS() const override { return m_targetAngle; }
    // Naively modify the opening angles to satisfy the LEQ constraints; just scale them uniformly.
    virtual void LEQStepFeasible() override {
        this->object.setAverageJointAngle(m_targetAngle);
        // The updated edge vectors at the joints must be applied to the linkage's rods...
        this->setVars(this->getVars());
    }

private:
    Real m_targetAngle = 0;
};

constexpr double TARGET_ANGLE_NONE = std::numeric_limits<Real>::max();

template<typename Object>
std::unique_ptr<EquilibriumProblem<Object>> equilibrium_problem(Object &obj, Real targetAverageAngle, const std::vector<size_t> &fixedVars = std::vector<size_t>()) {
    std::unique_ptr<EquilibriumProblem<Object>> problem;
    if (targetAverageAngle != TARGET_ANGLE_NONE) {
        problem = std::make_unique<AverageAngleConstrainedEquilibriumProblem<Object>>(obj);
        problem->setLEQConstraintRHS(targetAverageAngle);
    }
    else {
        problem = std::make_unique<EquilibriumProblem<Object>>(obj);
    }

    // Also fix the variables specified by the user.
    problem->addFixedVariables(fixedVars);
    return problem;
}

template<typename Object>
std::unique_ptr<NewtonOptimizer> get_equilibrium_optimizer(Object &obj, Real targetAverageAngle, const std::vector<size_t> &fixedVars = std::vector<size_t>()) {
    auto problem = equilibrium_problem(obj, targetAverageAngle, fixedVars);
    return std::make_unique<NewtonOptimizer>(std::move(problem));
}

// Target angle version
template<typename Object>
ConvergenceReport
compute_equilibrium(Object &obj, Real targetAverageAngle, const NewtonOptimizerOptions &opts = NewtonOptimizerOptions(), const std::vector<size_t> &fixedVars = std::vector<size_t>()) {
    auto opt = get_equilibrium_optimizer(obj, targetAverageAngle, fixedVars);
    opt->options = opts;
    return opt->optimize();
}

// No target angle angle version
template<typename Object>
ConvergenceReport
compute_equilibrium(Object &obj, const NewtonOptimizerOptions &opts = NewtonOptimizerOptions(), const std::vector<size_t> &fixedVars = std::vector<size_t>()) {
    return compute_equilibrium(obj, TARGET_ANGLE_NONE, opts, fixedVars);
}

// External forces version
template<typename Object>
ConvergenceReport
compute_equilibrium(Object &obj, const Eigen::VectorXd &externalForces, const NewtonOptimizerOptions &opts = NewtonOptimizerOptions(), const std::vector<size_t> &fixedVars = std::vector<size_t>()) {
    auto problem = equilibrium_problem(obj, TARGET_ANGLE_NONE, fixedVars);
    problem->external_forces = externalForces;
    NewtonOptimizer opt(std::move(problem));
    opt.options = opts;
    return opt.optimize();
}

#endif /* end of include guard: COMPUTE_EQUILIBRIUM_HH */
