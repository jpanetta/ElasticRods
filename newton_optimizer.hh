////////////////////////////////////////////////////////////////////////////////
// newton_optimizer.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Newton-type optimization method for large, sparse problems.
//  This is Newton's method with a (sparse) Hessian modification strategy to
//  deal with the indefinite case.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  09/27/2018 11:29:48
////////////////////////////////////////////////////////////////////////////////
#ifndef NEWTON_OPTIMIZER_HH
#define NEWTON_OPTIMIZER_HH

#include <vector>
#include <cmath>
#include <MeshFEM/SparseMatrices.hh>
#include "SparseMatrixOps.hh"
#include "eigensolver.hh"
#include "ConvergenceReport.hh"

struct NewtonProblem {
    virtual void setVars(const Eigen::VectorXd &vars) = 0;
    virtual const Eigen::VectorXd getVars() const = 0;
    virtual size_t numVars() const = 0;

    // Called at the start of each new iteration (after line search has been performed)
    void iterationCallback(size_t i) {
        m_clearCache();
        m_iterationCallback(i);
    }

    virtual Real energy() const = 0;
    // freshIterate: whether the gradient is being called immediately
    // after an iteration callback (without any change to the variables in between) instead
    // of, e.g., during the line search.
    // For some problems, a less expensive gradient expression can be used in this case.
    virtual Eigen::VectorXd gradient(bool freshIterate = false) const = 0;
    const SuiteSparseMatrix &hessian() const {
        if (!m_cachedHessian) {
            m_cachedHessian = std::make_unique<SuiteSparseMatrix>(hessianSparsityPattern());
            m_evalHessian(*m_cachedHessian);
        }
        return *m_cachedHessian;
    }

    // Positive definite matrix defining the metric used to define trust regions.
    // For efficiency, it must have the same sparsity pattern as the Hessian.
    // (This matrix is added to indefinite Hessians to produce a positive definite modified Hessian.)
    const SuiteSparseMatrix &metric() const {
        if (m_useIdentityMetric) {
            if (!m_identityMetric) {
                m_identityMetric = std::make_unique<SuiteSparseMatrix>(hessianSparsityPattern());
                m_identityMetric->setIdentity(true);
            }
            return *m_identityMetric;
        }
        if (!m_cachedMetric) {
            m_cachedMetric = std::make_unique<SuiteSparseMatrix>(hessianSparsityPattern());
            m_evalMetric(*m_cachedMetric);
        }
        return *m_cachedMetric;
    }

    Real hessianL2Norm() const { return largestMagnitudeEigenvalue(hessian(), 1e-2); }

    // Since computing the L2 norm is slightly expensive, we assume that it remains
    // constant throughout the solve. This is exactly true for ElasticRods, and should be
    // a good approximation for RodLinkages under mild deformation.
    // Also, an exact result should not be necessary since it's only used to determine a reasonable
    // initial guess for the Hessian modification magnitude.
    Real metricL2Norm() const {
        if (m_useIdentityMetric) return 1.0;
        if (m_metricL2Norm <= 0) m_metricL2Norm = largestMagnitudeEigenvalue(metric(), 1e-2);
        return m_metricL2Norm;
    }
    void setUseIdentityMetric(bool useIdentityMetric) { m_useIdentityMetric = useIdentityMetric; }

    virtual SuiteSparseMatrix hessianCSC() const = 0;
    // A compressed column sparse matrix with nonzero placeholders wherever the Hessian can ever have nonzero entries.
    virtual SuiteSparseMatrix hessianSparsityPattern() const = 0;

    // sparsity pattern with fixed variable rows/cols removed.
    virtual SuiteSparseMatrix hessianReducedSparsityPattern() const {
        auto hsp = hessianSparsityPattern();
        hsp.fill(1.0);
        std::vector<char> isFixed(numVars(), false);
        for (size_t fv : m_fixedVars) isFixed.at(fv) = true;
        hsp.rowColRemoval([&isFixed] (size_t i) { return isFixed[i]; });
        return hsp;
    }

    const std::vector<size_t> &fixedVars() const { return m_fixedVars; }
    void setFixedVars(const std::vector<size_t> &fv) { m_fixedVars = fv; }
    size_t numFixedVars() const { return fixedVars().size(); }
    size_t numReducedVars() const { return numVars() - fixedVars().size(); } // number of remaining variables after fixing fixedVars
    void addFixedVariables(const std::vector<size_t> &fv) { m_fixedVars.insert(std::end(m_fixedVars), std::begin(fv), std::end(fv)); }

    virtual bool         hasLEQConstraint()       const { return false; }
    virtual Eigen::VectorXd LEQConstraintMatrix() const { return Eigen::VectorXd(); }
    virtual Real            LEQConstraintRHS()    const { return 0.0; }
    virtual void         setLEQConstraintRHS(Real)      { throw std::runtime_error("Problem doesn't apply a LEQ constraint."); }
    virtual Real            LEQConstraintTol()    const { return 1e-7; }
    virtual void            LEQStepFeasible()           { throw std::runtime_error("Problem type doesn't implement direct feasibility step."); }
    // r = b - Ax
    Real LEQConstraintResidual() const { return LEQConstraintRHS() - LEQConstraintMatrix().dot(getVars()); }
    bool LEQConstraintIsFeasible() const { return std::abs(LEQConstraintResidual()) <= LEQConstraintTol(); }

    virtual void writeIterateFiles(size_t it) const = 0;
    bool writeIterates = false;
    virtual void writeDebugFiles(const std::string &errorName) const = 0;

    NewtonProblem &operator=(const NewtonProblem &b) = delete;

    struct BoundConstraint {
        enum Type { LOWER, UPPER};
        size_t idx;
        Real val;
        Type type;

        BoundConstraint(size_t i, Real v, Type t) : idx(i), val(v), type(t) { }

        // To avoid numerical issues as iterates approach the bound constraints, a constraint
        // is considered active if the variable is within "tol" of the bound.
        bool active(const Eigen::VectorXd &vars, const Eigen::VectorXd &g, Real tol = 1e-8) const {
            return ((type == Type::LOWER) && (vars[idx] <= val + tol) && ((g.size() == 0) || (g[idx] <= 0)))
                || ((type == Type::UPPER) && (vars[idx] >= val - tol) && ((g.size() == 0) || (g[idx] >= 0)));
        }

        // Decide whether the bound constraint should be removed form the working set.
        // For the Lagrange multiplier estimate to be accurate, we require the gradient to be small.
        // (Since we're working with bound constraints, the first order Lagrange multiplier estimate is simply the gradient component)
        bool shouldRemoveFromWorkingSet(const Eigen::VectorXd &g, const Eigen::VectorXd &g_free) const {
            if (type == Type::LOWER) { return g[idx] >  10 * g_free.norm(); }
            if (type == Type::UPPER) { return g[idx] < -10 * g_free.norm(); }
            throw std::runtime_error("Unknown bound type");
        }

        bool feasible(const Eigen::VectorXd &vars) const {
            if (type == Type::LOWER) return vars[idx] >= val;
            else                     return vars[idx] <= val;
            throw std::runtime_error("Unknown bound type");
        }
        void apply(Eigen::VectorXd &vars) const {
            if ((type == Type::LOWER) && (vars[idx] < val)) vars[idx] = val;
            if ((type == Type::UPPER) && (vars[idx] > val)) vars[idx] = val;
        }
        Real feasibleStepLength(const Eigen::VectorXd &vars, const Eigen::VectorXd &step) const {
            Real alpha = std::numeric_limits<Real>::max();
            if      (type == Type::LOWER) { if (step[idx] < 0) alpha = (val - vars[idx]) / step[idx]; }
            else if (type == Type::UPPER) { if (step[idx] > 0) alpha = (val - vars[idx]) / step[idx]; }
            else throw std::runtime_error("Unknown bound type");
            // Note: alpha will be negative if "vars" are already infeasible and step is nonzero.
            // This should never happen assuming active constraints are detected/handled properly.
            if (alpha < 0) throw std::runtime_error("Feasible step is negative");
            return alpha;
        }

        void report(const Eigen::VectorXd &vars, const Eigen::VectorXd &g) const {
            std::cout << "\t" << ((type == Type::LOWER) ? "lower" : "upper") << " bd on var " << idx
                      << " (curr val:" << vars[idx] << ", bd: " << val << ", lagrange multiplier: " << g[idx] << ")" << std::endl;
        }
    };

    const std::vector<BoundConstraint> &boundConstraints() const { return m_boundConstraints; }
    size_t                           numBoundConstraints() const { return m_boundConstraints.size(); }
    const BoundConstraint &boundConstraint(size_t i) const { return m_boundConstraints[i]; }

    Eigen::VectorXd applyBoundConstraints(Eigen::VectorXd vars) const {
        for (auto &bc : m_boundConstraints) bc.apply(vars);
        return vars;
    }

    void applyBoundConstraintsInPlace(Eigen::VectorXd &vars) const {
        for (auto &bc : m_boundConstraints) bc.apply(vars);
    }

    std::vector<BoundConstraint> activeBoundConstraints(const Eigen::VectorXd &vars, const Eigen::VectorXd &g = Eigen::VectorXd(), Real tol = 1e-8) const {
        std::vector<BoundConstraint> result;
        for (auto &bc : m_boundConstraints) {
            if (bc.active(vars, g, tol)) result.push_back(bc);
        }
        return result;
    }

    bool feasible(const Eigen::VectorXd &vars) {
        for (auto &bc : boundConstraints())
            if (!bc.feasible(vars)) return false;
        return true;
    }

    // Get feasible step length and the index of the step-limiting bound
    std::pair<Real, size_t> feasibleStepLength(const Eigen::VectorXd &vars, const Eigen::VectorXd &step) const {
        Real alpha = std::numeric_limits<Real>::max();
        size_t blocking_idx = std::numeric_limits<size_t>::max();

        for (size_t i = 0; i < m_boundConstraints.size(); ++i) {
            Real len = m_boundConstraints[i].feasibleStepLength(vars, step);
            if (len < alpha) { alpha = len; blocking_idx = i; }
        }
        return std::make_pair(alpha, blocking_idx);
    }

    // Get feasible step length and the index of the step-limiting bound
    std::pair<Real, size_t> feasibleStepLength(const Eigen::VectorXd &step) const {
        return feasibleStepLength(getVars(), step);
    }

    // "Physical" distance of a step relative to some characteristic lengthscale of the problem.
    // (Used to determine reasonable step lengths to take when the Newton step is not possible.)
    virtual Real characteristicDistance(const Eigen::VectorXd &/* d */) const { return -1.0; }

    // Allow problems to attach custom convergence information to each optimization iterate.
    virtual void customIterateReport(ConvergenceReport &/* report */) const { }

    virtual ~NewtonProblem() { }

protected:
    // Clear the cached per-iterate quantities
    void m_clearCache() { m_cachedHessian.reset(), m_cachedMetric.reset(); /* TODO: decide if we want this: m_metricL2Norm = -1; */ }
    // Called at the start of each new iteration (after line search has been performed)
    virtual void m_iterationCallback(size_t /* i */) { }

    virtual void m_evalHessian(SuiteSparseMatrix &result) const = 0;
    virtual void m_evalMetric (SuiteSparseMatrix &result) const = 0;

    std::vector<BoundConstraint> m_boundConstraints;
    std::vector<size_t> m_fixedVars;

    bool m_useIdentityMetric = false;

    // Cached values for the mass matrix and its L2 norm
    // Mass matrix is recomputed each iteration; L2 norm is estimated only
    // once across the entire solve.
    mutable std::unique_ptr<SuiteSparseMatrix> m_cachedHessian, m_cachedMetric, m_identityMetric;
    mutable Real m_metricL2Norm = -1;
};

struct WorkingSet {
    WorkingSet(const NewtonProblem &problem) : m_prob(problem), m_contains(problem.numBoundConstraints(), false), m_varFixed(problem.numVars(), false) { }

    // Check whether the working set contains a particular constraint
    bool contains(size_t idx) const { return m_contains[idx]; }
    bool fixesVariable(size_t vidx) const { return m_varFixed[vidx]; }

    // Returns true if the index was actually newly added to the set.
    bool add(size_t idx) {
        if (contains(idx)) return false;

        const size_t vidx = m_prob.boundConstraint(idx).idx;
        if (m_varFixed[vidx]) throw std::runtime_error("Only one active bound on a variable is supported (don't impose equality constraints with bounds!)");

        m_varFixed[vidx] = true;
        m_contains[idx] = true;
        ++m_count;

        return true;
    }

    template<class Predicate>
    void remove_if(const Predicate &p) {
        const size_t nbc = m_contains.size();
        for (size_t bci = 0; bci < nbc; ++bci) {
            if (m_contains[bci] && p(bci)) {
                m_contains[bci] = false;
                const size_t vidx = m_prob.boundConstraint(bci).idx;
                assert(m_varFixed[vidx]);
                m_varFixed[vidx] = false;
                --m_count;
            }
        }
    }

    size_t size() const { return m_count; }

    // Zero out the components for variables fixed by the working set. E.g., if "g" is the gradient,
    // compute the gradient with respect to the "free" variables (without resizing)
    Eigen::VectorXd getFreeComponent(Eigen::VectorXd g /* copy modified inside */) const {
        if (size_t(g.size()) != m_varFixed.size()) throw std::runtime_error("Gradient size mismatch");
        for (size_t vidx = 0; vidx < m_varFixed.size(); ++vidx)
            if (m_varFixed[vidx]) g[vidx] = 0.0;
        return g;
    }

private:
    const NewtonProblem &m_prob;
    size_t m_count = 0;
    std::vector<char> m_contains; // Whether a particular constraint is in the working set
    std::vector<char> m_varFixed; // Whether a variable is fixed by one of the constraints in the working set
};

struct NewtonOptimizerOptions {
    Real gradTol = 2e-8,
         beta = 1e-8;
    bool hessianScaledBeta = true;
    size_t niter = 100;                        // Maximum number of newton iterations
    bool useIdentityMetric = false;            // Whether to force the use of the identity matrix for Hessian modification (instead of the problem's custom metric)
    bool useNegativeCurvatureDirection = true; // Whether to compute and move in negative curvature directions to escape from saddle points.
    bool feasibilitySolve = true;              // Whether to solve for a feasible starting point or rely on the problem to jump to feasible parameters.
    int verbose = 1;
    bool writeIterateFiles = false;
};

// Cache temporaries and solve the KKT system:
// [H   a][   x  ] = [   b    ]
// [a^T 0][lambda]   [residual]
struct KKTSolver {
    Eigen::VectorXd Hinv_a, a;
    template<class Factorizer>
    void update(Factorizer &solver, Eigen::Ref<const Eigen::VectorXd> a_) {
        a = a_;
        solver.solve(a, Hinv_a);
    }

    Real           lambda(Eigen::Ref<const Eigen::VectorXd> Hinv_b, const Real residual = 0) const { return (a.dot(Hinv_b) - residual) / a.dot(Hinv_a); }
    Eigen::VectorXd solve(Eigen::Ref<const Eigen::VectorXd> Hinv_b, const Real residual = 0) const { return Hinv_b - lambda(Hinv_b, residual) * Hinv_a; }

    template<class Factorizer>
    Eigen::VectorXd operator()(Factorizer &solver, Eigen::Ref<const Eigen::VectorXd> b, const Real residual = 0) const { return solve(solver, b, residual); }

    template<class Factorizer>
    Eigen::VectorXd solve(Factorizer &solver, Eigen::Ref<const Eigen::VectorXd> b, const Real residual = 0) const {
        Eigen::VectorXd Hinv_b;
        solver.solve(b.eval(), Hinv_b);
        return solve(Hinv_b, residual);
    }
};

// Cache to avoid repeated re-evaluation of our rough Hessian eigenvalue
// estimate. Uses the trace to detect when the Hessian's spectrum has changed
// substantially.
struct CachedHessianL2Norm {
    CachedHessianL2Norm() { reset(); }

    static constexpr double TRACE_TOL = 0.5;
    Real get(const NewtonProblem &p) {
        const auto &H = p.hessian();
        Real tr = H.trace();
        if (std::abs(tr - hessianTrace) > TRACE_TOL * std::abs(hessianTrace)) {
            hessianTrace = tr;
            hessianL2Norm = p.hessianL2Norm();
        }
        return hessianL2Norm;
    }

    void reset() { hessianTrace  = std::numeric_limits<Real>::max();
                   hessianL2Norm = 1.0; }
private:
    Real hessianTrace, hessianL2Norm;
};

struct NewtonOptimizer {
    NewtonOptimizer(std::unique_ptr<NewtonProblem> &&p) : solver(p->hessianReducedSparsityPattern()) {
        prob = std::move(p);
        solver.factorizeSymbolic();
        const std::vector<size_t> fixedVars = prob->fixedVars();
        isFixed.assign(prob->numVars(), false);
        for (size_t fv : fixedVars) isFixed[fv] = true;

    }
    ConvergenceReport optimize();

    Real newton_step(Eigen::VectorXd &step, /* copy modified inside */ Eigen::VectorXd g, const WorkingSet &ws, Real &beta, const Real betaMin, const bool feasibility = false);

    // Calculate a Newton step with empty working set and default beta/betaMin.
    Real newton_step(Eigen::VectorXd &step, const Eigen::VectorXd &g) {
        Real beta = options.beta;
        const Real betaMin = std::min(beta, 1e-6);
        WorkingSet ws(*prob);
        return newton_step(step, g, ws, beta, betaMin);
    }

    // Update the factorizations of the Hessian/KKT system with the current
    // iterate's Hessian. This is necessary for sensitivity analysis after
    // optimize() has been called: when optimization terminates either because
    // the problem is solved or the iteration limit is reached, solver/kkt_solver
    // hold values from the previous iteration (before the final linesearch
    // step).
    void update_factorizations() {
        // Computing a Newton step updates the Cholesky factorization in
        // "solver" and (if applicable) the kkt_solver as a side-effect.
        Eigen::VectorXd dummy;
        newton_step(dummy, Eigen::VectorXd::Zero(prob->numVars()));
    }

    Real tauScale() const { return (options.hessianScaledBeta ? m_cachedHessianL2Norm.get(*prob) : 1.0) / prob->metricL2Norm(); }

    const NewtonProblem &get_problem() const { return *prob; }
          NewtonProblem &get_problem()       { return *prob; }

    // Construct a vector of reduced components by removing the entries of "x" corresponding
    // to fixed variables. This is a (partial) inverse of extractFullSolution.
    void removeFixedEntriesInPlace(Eigen::VectorXd &x) const {
        int back = 0;
        for (int i = 0; i < x.size(); ++i)
            if (!isFixed[i]) x[back++] = x[i];
        x.conservativeResize(back);
    }
    Eigen::VectorXd removeFixedEntries(const Eigen::VectorXd &x) const {
        auto result = x;
        removeFixedEntriesInPlace(result);
        return result;
    }

    // Extract the full linear system solution vector "x" from the reduced linear
    // system solution "xReduced" (which was solved by removing the rows/columns for fixed variables).
    void extractFullSolution(const Eigen::VectorXd &xReduced, Eigen::VectorXd &x) const {
        int back = 0;
        for (int i = 0; i < x.size(); ++i) {
            if (!isFixed[i]) x[i] = xReduced[back++];
            else             x[i] = 0.0;
        }
        assert(back == xReduced.size());
    }

    Eigen::VectorXd extractFullSolution(const Eigen::VectorXd &xReduced) const {
        Eigen::VectorXd x(prob->numVars());
        extractFullSolution(xReduced, x);
        return x;
    }

    NewtonOptimizerOptions options;
    CholmodFactorizer solver;
    KKTSolver kkt_solver;
    // We fix variables by constraining the newton step to have zeros for these entries
    std::vector<char> isFixed;
    mutable CachedHessianL2Norm m_cachedHessianL2Norm;
private:
    std::unique_ptr<NewtonProblem> prob;
};

#endif /* end of include guard: NEWTON_OPTIMIZER_HH */
