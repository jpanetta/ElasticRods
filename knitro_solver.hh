#ifndef KNITRO_PROBLEM_HH
#define KNITRO_PROBLEM_HH

#if HAS_KNITRO

#include <KTRSolver.h>
#include <KTRProblem.h>
#include <MeshFEM/unused.hh>

template<typename Object>
struct KnitroEquilibriumProblem : public knitro::KTRProblem {
    // constructor: pass number of variables and constraints to base class
    KnitroEquilibriumProblem(Object &obj, bool doRestLenSolve = false, const std::vector<size_t> &fixedVars = std::vector<size_t>())
            : KTRProblem(doRestLenSolve ? obj.numExtendedDoF() : obj.numDoF(), 0 /* num constraints */, 0 /* jacobian size */, obj.hessianNNZ(doRestLenSolve)),
              object(obj), restLenSolve(doRestLenSolve),
              hessianSparsity(obj.hessianSparsityPattern(restLenSolve))
    {
        setObjType(KTR_OBJTYPE_GENERAL);
        setObjGoal(KTR_OBJGOAL_MINIMIZE);

        const size_t ndofs = restLenSolve ? obj.numExtendedDoF() : obj.numDoF();

        std::vector<double> loBounds(ndofs, -KTR_INFBOUND);
        std::vector<double> upBounds(ndofs,  KTR_INFBOUND);

        // We need lower bounds for the length variables
        const Real lenBound = 0.25 * obj.initialMinRestLength();
        for (size_t var : obj.lengthVars(restLenSolve)) {
            loBounds[var] = lenBound;
        }

        // Pin the fixed variables
        auto dofs = restLenSolve ? obj.getExtendedDoFs() : obj.getDoFs();
        for (size_t var : fixedVars)
            loBounds[var] = upBounds[var] = dofs[var];

        // Also pin the fixed variables required for the rest length solve.
        if (doRestLenSolve) {
            for (size_t var : object.restLenFixedVars())
                loBounds[var] = upBounds[var] = dofs[var];
        }

#if 0
        std::cout << "Set length lower bound to " << lenBound << std::endl;
        size_t varOnBoundCount = 0;
        for (size_t var = 0; var < ndofs; ++var) {
            if (dofs[var] < loBounds[var]) {
                std::cout << "Lower bound on variable " << var << " violated: " << dofs[var] << " < " << loBounds[var] << std::endl;
            }
            if (dofs[var] > upBounds[var]) {
                std::cout << "Upper bound on variable " << var << " violated: " << dofs[var] << " > " << upBounds[var] << std::endl;
            }
            if ((dofs[var] == loBounds[var]) || (dofs[var] == upBounds[var])) {
                std::cout << "Variable on bound " << var << ": " << dofs[var] << std::endl;
                ++varOnBoundCount;
            }
        }
        std::cout << "Variables on bounds: " << varOnBoundCount << std::endl;
#endif

        setVarLoBnds(loBounds);
        setVarUpBnds(upBounds);

        std::vector<int> hrows, hcols;
        std::vector<double> hvals;
        hessianSparsity.getIJV(hrows, hcols, hvals);

        setHessIndexCols(hcols);
        setHessIndexRows(hrows);
    }

    void setDoFs(const std::vector<double> &x) {
        if (restLenSolve) object.setExtendedDoFs(Eigen::Map<const Eigen::VectorXd>(x.data(), x.size()));
        else              object.setDoFs(x);
    }

    double evaluateFC(const std::vector<double> &x,
                            std::vector<double> &cval,
                            std::vector<double> &objGrad,
                            std::vector<double> &jac) override {
        assert(cval.size() == 0);
        assert(jac.size() == 0);
        UNUSED(cval);
        UNUSED(jac);

        setDoFs(x);
        double val = object.energy();
        auto g = object.gradient(false, ElasticRod::EnergyType::Full, restLenSolve);
        if (size_t(g.size()) != objGrad.size()) throw std::runtime_error("Unexpected gradient size");

        if (restLenSolve) {
            val += laplacianRegularizationWeight * object.restLengthLaplacianEnergy();
            g.segment(object.restLenOffset(), object.numRestLengths()) += laplacianRegularizationWeight * object.restLengthLaplacianGradEnergy();
        }

        // std::cout << "elastic energy: " << object.energy() << std::endl;
        // std::cout << "laplacian energy: " << object.restLengthLaplacianEnergy() << std::endl;
        // std::cout << "objective: " << val << std::endl;
        // std::cout << "Evaluated " << (restLenSolve ? "rest length" : "equilibrium") << " objective at:" << std::endl;
        // std::cout << Eigen::Map<const Eigen::VectorXd>(x.data(), x.size()).transpose() << std::endl;


        for (size_t i = 0; i < objGrad.size(); ++i)
            objGrad[i] = g[i];

        return val;
    }

    // Gradient is evaluated in evaluateFC
    int evaluateGA(const std::vector<double> &x, std::vector<double> &objGrad, std::vector<double>&jac) override {
        // According to Knitro's documentation, it suffices to simply implement
        // evaluateFC, but this isn't working for some reason...
        std::vector<double> cval;
        evaluateFC(x, cval, objGrad, jac);
        return 0;
    }

    int evaluateHess(const std::vector<double>& x, double objScalar, const std::vector<double>& /* lambda */,
                     std::vector<double>& hess) override {
        // Note: Knitro gives us the lagrange multipliers for the bound/fixed constraints in "lambda"

        assert(objScalar == 1.0);
        UNUSED(objScalar);
        assert(hess.size() == size_t(hessianSparsity.nz));

        // Should actually have been called by KnitroEquilibriumProblemNewPtCallback...
        // (which also updates the source frame so that the Hessian formula will be accurate.)
        setDoFs(x);

        hessianSparsity.setZero();
        object.hessian(hessianSparsity, ElasticRod::EnergyType::Full, restLenSolve);

        if (restLenSolve) {
            // Add in the laplacian regularization term
            auto L = object.restLengthLaplacianHessEnergy();
            const size_t offset = object.restLenOffset();
            for (auto &t : L.nz)
                hessianSparsity.addNZ(t.i + offset, t.j + offset, laplacianRegularizationWeight * t.v);
        }

        hess = hessianSparsity.Ax;

        return 0;
    }

    Object &object;
    const bool restLenSolve;
    Real laplacianRegularizationWeight = 0.0;
    SuiteSparseMatrix hessianSparsity;
};

template<class Object>
struct KnitroEquilibriumProblemNewPtCallback : public knitro::KTRNewptCallback {
    using EP = KnitroEquilibriumProblem<Object>;
    KnitroEquilibriumProblemNewPtCallback(EP &prob) : m_prob(prob) { }

    virtual int CallbackFunction(const std::vector<double> &x, const std::vector<double>& /* lambda */, double /* obj */,
            const std::vector<double>& /* c */, const std::vector<double>& /* objGrad */,
            const std::vector<double>& /* jac */, knitro::KTRISolver * /* solver */) override {
        m_prob.setDoFs(x);
        m_prob.object.updateSourceFrame();
        return 0;
    }

private:
    EP &m_prob;
};

// Note: initial parameters must already be set in the problem!
// (using problem.setXInitial()).
template<class Object>
void optimize_knitro_ip(KnitroEquilibriumProblem<Object> &problem, const size_t maxiter, Real gradTol)
{
    KnitroEquilibriumProblemNewPtCallback<Object> callback(problem);
    problem.setNewPointCallback(&callback);
    // Create a solver - optional arguments:
    // exact first and second derivatives; no KTR_GRADOPT_* or KTR_HESSOPT_* parameter is needed.
    knitro::KTRSolver solver(&problem);
    solver.useNewptCallback();
    solver.setParam(KTR_PARAM_HONORBNDS, KTR_HONORBNDS_ALWAYS); // always respect bounds during optimization
    solver.setParam(KTR_PARAM_MAXIT, int(maxiter));
    solver.setParam(KTR_PARAM_PRESOLVE, KTR_PRESOLVE_NONE);
    solver.setParam(KTR_PARAM_ALGORITHM, KTR_ALG_BAR_DIRECT);   // interior point with exact Hessian
    solver.setParam(KTR_PARAM_PAR_NUMTHREADS, 12);
    // solver.setParam(KTR_PARAM_LINSOLVER, KTR_LINSOLVER_MKLPARDISO);
    // solver.setParam(KTR_PARAM_ALGORITHM, KTR_ALG_IPDIRECT);
    // solver.setParam(KTR_PARAM_BAR_FEASIBLE, KTR_BAR_FEASIBLE_NO);

    solver.setParam(KTR_PARAM_OPTTOL, gradTol);

    try {
        int solveStatus = solver.solve();

        if (solveStatus != 0) {
            std::cout << std::endl;
            std::cout << "KNITRO failed to solve the problem, final status = ";
            std::cout << solveStatus << std::endl;
        }
    }
    catch (knitro::KTRException &e) {
        problem.setNewPointCallback(nullptr);
        e.printMessage();
        throw e;
    }
    problem.setNewPointCallback(nullptr);
}

template<class Object>
void knitro_compute_equilibrium(Object &obj, const size_t maxiter, const std::vector<size_t> &fixedVars = std::vector<size_t>(), Real gradTol = 2e-8) {
    KnitroEquilibriumProblem<Object> problem(obj, false, fixedVars);
    
    auto dofs = obj.getDoFs();
    for (size_t p = 0; p < size_t(dofs.size()); ++p)
        problem.setXInitial(p, dofs[p]);
    optimize_knitro_ip(problem, maxiter, gradTol);
}

template<class Object>
void knitro_restlen_solve(Object &obj, Real laplacianRegWeight, const size_t maxiter, const std::vector<size_t> &fixedVars = std::vector<size_t>(), Real gradTol = 2e-8) {
    KnitroEquilibriumProblem<Object> problem(obj, true, fixedVars);
    problem.laplacianRegularizationWeight = laplacianRegWeight;
    
    auto dofs = obj.getExtendedDoFs();
    for (size_t p = 0; p < size_t(dofs.size()); ++p)
        problem.setXInitial(p, dofs[p]);
    optimize_knitro_ip(problem, maxiter, gradTol);
}


#endif // !HAS_KNITRO

#endif /* end of include guard: KNITRO_PROBLEM_HH */
