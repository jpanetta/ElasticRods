#ifndef CG_SOLVER_HH
#define CG_SOLVER_HH

#include <iostream>
#include "TemplatedTypes.hh"

template<bool NoTermination = false, class MatVec, class IterationCallback, typename Real_>
size_t cg_solver(const MatVec &apply_A, const VecX_T<Real_> &b, VecX_T<Real_> &x, const IterationCallback &cb, const size_t niters = 50, const double eps = 1e-6) {
    auto r = (b - apply_A(x)).eval();
    auto p = r;
    size_t k = 0;
    Real_ r_dot_r = r.squaredNorm();
    while (k < niters) {
        cb(k, r);
        auto Ap = apply_A(p);
        Real_ p_dot_Ap = p.dot(Ap);

        if (p_dot_Ap <= 0) {
            std::cerr << "Direction of negative curvature detected in CG" << std::endl;
            if (k == 0) { x = r; }
            if (!NoTermination) return k;
        }

        Real_ alpha = r_dot_r / p_dot_Ap;

        // std::cout << "\tDirectional derivative: " << -p.dot(r) << ", step size: " << alpha << std::endl;
        // Real_ alpha_rnorm = Ap.dot(r) / Ap.squaredNorm();
        // std::cout << "\tDirectional derivative of residual norm: " << -Ap.dot(r) << ", step size: " << alpha_rnorm << std::endl;

        // auto x_residual_norm = (x + alpha_rnorm * p).eval();
        x += alpha * p;
        r -= alpha * Ap;
        // std::cout << "\tA vs A^T A step residual norm: " << r.norm() << ", " << (b - apply_A(x_residual_norm)).norm() << std::endl;;
        // std::cout << "\tquadratic objective: " << 0.5 * x.dot(apply_A(x)) - b.dot(x) << std::endl;

        Real_ r_dot_r_new = r.squaredNorm();
        if ((r_dot_r_new < eps * eps) && !NoTermination) return k;
        p = r + (r_dot_r_new / r_dot_r) * p;
        r_dot_r = r_dot_r_new;

        ++k;
    }
    return k;
}

template<bool NoTermination = false, class MatVec, typename Real_>
size_t cg_solver(const MatVec &apply_A, const VecX_T<Real_> &b, VecX_T<Real_> &x, const size_t niters = 50, const double eps = 1e-6) {
    return cg_solver(apply_A, b, x, [&](size_t /* k */, const VecX_T<Real_> &/* r */) { }, niters, eps);
}

#endif /* end of include guard: CG_SOLVER_HH */
