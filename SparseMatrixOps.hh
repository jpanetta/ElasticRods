#ifndef SPARSEMATRIXOPS_HH
#define SPARSEMATRIXOPS_HH

#include "TriDiagonalSystem.hh"
#include <MeshFEM/SparseMatrices.hh>
#include <Eigen/Dense>

template<typename Real_>
TripletMatrix<Triplet<Real_>> triplet_matrix_from_tridiagonal(const TriDiagonalSystem<Real_> &A) {
    const size_t n = A.rows();
    using TM = TripletMatrix<Triplet<Real_>>;
    TM result(n, n);

    bool symmetric = true;
    for (size_t i = 0; i < n - 1; ++i) {
        if (A.upperDiagonal()[i] != A.lowerDiagonal()[i]) {
            symmetric = false;
            break;
        }
    }
    if (symmetric)
        result.symmetry_mode = TM::SymmetryMode::UPPER_TRIANGLE;

    result.reserve(symmetric ? n + (n - 1) : (n + 2 * (n - 1)));
    for (size_t i = 0; i < n; ++i)
        result.addNZ(i, i, A.diagonal()[i]);
    for (size_t i = 0; i < n - 1; ++i) {
        result.addNZ(i, i + 1, A.upperDiagonal()[i]);
        if (!symmetric)
            result.addNZ(i + 1, i, A.lowerDiagonal()[i]);
    }

    return result;
}

template<typename Derived>
TripletMatrix<Triplet<typename Derived::Scalar>> triplet_matrix_from_diag(const Eigen::MatrixBase<Derived> &d) {
    const size_t n = d.rows();
    TripletMatrix<Triplet<typename Derived::Scalar>> result(n, n);
    result.reserve(n);
    for (size_t i = 0; i < n; ++i)
        result.addNZ(i, i, d[i]);
    return result;
}

#endif /* end of include guard: SPARSEMATRIXOPS_HH */
