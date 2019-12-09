////////////////////////////////////////////////////////////////////////////////
// TriDiagonalSystem.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  O(n) solution for tridiagonal systems of linear equations (permitting
//  constraints that fix certain variables to specified values).
//
//  A tridiagonal matrix is represented by three vectors, the diagonal 'd' and
//  off-diagonals 'a' and 'c':
//          [ d_0 c_0                                        ]
//          [ a_0 d_1 c_1                                    ]
//          [     a_1 d_2 c_2                                ]
//      A = [             . . .                              ]
//          [               . . .                            ]
//          [                   a_{n - 3} d_{n - 2} c_{n - 2}]
//          [                             a_{n - 2} d_{n - 1}]
//
//  The system "Ax = b" can be placed in upper-triangular form by a single
//  forward sweep:
//
//          [  1  c'_0                          ][    x_0    ]   [    b'_0    ]
//          [      1  c'_1                      ][    x_1    ]   [    b'_1    ]
//          [          1  c'_2                  ][    x_2    ]   [    b'_3    ]
//          [             . . .                 ][     .     ] = [     .      ]
//          [               . . .               ][     .     ]   [     .      ]
//          [                      1  c'_{n - 2}][ x_{n - 2} ]   [ b'_{n - 2} ]
//          [                              1    ][ x_{n - 1} ]   [ b'_{n - 1} ]
//  where c'_i = c_i / den[i]
//        b'_i = (b_i - a_{i - 1} b'_{i - 1}) / den[i]      (forward substitution)
//        den[i]  = d_i - a_{i - 1} c'_{i - 1}
//  (negatively-indexed quantities evaluate to zero).
//  This transformed system can then be solved with back substitution:
//      x[n - 1] = b'[n - 1]
//      x[i] = b'[i] - c'[i + 1] * x[i + 1]
//  Fixing variable 'i' to 'val' is implemented by replacing the ith row with
//  the ith row of the identity matrix and replacing the ith entry of the rhs
//  with 'val 
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  03/29/2018 17:53:43
////////////////////////////////////////////////////////////////////////////////
#ifndef TRIDIAGONALSYSTEM_HH
#define TRIDIAGONALSYSTEM_HH
#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>

template<typename Real_>
class TriDiagonalSystem {
public:
    TriDiagonalSystem(std::vector<Real_> &&a, std::vector<Real_> &&d, std::vector<Real_> &&c)
        : m_a(std::move(a)), m_d(std::move(d)), m_c(std::move(c)) { m_init(); }

    TriDiagonalSystem(const std::vector<Real_> &a, const std::vector<Real_> &d, const std::vector<Real_> &c)
        : m_a(a), m_d(d), m_c(c) { m_init(); }

    size_t rows() const { return m_d.size(); }
    size_t cols() const { return m_d.size(); }
    size_t nnz()  const { size_t n =  rows(); return n + 2 * (n - 1); }

    void fixVariable(size_t idx, Real_ val) {
        if  (m_isFixed.at(idx)) std::cerr << "WARNING: changed value for constrained variable " << idx << std::endl;
        else m_isFixed[idx] = true;
        m_fixedVarValues[idx] = val;
        m_factorization.reset();
    }

    void freeVariable(size_t idx) {
        if (!m_isFixed.at(idx))
            std::cerr << "WARNING: Variable " << idx << " was not fixed." << std::endl;
        m_isFixed[idx] = false;
        m_factorization.reset();
    }

    std::vector<Real_> solve(std::vector<Real_> rhs) const {
        const size_t n = rows();
        factorize();

        const auto &iden = m_factorization->inv_den;
        const auto &cp   = m_factorization->cPrime;

        // Forward-solve
        if (m_isFixed[0]) rhs[0] = m_fixedVarValues[0];
        else              rhs[0] *= iden[0];
        for (size_t i = 1; i < n; ++i) {
            if (m_isFixed[i]) rhs[i] = m_fixedVarValues[i];
            else              rhs[i] = (rhs[i] - m_a[i - 1] * rhs[i - 1]) * iden[i];
        }

        // Back-solve
        std::vector<Real_> x(n);
        x[n - 1] = rhs[n - 1];
        for (int i = int(n) - 2; i >= 0; --i)
            x[i] = rhs[i] - cp[i] * x[i + 1];

        return x;
    }

    // Cache quantities needed for forward/back substitution
    void factorize() const {
        const size_t n = rows();
        if (n == 0) throw std::runtime_error("Empty system");

        if (m_factorization) return;

        m_factorization = std::make_unique<Factorization>();
        auto &invden = m_factorization->inv_den;
        auto &cp     = m_factorization->cPrime;

        invden.resize(n);
        cp.resize(n - 1);

        // Cache the 'factorization' quantities
        //  c'_i = c_i / den[i]
        //  den[i]  = d_i - a_{i - 1} c'_{i - 1}
        invden[0] = m_isFixed[0] ? Real_(1.0) : 1.0 / m_d[0];
        if (n == 1) return;
        cp[0]     = m_isFixed[0] ? Real_(0.0) : m_c[0] * invden[0];
        for (size_t i = 1; i < n; ++i) {
            if (m_isFixed[i]) {
                invden[i] = 1.0;
                if (i < (n - 1)) cp[i] = 0.0;
            }
            else {
                invden[i] = 1.0 / (m_d[i] - m_a[i - 1] * cp[i - 1]);
                if (i < (n - 1)) cp[i]  = m_c[i] * invden[i];
            }
        }
    }

    // Matvec
    template<typename _Vector>
    _Vector apply(const _Vector &x) const {
        const size_t m = rows();
        if (size_t(x.size()) != m) throw std::runtime_error("Tri-diagonal matvec size mismatch.");
        _Vector result(rows());
        for (size_t i = 0; i < m; ++i)
            result[i] = m_d[i] * x[i];
        for (size_t i = 0; i < m - 1; ++i) {
            result[i + 1] += m_a[i] * x[i    ];
            result[i    ] += m_c[i] * x[i + 1];
        }
        return result;
    }

    const std::vector<Real_> &lowerDiagonal() const { return m_a; }
    const std::vector<Real_> &     diagonal() const { return m_d; }
    const std::vector<Real_> &upperDiagonal() const { return m_c; }

private:
    void m_init() {
        if (m_a.size() != rows() - 1) throw std::runtime_error("Invalid lower diagonal size");
        if (m_c.size() != rows() - 1) throw std::runtime_error("Invalid upper diagonal size");
        m_isFixed.assign(rows(), false);
        m_fixedVarValues.resize(rows());
    }

    std::vector<bool> m_isFixed;
    std::vector<Real_> m_fixedVarValues;

    std::vector<Real_> m_a, m_d, m_c;

    struct Factorization {
        std::vector<Real_> inv_den, cPrime;
    };

    mutable std::unique_ptr<Factorization> m_factorization;
};

#endif /* end of include guard: TRIDIAGONALSYSTEM_HH */
