#ifndef LINKAGETERMINALEDGESENSITIVITY_HH
#define LINKAGETERMINALEDGESENSITIVITY_HH

#include "RodLinkage.hh"

template<typename Derived, typename Real_> std::enable_if_t<Derived::RowsAtCompileTime == 0> unpack_delta_jparams(const Eigen::MatrixBase<Derived> & , const size_t           , Eigen::Matrix<Real_, 3, 1>,              Real_ &,            Real_ &         ) { throw std::logic_error("Fail."); }
template<typename Derived, typename Real_> std::enable_if_t<Derived::RowsAtCompileTime == 6> unpack_delta_jparams(const Eigen::MatrixBase<Derived> &v, const size_t len_offset, Eigen::Matrix<Real_, 3, 1> &delta_omega, Real_ &delta_alpha, Real_ &delta_len) { delta_omega = v.head(3); delta_alpha = v[3]; delta_len = v[len_offset];  }

// Store the derivatives of the constrained terminal edges' edge vectors
// and material frame angles with respect to the controlling joint's 
// parameters. This is represented by the following Jacobian matrix:
//  e^j     [     de^j/d_omega       de^j/d_alpha       de^j/d_len_A       de^j/d_len_B]
//  theta^j [ dtheta^j/d_omega   dtheta^j/d_alpha   dtheta^j/d_len_A   dtheta^j/d_len_B]
//
// To obtain the change in the centerline positions, this Jacobian
// should be composed with the following linear map:
//           pos        e_X     theta^j
// x_j     [  I    -s_jX 0.5 I     0   ] [ I 0 ... 0]
// x_{j+1} [  I     s_jX 0.5 I     0   ] [ jacobian ]
// theta^j [  0          0         I   ]
// (here, "x_j" and "x_{j+1}" are the tail and tip vertex centerline positions)
template<typename Real_>
struct LinkageTerminalEdgeSensitivity {
    using Vec3 = Vec3_T<Real_>;
    using Mat3 = Mat3_T<Real_>;
    using Joint = typename RodLinkage_T<Real_>::Joint;
    using ropt = typename RodLinkage_T<Real_>::ropt;

    // Whether edge j is part of the joint's rod "A" or "B",
    // and whether its orientation agrees with e_A/e_B.
    size_t j;
    bool is_A;
    double s_jA, s_jB; // +/- 1 or 0 (one of these is zero).
    double s_jX;       // +/- 1

    static constexpr size_t JacobianRows = 4;
    static constexpr size_t JacobianCols = 6;

    // Derivative of [e^j, theta^j]^T with respect to [omega, alpha, len_A, len_B]
    Eigen::Matrix<Real_, JacobianRows, JacobianCols> jacobian;

    // hessian[i] holds the Hessian of the i^th component of [e^j, theta^j]^T
    // with respect to (omega, alpha, len_A, len_B).
    std::array<Eigen::Matrix<Real_, JacobianCols, JacobianCols>, JacobianRows> hessian;

    // Directional derivative of jacobian (evaluated if update was called with a joint parameter perturbation vector).
    Eigen::Matrix<Real_, JacobianRows, JacobianCols> delta_jacobian;

    // Leave uninitialized; update must be called before this instance is used!
    LinkageTerminalEdgeSensitivity() { }

    LinkageTerminalEdgeSensitivity(const Joint &joint, size_t si, const ElasticRod_T<Real_> &rod, bool updatedSource, bool evalHessian = false) {
        update(joint, si, rod, updatedSource, evalHessian);
    }

    void update(const Joint &joint, size_t si, const ElasticRod_T<Real_> &rod, bool updatedSource, bool evalHessian) {
        update(joint, si, rod, updatedSource, evalHessian, Eigen::Matrix<Real, 0, 0>());
    }

    template<typename Derived>
    void update(const Joint &joint, size_t si, const ElasticRod_T<Real_> &rod, bool updatedSource, bool evalHessian, const Eigen::MatrixBase<Derived> &delta_jparams) {
        bool isStart;
        std::tie(s_jA, s_jB, isStart) = joint.terminalEdgeIdentification(si);
        is_A = (s_jB == 0);
        if ((s_jA == 0) == is_A) throw std::runtime_error("Terminal edge of segment passed to LinkageTerminalEdgeSensitivity should be controlled by exactly one of the joint's edge vectors");
        j = isStart ? 0 : rod.numEdges() - 1;
        s_jX = s_jA + s_jB;

        const double dangle_dalpha = is_A ? -0.5 : 0.5; // Angle between the joint's bisecting "tangent" vector is (-alpha / 2) for rod A and (alpha / 2) for rod B
        const auto &dc = rod.deformedConfiguration();

        const auto &t  = dc.tangent[j],
                   &d1 = dc.materialFrame[j].d1,
                   &w  = joint.omega(),
                   &ns = joint.source_normal(),
                   &n  = joint.normal();
                    
        const auto  tsX = is_A ? joint.source_t_A() : joint.source_t_B();
        const Real_ len = is_A ? joint.len_A() : joint.len_B();
        const size_t len_offset = 4 + (is_A ? 0 : 1),
                   alpha_offset = 3,
                   theta_offset = 3;

        // The following code has inlined, optimized expressions for the quantities:
        //      Vec3 dt_dalpha   = ropt::rotated_vector(w, ns.cross(tsX));
        //      Vec3 tX          = ropt::rotated_vector(w, tsX);
        //      Mat3 dtX_domega  = ropt::grad_rotated_vector(w, tsX);
        //      Mat3 d_n_d_omega = ropt::grad_rotated_vector(w, ns); // actually we compute -dc.materialFrame[j].d1 dotted with this quantity
        // These quantities share intermediate values and are a bottleneck, especially for autodiff types
        const Real_ theta_sq    = w.squaredNorm();

        // Use simpler formulas for rotation variations around the identity
        // (But only if we're using a native floating point type; for autodiff
        // types, we need the full formulas).
        const bool variation_around_identity = (theta_sq == 0) && (std::is_arithmetic<Real_>::value);

        const Real_ theta       = sqrt(theta_sq);
        const Real_ sinc_th     = sinc(theta, theta_sq),
                    omcdthsq    = one_minus_cos_div_theta_sq(theta, theta_sq),
                    tcmsdtc     = theta_cos_minus_sin_div_theta_cubed(theta, theta_sq),
                    tcm2ptsdtp4 = two_cos_minus_2_plus_theta_sin_div_theta_pow_4(theta, theta_sq);
        const Real_ w_dot_tsX   = w.dot(tsX);
        const Vec3 w_cross_tsX  = w.cross(tsX);
        const Vec3 tX           = s_jX * t;
        const Vec3 dt_dalpha    = s_jX * d1; // n.cross(tX) = s_jX * n.cross(t) = s_jX * d1
        const Vec3 neg_tsX_sinc = tsX * (-sinc_th);

        Mat3 dtX_domega;
        dtX_domega <<       0, -neg_tsX_sinc[2],  neg_tsX_sinc[1],
              neg_tsX_sinc[2],                0, -neg_tsX_sinc[0],
             -neg_tsX_sinc[1],  neg_tsX_sinc[0],                0;
        if (!variation_around_identity) {
            dtX_domega += (neg_tsX_sinc + w_cross_tsX * tcmsdtc + w * (w_dot_tsX * tcm2ptsdtp4)) * w.transpose() + (omcdthsq * w) * tsX.transpose();
            dtX_domega.diagonal().array() += w_dot_tsX * omcdthsq;
        }

        const Vec3 neg_ns_sinc = ns * (-sinc_th);
        Vec3 w_cross_ns, d_n_d_omega_w_coeff;
        Real_ w_dot_ns;
        if (!variation_around_identity) {
            w_cross_ns = w.cross(ns);
            w_dot_ns = w.dot(ns);
            d_n_d_omega_w_coeff = (neg_ns_sinc + w_cross_ns * tcmsdtc + w * (w_dot_ns * tcm2ptsdtp4));
        }

        jacobian.setZero();
        jacobian.template block<3, 3>(0,            0) = dtX_domega * len;
        jacobian.template block<3, 1>(0, alpha_offset) = dt_dalpha * (len * dangle_dalpha); // d e^j / d_alpha
        jacobian.template block<3, 1>(0,   len_offset) = tX;

        // Gradient due to the rotating joint normal
        // jacobian.template block<1, 3>(theta_offset, 0) = -d1.transpose() * d_n_d_omega;
        if (variation_around_identity) jacobian.template block<1, 3>(theta_offset, 0) = neg_ns_sinc.cross(d1).transpose();
        else                           jacobian.template block<1, 3>(theta_offset, 0) = neg_ns_sinc.cross(d1).transpose() -(d1.dot(d_n_d_omega_w_coeff)) * w.transpose() - ((d1.dot(w)) * omcdthsq) * ns.transpose() - (w_dot_ns * omcdthsq) * d1.transpose();

        // Contribution due to the rotating reference directors.
        // (Zero if the source frame is updated: if we're evaluating at the
        // source frame configuration, the reference directors don't rotate
        // around the tangent.)
        if (!updatedSource) {
            const auto &rds1 = dc.sourceReferenceDirectors[j].d1,
                       & rd1 = dc.referenceDirectors[j].d1,
                       & rd2 = dc.referenceDirectors[j].d2,
                       &  ts = dc.sourceTangent[j];
            // As t approaches -ts, the parallel transport operator becomes singular;
            // see discussion in the parallelTransportNormalized function in VectorOperations.hh.
            // To avoid NaNs in this case, we arbitrarily define the parallel
            // transport operator to be the identity (so its derivative is 0).
            const Real_ chi_hat = 1.0 + ts.dot(t);
            if (std::abs(stripAutoDiff(chi_hat)) > 1e-14) {
                Real_ inv_chi_hat = 1.0 / chi_hat;
                Real_ rds1_dot_t = rds1.dot(t),
                      rd2_dot_ts = rd2.dot(ts);

                // Angular velocity of the reference director as the terminal edge is perturbed
                Vec3 neg_d_theta_ref_dt = inv_chi_hat * (rds1_dot_t * rd1.cross(ts) + rd2_dot_ts * rds1)
                                        - (inv_chi_hat * inv_chi_hat * rds1_dot_t * rd2_dot_ts) * ts
                                        + rds1.cross(rd1);
                jacobian.template block<1, 3>(theta_offset, 0) += s_jX * neg_d_theta_ref_dt.transpose() * dtX_domega;
                jacobian(theta_offset, alpha_offset) = (s_jX * dangle_dalpha) * neg_d_theta_ref_dt.dot(dt_dalpha);
            }
        }

        if (!evalHessian) return;

        const Real_ eptsmecmftsdtp6 = eight_plus_theta_sq_minus_eight_cos_minus_five_theta_sin_div_theta_pow_6(theta, theta_sq);
        const Real_ ttcptsm3sdtp5   = three_theta_cos_plus_theta_sq_minus_3_sin_div_theta_pow_5(theta, theta_sq);
        const Vec3 w_tcm2ptsdtp4 = tcm2ptsdtp4 * w;

        // t.cross(dt_dalpha) = t.cross(s_jX * d1) = s_jX * (t.cross(d1)) = s_jX * n
        const Vec3 d2_theta_dalpha_domega = (-0.5 * dangle_dalpha * s_jX) * (dtX_domega.transpose() * n);

        Mat3 d_n_d_omega; // ropt::grad_rotated_vector(w, ns)
        d_n_d_omega <<     0, -neg_ns_sinc[2],  neg_ns_sinc[1],
              neg_ns_sinc[2],               0, -neg_ns_sinc[0],
             -neg_ns_sinc[1],  neg_ns_sinc[0],               0;
        if (!variation_around_identity) {
            d_n_d_omega += d_n_d_omega_w_coeff * w.transpose() + (omcdthsq * w) * ns.transpose();
            d_n_d_omega.diagonal().array() += w_dot_ns * omcdthsq;
        }

        // dtperp_domega = (n x dtX_domega) - (tX x d_n_d_omega) = (n x dt_dalpha) * dt_dalpha.transpose() * dtX_domega - (tX x dt_dalpha) * (dt_dalpha.transpose() * dtX_domega)
        const auto tperp_dot_dtX_domega = (dt_dalpha.transpose() * dtX_domega).eval();
        const Mat3 dtperp_domega = -tX * tperp_dot_dtX_domega - n * (dt_dalpha.transpose() * d_n_d_omega);

        constexpr size_t delta_size = Derived::RowsAtCompileTime;
        static_assert((delta_size == 0) || (delta_size == 6), "Invalid delta joint parameter vector size");

        // Evaluate full Hessian
        if (delta_size == 0) {
            for (size_t i = 0; i < 4; ++i) { hessian[i].setZero(); }

            ///////////////////////////////////////////////////////////////////
            // e^j hessian
            ///////////////////////////////////////////////////////////////////
            // d^2 e^j / d_omega d_omega
            {
                // std::array<Eigen::Ref<Mat3>, 3> hess_e_omega{{hessian[0].template block<3, 3>(0, 0), hessian[1].template block<3, 3>(0, 0), hessian[2].template block<3, 3>(0, 0)}};
                // ropt::hess_rotated_vector(w, len * tsX, hess_e_omega);

                const Vec3 v = len * tsX;
                if (variation_around_identity) {
                    const Vec3 half_v = 0.5 * v;
                    for (size_t i = 0; i < 3; ++i) {
                        // hess_comp[i] = -v[i] * I + 0.5 * (I.col(i) * v.transpose() + v * I.row(i));
                        auto dst = hessian[i].template block<3, 3>(0, 0);
                        dst.diagonal().array() = -v[i];
                        dst.row(i) += half_v.transpose();
                        dst.col(i) += half_v;
                    }
                }
                else {
                    const Real_ w_dot_v = len * w_dot_tsX;
                    const Mat3 v_cross_term = ropt::cross_product_matrix(tcmsdtc * v);
                    const Vec3 v_cross_w = (-len) * w_cross_tsX;
                    for (size_t i = 0; i < 3; ++i) {
                        auto dst = hessian[i].template block<3, 3>(0, 0);
                        dst = ((0.5 * (w_dot_v * eptsmecmftsdtp6 * w[i] + ttcptsm3sdtp5 * v_cross_w[i] - tcmsdtc * v[i])) * w + v_cross_term.col(i) + w_tcm2ptsdtp4[i] * v) * w.transpose();
                        dst.col(i) += omcdthsq * v + w_dot_v * w_tcm2ptsdtp4;
                        dst += dst.transpose().eval();
                        dst.diagonal().array() += w_dot_v * w_tcm2ptsdtp4[i] - sinc_th * v[i] - tcmsdtc * v_cross_w[i];
                    }
                }
            }

            // d^2 e^j / d_omega d_len
            for (size_t i = 0; i < 3; ++i) {
                hessian[i].template block<3, 1>(0, len_offset) = dtX_domega.row(i).transpose();
                hessian[i].template block<1, 3>(len_offset, 0) = dtX_domega.row(i);
            }

            for (size_t i = 0; i < 3; ++i) {
                hessian[i].template block<3, 1>(0, alpha_offset) = (dangle_dalpha * len) * dtperp_domega.row(i).transpose(); // (omega, alpha)
                hessian[i].template block<1, 3>(alpha_offset, 0) = (dangle_dalpha * len) * dtperp_domega.row(i);             // (alpha, omega)
                hessian[i](len_offset,   alpha_offset) = dangle_dalpha * dt_dalpha[i];                                       // (alpha,   len)
                hessian[i](alpha_offset,   len_offset) = dangle_dalpha * dt_dalpha[i];                                       // (  len, alpha)
                hessian[i](alpha_offset, alpha_offset) = -0.25 * (is_A ? joint.e_A()[i] : joint.e_B()[i]);                   // (alpha, alpha) (0.25 is dangle_dalpha^2)
            }

            ///////////////////////////////////////////////////////////////////
            // Theta hessian
            ///////////////////////////////////////////////////////////////////

            // std::array<Mat3, 3> d2_n_domega_domega;
            // ropt::hess_rotated_vector(w, ns, d2_n_domega_domega);
            Mat3 d1_dot_pder2_n_domega_domega_presym;
            {
                if (variation_around_identity) {
                    const Vec3 half_ns = 0.5 * ns;
                    d1_dot_pder2_n_domega_domega_presym = d1 * half_ns.transpose();
                    // d1_dot_pder2_n_domega_domega_presym.diagonal().array() += -d1.dot(half_ns); // Note: d1 should be perpendicular to ns in this case!
                }
                else {
                    const Real_ w_dot_d1 = w.dot(d1);
                    d1_dot_pder2_n_domega_domega_presym  = d1 * (omcdthsq * ns + w_dot_ns * w_tcm2ptsdtp4).transpose()
                              + ((0.5 * (w_dot_ns * eptsmecmftsdtp6 * w_dot_d1 - ttcptsm3sdtp5 * w_cross_ns.dot(d1) - tcmsdtc * ns.dot(d1))) * w + tcmsdtc * ns.cross(d1) + (tcm2ptsdtp4 * w_dot_d1) * ns) * w.transpose();
                    d1_dot_pder2_n_domega_domega_presym.diagonal().array() += 0.5 * d1.dot(d_n_d_omega_w_coeff);
                }
            }

            // const Mat3 presym_block = (-0.5 * s_jX) * (d_n_d_omega.transpose() * dtperp_domega) - d1_dot_pder2_n_domega_domega_presym;
            const Mat3 presym_block = (0.5 * s_jX) * ((d_n_d_omega.transpose() * tX) * tperp_dot_dtX_domega) - d1_dot_pder2_n_domega_domega_presym;
            hessian[theta_offset].template block<3, 3>(0, 0) = presym_block + presym_block.transpose();

            hessian[theta_offset].template block<3, 1>(0, alpha_offset) = d2_theta_dalpha_domega;
            hessian[theta_offset].template block<1, 3>(alpha_offset, 0) = d2_theta_dalpha_domega.transpose();
        }
        // Evaluate directional derivative only
        else {
            Vec3 delta_omega;
            Real_ delta_alpha, delta_len;
            unpack_delta_jparams(delta_jparams, len_offset, delta_omega, delta_alpha, delta_len);

            delta_jacobian.setZero();

            const Real_ w_dot_delta_omega       = w.dot(delta_omega);
            const Real_ len_tsX_dot_delta_omega = len * tsX.dot(delta_omega);

            ///////////////////////////////////////////////////////////////////
            // e^j hessian
            ///////////////////////////////////////////////////////////////////
            // d^2 e^j / d_omega d_omega
            {
                const Vec3 v = len * tsX;
                if (variation_around_identity) {
                    const Vec3 half_v = 0.5 * v;
                    for (size_t i = 0; i < 3; ++i) {
                        auto dst = delta_jacobian.template block<3, 3>(0, 0);
                        dst = delta_omega * half_v.transpose()
                            - v * delta_omega.transpose();
                        dst.diagonal().array() += 0.5 * len_tsX_dot_delta_omega;
                    }
                }
                else {
                    const Real_ w_dot_v = len * w_dot_tsX;
                    const Mat3 v_cross_term = ropt::cross_product_matrix(tcmsdtc * v);
                    const Vec3 v_cross_w = (-len) * w_cross_tsX;
                    auto dst = delta_jacobian.template block<3, 3>(0, 0);
                    dst = (-w_dot_delta_omega) * v_cross_term +
                          (w_tcm2ptsdtp4 * w_dot_delta_omega) * v.transpose()
                          + ((w_dot_v * eptsmecmftsdtp6 * w + ttcptsm3sdtp5 * v_cross_w - tcmsdtc * v) * w_dot_delta_omega + w_tcm2ptsdtp4 * len_tsX_dot_delta_omega - tcmsdtc * v.cross(delta_omega)) * w.transpose()
                          + delta_omega * (omcdthsq * v + w_dot_v * w_tcm2ptsdtp4).transpose()
                          + (w_dot_v * w_tcm2ptsdtp4 - sinc_th * v - tcmsdtc * v_cross_w) * delta_omega.transpose();
                    dst.diagonal().array() += omcdthsq * len_tsX_dot_delta_omega + w_dot_v * tcm2ptsdtp4 * w_dot_delta_omega;
                }
            }

            const Vec3 delta_tperp = dtperp_domega * delta_omega;

            delta_jacobian.template block<3, 1>(0, len_offset)  = dtX_domega * delta_omega
                                                                + (dangle_dalpha * delta_alpha) * dt_dalpha;           // (  len, alpha)
            delta_jacobian.template block<3, 3>(0,          0) += dtX_domega * delta_len
                                                                + (dangle_dalpha * len * delta_alpha) * dtperp_domega; // (omega, alpha)


            delta_jacobian.template block<3, 1>(0, alpha_offset).noalias() += (dangle_dalpha * len) * delta_tperp      // (alpha, omega)
                                                                           -  (0.25 * delta_alpha * len) * tX          // (alpha, alpha) (0.25 is dangle_dalpha^2)
                                                                           +  (dangle_dalpha * delta_len) * dt_dalpha; // (alpha,   len)

            ///////////////////////////////////////////////////////////////////
            // Theta hessian
            ///////////////////////////////////////////////////////////////////

            // std::array<Mat3, 3> d2_n_domega_domega;
            // ropt::hess_rotated_vector(w, ns, d2_n_domega_domega);
            Vec3 d1_dot_delta_n_domega;
            const Real_ ns_dot_delta_omega = ns.dot(delta_omega);
            const Real_ d1_dot_delta_omega = d1.dot(delta_omega);
            {
                if (variation_around_identity) {
                    d1_dot_delta_n_domega = d1 * (0.5 * ns_dot_delta_omega) + (0.5 * d1_dot_delta_omega) * ns;
                }
                else {
                    const Real_ w_dot_d1 = w.dot(d1);
                    const Vec3 tmp2 = tcmsdtc * ns.cross(d1);

                    d1_dot_delta_n_domega  = (w_dot_delta_omega * (w_dot_ns * eptsmecmftsdtp6 * w_dot_d1 - ttcptsm3sdtp5 * w_cross_ns.dot(d1) - tcmsdtc * ns.dot(d1)) + tcm2ptsdtp4 * (ns_dot_delta_omega * w_dot_d1 + d1_dot_delta_omega * w_dot_ns) + tmp2.dot(delta_omega)) * w
                                           + (w_dot_delta_omega * (tcm2ptsdtp4 * w_dot_d1) + omcdthsq * d1_dot_delta_omega) * ns
                                           + d1.dot(d_n_d_omega_w_coeff) * delta_omega
                                           + w_dot_delta_omega * tmp2
                                           + (ns_dot_delta_omega * omcdthsq + w_dot_ns * tcm2ptsdtp4 * w_dot_delta_omega) * d1;
                }
            }
            // // dtperp_domega = (n x dtX_domega) - (tX x d_n_d_omega) = (n x dt_dalpha) * dt_dalpha.transpose() * dtX_domega - (tX x dt_dalpha) * (dt_dalpha.transpose() * dtX_domega)
            // const Mat3 dtperp_domega = -tX * (dt_dalpha.transpose() * dtX_domega) - n * (dt_dalpha.transpose() * d_n_d_omega);
            // delta_jacobian.template block<1, 3>(theta_offset, 0) = (-0.5 * s_jX) * (d_n_d_omega.transpose() * delta_tperp + dtperp_domega.transpose() * (d_n_d_omega * delta_omega)) - d1_dot_delta_n_domega;
            delta_jacobian.template block<1, 3>(theta_offset, 0) = (-0.5 * s_jX) * (d_n_d_omega.transpose() * delta_tperp - tperp_dot_dtX_domega.transpose() * tX.dot(d_n_d_omega * delta_omega)) - d1_dot_delta_n_domega
                                                                 + d2_theta_dalpha_domega * delta_alpha;
            delta_jacobian(theta_offset, alpha_offset) = d2_theta_dalpha_domega.dot(delta_omega);
        }
    }

    // Fix Eigen alignment issues
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


#endif /* end of include guard: LINKAGETERMINALEDGESENSITIVITY_HH */
