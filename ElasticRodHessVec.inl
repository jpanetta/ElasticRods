////////////////////////////////////////////////////////////////////////////////
// ElasticRodHessVec.inl
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Elastic energy Hessian-vector product formulas.
//  These are mostly copied from the Hessian sparse matrix implementation;
//  more efficient directional derivative formulas could be derived.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  12/08/2018 20:01:45
////////////////////////////////////////////////////////////////////////////////

template<typename Real_>
void ElasticRod_T<Real_>::applyHessEnergy(const VecX &v, VecX &result, bool variableRestLen, const HessianComputationMask &mask) const {
    // BENCHMARK_SCOPED_TIMER_SECTION timer("ElasticRod_T::applyHessEnergy");
    const size_t ndof = variableRestLen ? numExtendedDoF() : numDoF();
    if (size_t(     v.size()) != ndof) throw std::runtime_error( "Input vector size mismatch");
    if (size_t(result.size()) != ndof) throw std::runtime_error("Output vector size mismatch");

    using M32d = Eigen::Matrix<Real_, 3, 2>;

    const size_t nv = numVertices(), ne = numEdges();
    const auto &dc = deformedConfiguration();

    for (size_t i = 1; i < nv - 1; ++i) {
        //////////////////////////////////////////////////////
        // Quantities needed by multiple parts of the Hessian.
        //////////////////////////////////////////////////////
        const auto &kb = dc.kb[i];
        const auto &ti    = dc.tangent[i],
                   &tim1  = dc.tangent[i - 1];
        M32d t; // copy for vectorization alignment...
        t.col(0) = tim1; t.col(1) = ti;
        const Vec2 inv_len(1.0 / dc.len[i - 1],
                           1.0 / dc.len[i    ]);
        const Vec2 rlen(m_restLen[i - 1], m_restLen[i]);
        const Real_ inv_2libar = 1.0 / rlen.sum();
        const Real_ beta_div_2libar = m_twistingStiffness[i] * inv_2libar;

        const Real_ ks = density(i - 1) * m_stretchingStiffness[i - 1];
        const Real_ inv_restlen = 1.0 / rlen[0];
        const Real_ ks_inv_len = ks * inv_len[0];

        const Real_ t_im1_dot_ti = t.col(0).dot(t.col(1));
        const Real_ inv_chi = 1.0 / (1.0 + t_im1_dot_ti);
        const Vec3  t_tilde = t.rowwise().sum() * inv_chi;

        const size_t x_offset = 3 * (i - 1),      // Index of the first position variable for the stencil
                 theta_offset = 3 * nv + (i - 1); // Index of the first theta variable

        const Vec2 B_div_2libar(m_bendingStiffness[i].lambda_1 * inv_2libar,
                                m_bendingStiffness[i].lambda_2 * inv_2libar);

        M32d delta_e;
        delta_e.col(0) = v.template segment<3>(x_offset + 3) - v.template segment<3>(x_offset    );
        delta_e.col(1) = v.template segment<3>(x_offset + 6) - v.template segment<3>(x_offset + 3);
        const Vec2 delta_theta = v.template segment<2>(theta_offset);

        // Twisting quantities
        const Real_ m = dc.theta(i) - dc.theta(i - 1) + dc.referenceTwist[i] - m_restTwist[i];
        const Real_ dE_dm = 2 * beta_div_2libar * m;
        const Vec2 invlen_kb_dot_delta_e = inv_len.asDiagonal() * (delta_e.transpose() * kb);
        Vec2 kb_coeff(Vec2::Zero()), t_coeff(Vec2::Zero());

        const Vec2 inv_len_neg(-inv_len[0], inv_len[1]);
        const Vec2 two_inv_chilen_neg = (2 * inv_chi) * inv_len_neg;
        std::array<std::array<M32d, 2>, 2> cross_prod_term;
        {
            // cross_prod_term[k][0].col(0) =   coeff  * ti  .cross(dc.materialFrame[i - 1].get(kother));
            // cross_prod_term[k][1].col(1) = (-coeff) * tim1.cross(dc.materialFrame[i    ].get(kother));
            // cross_prod_term[k][0].col(1) = (-coeff) * tim1.cross(dc.materialFrame[i - 1].get(kother));
            // cross_prod_term[k][1].col(0) =   coeff  * ti  .cross(dc.materialFrame[i    ].get(kother));
            const Vec2 two_inv_chilen_neg_t_im1_dot_ti = two_inv_chilen_neg * t_im1_dot_ti;
            cross_prod_term[0][0].col(0) = (inv_len[0] * dc.per_corner_kappa[i](0, 0)) * t.col(0) + two_inv_chilen_neg_t_im1_dot_ti[0] * dc.materialFrame[i - 1].d1;
            cross_prod_term[0][0].col(1) =                                                                       two_inv_chilen_neg[1] * dc.materialFrame[i - 1].d1;
            cross_prod_term[0][1].col(0) =                                                                       two_inv_chilen_neg[0] * dc.materialFrame[i    ].d1;
            cross_prod_term[0][1].col(1) = (inv_len[1] * dc.per_corner_kappa[i](0, 1)) * t.col(1) + two_inv_chilen_neg_t_im1_dot_ti[1] * dc.materialFrame[i    ].d1;
            cross_prod_term[1][0].col(0) = (inv_len[0] * dc.per_corner_kappa[i](1, 0)) * t.col(0) + two_inv_chilen_neg_t_im1_dot_ti[0] * dc.materialFrame[i - 1].d2;
            cross_prod_term[1][0].col(1) =                                                                       two_inv_chilen_neg[1] * dc.materialFrame[i - 1].d2;
            cross_prod_term[1][1].col(0) =                                                                       two_inv_chilen_neg[0] * dc.materialFrame[i    ].d2;
            cross_prod_term[1][1].col(1) = (inv_len[1] * dc.per_corner_kappa[i](1, 1)) * t.col(1) + two_inv_chilen_neg_t_im1_dot_ti[1] * dc.materialFrame[i    ].d2;
        }

        // d_kappa_k_j_de[k][j].col(0) := d/deim1 (kappa_k)_i^j
        // d_kappa_k_j_de[k][j].col(1) := d/dei   (kappa_k)_i^j
        // d_kappa_k_j_dtheta_j(k, j)  := d/dtheta_j (kappa_k)_i^j
        std::array<std::array<M32d, 2>, 2> d_kappa_k_j_de;
        Mat2 d_kappa_k_j_dtheta_j;
        {
            const M32d t_tilde_otimes_invlen = t_tilde * inv_len.transpose();
            d_kappa_k_j_dtheta_j.row(0) =  dc.per_corner_kappa[i].row(1);
            d_kappa_k_j_dtheta_j.row(1) = -dc.per_corner_kappa[i].row(0);
            d_kappa_k_j_de[0][0] = cross_prod_term[0][0] - dc.per_corner_kappa[i](0, 0) * t_tilde_otimes_invlen;
            d_kappa_k_j_de[0][1] = cross_prod_term[0][1] - dc.per_corner_kappa[i](0, 1) * t_tilde_otimes_invlen;
            d_kappa_k_j_de[1][0] = cross_prod_term[1][0] - dc.per_corner_kappa[i](1, 0) * t_tilde_otimes_invlen;
            d_kappa_k_j_de[1][1] = cross_prod_term[1][1] - dc.per_corner_kappa[i](1, 1) * t_tilde_otimes_invlen;
        }

        M32d delta_dE_de(M32d::Zero());
        Vec2 delta_dE_dtheta(Vec2::Zero()),
             delta_dE_drlen(Vec2::Zero());

        Mat2 delta_kappa_k_j;
        for (size_t k = 0; k < 2; ++k) {
            for (size_t adj_edge = 0; adj_edge < 2; ++adj_edge) {
                // delta_kappa_k_j(k, adj_edge) = d_kappa_k_j_de[k][adj_edge].cwiseProduct(delta_e).sum(); // strangely, this benchmarks slower...
                delta_kappa_k_j(k, adj_edge) = d_kappa_k_j_de[k][adj_edge].col(0).dot(delta_e.col(0))
                                             + d_kappa_k_j_de[k][adj_edge].col(1).dot(delta_e.col(1))
                                             + d_kappa_k_j_dtheta_j(k, adj_edge) * delta_theta[adj_edge];
            }
        }
        Mat2 d_kappa_k_j_de_coeff(Mat2::Zero());

        if (mask.dof_in && mask.dof_out) { // only compute dof-dof part if needed
            ////////////////////////////////////////////////////////////////////////
            // Gradient outer product terms
            ////////////////////////////////////////////////////////////////////////
            if (m_bendingEnergyType == BendingEnergyType::Bergou2010) {
                const Vec2 delta_kappa_sum = 0.5 * B_div_2libar.asDiagonal() * delta_kappa_k_j.rowwise().sum();
                delta_dE_de += delta_kappa_sum[0] * (d_kappa_k_j_de[0][0] + d_kappa_k_j_de[0][1])
                            +  delta_kappa_sum[1] * (d_kappa_k_j_de[1][0] + d_kappa_k_j_de[1][1]);
                delta_dE_dtheta += d_kappa_k_j_dtheta_j.transpose() * delta_kappa_sum;
            }
            if (m_bendingEnergyType == BendingEnergyType::Bergou2008) {
                const Mat2 scaled_delta_kappa_k_j = B_div_2libar.asDiagonal() * delta_kappa_k_j * (rlen * (2 * inv_2libar)).asDiagonal();
                delta_dE_dtheta += (scaled_delta_kappa_k_j.transpose() * d_kappa_k_j_dtheta_j).diagonal();
                delta_dE_de += scaled_delta_kappa_k_j(0, 0) * d_kappa_k_j_de[0][0]
                            +  scaled_delta_kappa_k_j(0, 1) * d_kappa_k_j_de[0][1]
                            +  scaled_delta_kappa_k_j(1, 0) * d_kappa_k_j_de[1][0]
                            +  scaled_delta_kappa_k_j(1, 1) * d_kappa_k_j_de[1][1];
            }

            ////////////////////////////////////////////////////////////////////////
            // Kappa Hessian terms
            ////////////////////////////////////////////////////////////////////////
            // dE_dkappa_k_j(k, j) = dE/(kappa_k)_i^j
            Mat2 dE_dkappa_k_j;
            if (m_bendingEnergyType == BendingEnergyType::Bergou2010) { dE_dkappa_k_j.colwise() = B_div_2libar.asDiagonal() * (dc.kappa[i] - m_restKappa[i]); }
            if (m_bendingEnergyType == BendingEnergyType::Bergou2008) { dE_dkappa_k_j = (2 * inv_2libar) * B_div_2libar.asDiagonal() * (dc.per_corner_kappa[i].colwise() - m_restKappa[i]) * rlen.asDiagonal(); }

            Mat2 t_cross_kb_calculator;
            t_cross_kb_calculator << 2 * inv_chi * t_im1_dot_ti,  2 * inv_chi,
                                                   -2 * inv_chi, -2 * inv_chi * t_im1_dot_ti;
            const Mat2 t_dot_delta_e = delta_e.transpose() * t;

            const M32d t_cross_kb = t * t_cross_kb_calculator;
            // const M32d t_cross_kb = t.colwise().cross(kb);
            const Vec2 t_tilde_dot_delta_e = t_dot_delta_e.rowwise().sum() * inv_chi;
            const Vec2 t_dot_delta_e_diag(t_dot_delta_e.diagonal());
            const Real_ coeff_a = inv_len.dot(t_tilde_dot_delta_e);

            const Vec2 ilen_sq = inv_len.asDiagonal() * inv_len;

            const Vec2 ilen_sq_kb_dot_delta_e = inv_len.asDiagonal() * invlen_kb_dot_delta_e;
            const Vec2 ilen_sq_tb_cross_kb_dot_delta_e = (0.5 * ilen_sq).asDiagonal() * (t_dot_delta_e * t_cross_kb_calculator).diagonal();

            const Mat2 coeff = dE_dkappa_k_j * ilen_sq.asDiagonal();

            Mat2 d_k_dot_delta_e;
            d_k_dot_delta_e << dc.materialFrame[i - 1].d1.dot(delta_e.col(0)), dc.materialFrame[i].d1.dot(delta_e.col(1)),
                               dc.materialFrame[i - 1].d2.dot(delta_e.col(0)), dc.materialFrame[i].d2.dot(delta_e.col(1));
            const Vec2 t_cross_kb_coeff = -0.5 * (coeff.transpose() * d_k_dot_delta_e).diagonal();

            kb_coeff = coeff.row(0).array() * d_k_dot_delta_e.row(1).array()
                     - coeff.row(1).array() * d_k_dot_delta_e.row(0).array();

            d_kappa_k_j_de_coeff.row(1) += dE_dkappa_k_j.row(0) * delta_theta.asDiagonal();
            d_kappa_k_j_de_coeff.row(0) -= dE_dkappa_k_j.row(1) * delta_theta.asDiagonal();

            delta_dE_dtheta += (dE_dkappa_k_j.row(0).array() * delta_kappa_k_j.row(1).array()
                              - dE_dkappa_k_j.row(1).array() * delta_kappa_k_j.row(0).array()).matrix().transpose();

            const M32d weighted_cross_prod_term =
                dE_dkappa_k_j(0, 0) * cross_prod_term[0][0] +
                dE_dkappa_k_j(1, 0) * cross_prod_term[1][0] +
                dE_dkappa_k_j(0, 1) * cross_prod_term[0][1] +
                dE_dkappa_k_j(1, 1) * cross_prod_term[1][1];

            const Vec2 contrib = (dE_dkappa_k_j.transpose() * dc.per_corner_kappa[i]).diagonal();
            const Vec2 finite_xport_coeff = ilen_sq.asDiagonal() * contrib;
            const Vec2 half_invlen_dE_dm = (dE_dm * 0.5) * inv_len;
            const Vec2 coeff2 = -0.5 * half_invlen_dE_dm;
            const Real_ coeff3 = contrib.sum() * inv_chi;

            {
                const Vec2 tmp = coeff2.asDiagonal() * invlen_kb_dot_delta_e;
                t_coeff = (inv_len_neg * (coeff3 * inv_len_neg.dot(t_dot_delta_e_diag)))
                             + finite_xport_coeff.asDiagonal() * t_dot_delta_e_diag
                             + tmp;
                const Vec2 t_tilde_coeff = (contrib.sum() * 2 * coeff_a - (weighted_cross_prod_term.transpose() * delta_e).trace()) * inv_len + tmp;

                // stretching contributions
                t_coeff[0] += (ks_inv_len * t_dot_delta_e_diag[0]);
                Vec2 delta_e_coeff = finite_xport_coeff;
                delta_e_coeff[0] += ks_inv_len - ks * inv_restlen;

                delta_dE_de.noalias() += (delta_e * inv_len) * (-coeff3 * inv_len).transpose()
                                      - delta_e * (delta_e_coeff).asDiagonal()
                                      + t_tilde * t_tilde_coeff.transpose()
                                      + t_cross_kb * t_cross_kb_coeff.asDiagonal()
                                      - coeff_a * weighted_cross_prod_term;
            }

            delta_dE_de.col(0) += (dE_dkappa_k_j(0, 0) * ilen_sq_kb_dot_delta_e[0] - dE_dkappa_k_j(1, 0) * ilen_sq_tb_cross_kb_dot_delta_e[0]) * dc.materialFrame[i - 1].d2
                                - (dE_dkappa_k_j(1, 0) * ilen_sq_kb_dot_delta_e[0] + dE_dkappa_k_j(0, 0) * ilen_sq_tb_cross_kb_dot_delta_e[0]) * dc.materialFrame[i - 1].d1;
            delta_dE_de.col(1) += (dE_dkappa_k_j(0, 1) * ilen_sq_kb_dot_delta_e[1] - dE_dkappa_k_j(1, 1) * ilen_sq_tb_cross_kb_dot_delta_e[1]) * dc.materialFrame[i].d2
                                - (dE_dkappa_k_j(1, 1) * ilen_sq_kb_dot_delta_e[1] + dE_dkappa_k_j(0, 1) * ilen_sq_tb_cross_kb_dot_delta_e[1]) * dc.materialFrame[i].d1;

            const Real_ beta_i_inv_2libar_delta_m = beta_div_2libar * (0.5 * invlen_kb_dot_delta_e.sum() + delta_theta[1] - delta_theta[0]);
            delta_dE_dtheta += Eigen::Vector2d(-2, 2) * beta_i_inv_2libar_delta_m;

            Vec3 vector_for_crossing = (two_inv_chilen_neg[0] * inv_len[1]) *
                (dE_dkappa_k_j(0, 0) * dc.materialFrame[i - 1].d2 - dE_dkappa_k_j(1, 0) * dc.materialFrame[i - 1].d1
               + dE_dkappa_k_j(0, 1) * dc.materialFrame[i    ].d2 - dE_dkappa_k_j(1, 1) * dc.materialFrame[i    ].d1);
            delta_dE_de.col(0) -= ((two_inv_chilen_neg[0] * half_invlen_dE_dm[1]) * t.col(0) - vector_for_crossing).cross(delta_e.col(1));
            delta_dE_de.col(1) -= ((two_inv_chilen_neg[1] * half_invlen_dE_dm[0]) * t.col(1) + vector_for_crossing).cross(delta_e.col(0));

            kb_coeff[0] += inv_len[0] * (coeff2[0] * (t_dot_delta_e_diag[0] + t_tilde_dot_delta_e[0]) + beta_i_inv_2libar_delta_m - half_invlen_dE_dm[1] * t_tilde_dot_delta_e[1]);
            kb_coeff[1] += inv_len[1] * (coeff2[1] * (t_dot_delta_e_diag[1] + t_tilde_dot_delta_e[1]) + beta_i_inv_2libar_delta_m - half_invlen_dE_dm[0] * t_tilde_dot_delta_e[0]);
        }

        /////////////////////////////////////////////
        // Rest length derivatives
        /////////////////////////////////////////////
        if (variableRestLen && (mask.restlen_in || mask.restlen_out)) {
            const size_t rl_offset = theta_offset + ne; // Index of the first rest length variable for the stencil
            const Vec2 delta_rlen = v.template segment<2>(rl_offset);

            // Derivative of the bending energy with respect to (kappa_k)_i^j
            Vec2 d2E_dkappa_k_j_dljbar, d2E_dkappa_k_j_dljotherbar;
            if (m_bendingEnergyType == BendingEnergyType::Bergou2010) {
                const Vec2 neg_kappaDiff = m_restKappa[i] - dc.kappa[i];
                d2E_dkappa_k_j_dljbar = inv_2libar * B_div_2libar.asDiagonal() * neg_kappaDiff;
                d2E_dkappa_k_j_dljotherbar = d2E_dkappa_k_j_dljbar;
                delta_dE_drlen.array() = (delta_rlen.sum() * (2 * inv_2libar)) * d2E_dkappa_k_j_dljbar.dot(neg_kappaDiff);
            }

            for (size_t adj_edge = 0; adj_edge < 2; ++adj_edge) { // 0 ==> i - 1, 1 ==> i
                if (m_bendingEnergyType == BendingEnergyType::Bergou2008) {
                    const Vec2 kappaDiff = dc.per_corner_kappa[i].col(adj_edge) - m_restKappa[i];
                    d2E_dkappa_k_j_dljbar      = (     inv_2libar * inv_2libar) /* m_restLen[jother] - m_restLen[j] */ * (B_div_2libar.asDiagonal() * kappaDiff); // omit 2 * restlen factor for now to allow reusing this term in computing "contrib" below
                    d2E_dkappa_k_j_dljotherbar = (-4 * inv_2libar * inv_2libar)  * (                 rlen[adj_edge]  ) * (B_div_2libar.asDiagonal() * kappaDiff);

                    Real_ contrib = inv_2libar * d2E_dkappa_k_j_dljbar.dot(kappaDiff);

                    d2E_dkappa_k_j_dljbar *= 2 * (rlen[1 - adj_edge] - rlen[adj_edge]);

                    Real_ coeff   = 4 *  rlen[adj_edge] - 2 * rlen[1 - adj_edge];
                    Real_ two_sum = 2 * rlen.sum();

                    Vec2 signed_two_sum(two_sum, two_sum);
                    signed_two_sum[adj_edge] *= -1;
                    // Equivalent to:
                    //      size_t var      = rl_offset +     adj_edge,
                    //             varOther = rl_offset + 1 - adj_edge;
                    //      result[var     ] += contrib * ((coeff - two_sum) * v[var     ] + coeff * v[varOther]);
                    //      result[varOther] += contrib * ((coeff + two_sum) * v[varOther] + coeff * v[var     ]);
                    delta_dE_drlen.array() += contrib * (coeff * delta_rlen.sum() + signed_two_sum.array() * delta_rlen.array());
                }

                // Accumulate energy dependence through (kappa_k)_i^j
                for (size_t k = 0; k < 2; ++k) {
                    // The formulas below are more efficient if we swap the "j" and "jother" labels when (adj_edge == 1).
                    Vec2 d2E_dkappa_k_swapped(d2E_dkappa_k_j_dljbar[k],
                                              d2E_dkappa_k_j_dljotherbar[k]);
                    if (adj_edge == 1) std::swap(d2E_dkappa_k_swapped[0], d2E_dkappa_k_swapped[1]);
                    const Real_ delta_kappa = d2E_dkappa_k_swapped.dot(delta_rlen);

                    delta_dE_dtheta[adj_edge] += d_kappa_k_j_dtheta_j(k, adj_edge) * delta_kappa;

                    d_kappa_k_j_de_coeff(k, adj_edge) += delta_kappa;
                    delta_dE_drlen += d2E_dkappa_k_swapped * delta_kappa_k_j(k, adj_edge);
                }
            }

            const Real_ d2E_dljbar_dm = -dE_dm * inv_2libar;
            const Real_ delta_total_restlen_factor = d2E_dljbar_dm * (v[rl_offset] + v[rl_offset + 1]);

            kb_coeff += (0.5 * delta_total_restlen_factor) * inv_len;

            delta_dE_dtheta[0] -= delta_total_restlen_factor;
            delta_dE_dtheta[1] += delta_total_restlen_factor;

            const Real_ fracLen = dc.len[i - 1] * inv_restlen;
            const Real_ ks_coeff = ks * fracLen * inv_restlen;
            if (mask.restlen_out) {
                const Real_ contrib = d2E_dljbar_dm * ((0.5 * invlen_kb_dot_delta_e.sum()) + (delta_theta[1] - delta_theta[0])) - inv_2libar * m * delta_total_restlen_factor;
                result[rl_offset    ] += contrib + ks_coeff * (fracLen * delta_rlen[0] - t.col(0).dot(delta_e.col(0)));
                result[rl_offset + 1] += contrib;
                result.template segment<2>(rl_offset) += delta_dE_drlen;
            }

            // stretching contrib to delta_dE_de
            t_coeff[0] -= ks_coeff * delta_rlen[0];
        }

        delta_dE_de += d_kappa_k_j_de_coeff(0, 0) * d_kappa_k_j_de[0][0]
                    +  d_kappa_k_j_de_coeff(0, 1) * d_kappa_k_j_de[0][1]
                    +  d_kappa_k_j_de_coeff(1, 0) * d_kappa_k_j_de[1][0]
                    +  d_kappa_k_j_de_coeff(1, 1) * d_kappa_k_j_de[1][1]
                    +  kb * kb_coeff.transpose()
                    +  t * t_coeff.asDiagonal();

        result.template segment<3>(x_offset + 0) -= delta_dE_de.col(0);
        result.template segment<3>(x_offset + 3) += delta_dE_de.col(0) - delta_dE_de.col(1);
        result.template segment<3>(x_offset + 6) += delta_dE_de.col(1);
        result.template segment<2>(theta_offset) += delta_dE_dtheta;
    }

    // Stretching term for final edge (not accumulated in internal vertex loop above)
    {
        const size_t j = ne - 1;
        const Real_ ks = density(j) * m_stretchingStiffness[j];
        const Real_ inv_restlen = 1.0 / m_restLen[j];
        const Real_ ks_inv_len = ks / dc.len[j];
        const auto &t = dc.tangent[j];

        const size_t x_offset = 3 * j;
        const Vec3 delta_e = v.template segment<3>(x_offset + 3) - v.template segment<3>(x_offset);

        const Real_ t_dot_delta_e = t.dot(delta_e);
        Real_ t_coeff = (ks_inv_len * t_dot_delta_e);

        if (variableRestLen) {
            const size_t rl_offset = 3 * nv + ne + j;
            Real_ fracLen = dc.len[j] * inv_restlen;
            Real_ coeff = ks * fracLen * inv_restlen;
            t_coeff -= coeff * v[rl_offset];
            result[rl_offset] += coeff * (fracLen * v[rl_offset] - t_dot_delta_e);
        }

        Vec3 delta_dE_de = (ks * inv_restlen - ks_inv_len) * delta_e + t_coeff * t;

        result.template segment<3>(3 * (j + 1)) += delta_dE_de;
        result.template segment<3>(3 * (j    )) -= delta_dE_de;
    }
}
