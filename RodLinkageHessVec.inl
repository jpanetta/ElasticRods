////////////////////////////////////////////////////////////////////////////////
// RodLinkageHessVec.inl
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Elastic energy Hessian-vector product formulas for RodLinkage.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  12/09/2018 15:31:49
////////////////////////////////////////////////////////////////////////////////
#ifndef RODLINKAGEHESSVEC_INL
#define RODLINKAGEHESSVEC_INL

template<typename Real_>
struct RodHessianApplierData {
    VecX_T<Real_> v_local, Hv_local, Hv;
    bool constructed = false;
};

#if MESHFEM_WITH_TBB
template<typename Real_>
using RHALocalData = tbb::enumerable_thread_specific<RodHessianApplierData<Real_>>;

template<typename F, typename Real_>
struct RodHessianApplier {
    RodHessianApplier(F &f, const size_t nvars, RHALocalData<Real_> &locals) : m_f(f), m_nvars(nvars), m_locals(locals) { }

    void operator()(const tbb::blocked_range<size_t> &r) const {
        RodHessianApplierData<Real_> &data = m_locals.local();
        if (!data.constructed) { data.Hv.setZero(m_nvars); data.constructed = true; }
        for (size_t si = r.begin(); si < r.end(); ++si) { m_f(si, data); }
    }
private:
    F &m_f;
    size_t m_nvars;
    RHALocalData<Real_> &m_locals;
};

template<typename F, typename Real_>
RodHessianApplier<F, Real_> make_rod_hessian_applier(F &f, size_t nvars, RHALocalData<Real_> &locals) {
    return RodHessianApplier<F, Real_>(f, nvars, locals);
}
#endif

template<typename Real_>
auto RodLinkage_T<Real_>::applyHessian(const VecX &v, bool variableRestLen, const HessianComputationMask &mask) const -> VecX {
    BENCHMARK_SCOPED_TIMER_SECTION timer("RodLinkage_T::applyHessian");
    const size_t ndof = variableRestLen ? numExtendedDoF() : numDoF();
    if (size_t(v.size()) != ndof) throw std::runtime_error("Input vector size mismatch");

    // Our Hessian can only be evaluated after the source configuration has
    // been updated; use the more efficient gradient formulas.
    const bool updatedSource = true;
    {
        const bool hessianNeeded = mask.dof_in && mask.dof_out; // joint parametrization Hessian only needed for dof-dof part
        if (hessianNeeded) m_sensitivityCache.update(*this, updatedSource, v); // directional derivative only
        else               m_sensitivityCache.update(*this, updatedSource, false); // In all cases, we need at least the Jacobian
    }

    auto applyPerSegmentHessian = [&](const size_t si, RodHessianApplierData<Real_> &data) {
        VecX & v_local = data.v_local;
        VecX &Hv_local = data.Hv_local;
        VecX &Hv       = data.Hv;

        const auto &s = m_segments[si];
        const auto &r = s.rod;

        std::array<const LinkageTerminalEdgeSensitivity<Real_> *, 2> jointSensitivity{{ nullptr, nullptr }};
        std::array<size_t, 2> segmentJointDofOffset, segmentJointRestLenDofOffset;
        for (size_t i = 0; i < 2; ++i) {
            size_t ji = s.joint(i);
            if (ji == NONE) continue;
            jointSensitivity[i] = &m_sensitivityCache.lookup(si, static_cast<TerminalEdge>(i));
            segmentJointDofOffset[i] = m_dofOffsetForJoint[ji];

            if (variableRestLen) {
                // Index of rest global length variable controlling segment si's end at local joint i
                size_t abOffset = joint(ji).segmentABOffset(si);
                assert(abOffset != NONE);
                segmentJointRestLenDofOffset[i] = m_restLenDoFOffsetForJoint[ji] + abOffset;
            }
        }

        const size_t ndof_local = variableRestLen ? s.rod.numExtendedDoF() : s.rod.numDoF();

        // Apply dv_dr (compute the perturbation of the rod variables).
        v_local.resize(ndof_local);
        if (mask.dof_in) {
            // Copy over the interior/free-end vertex and theta perturbations.
            const size_t free_vtx_components = 3 * s.numFreeVertices(),
                         local_theta_offset = s.rod.thetaOffset();
            v_local.segment((3 * 2) * s.hasStartJoint(),         free_vtx_components) = v.segment(m_dofOffsetForSegment[si], free_vtx_components);
            v_local.segment(local_theta_offset + s.hasStartJoint(), s.numFreeEdges()) = v.segment(m_dofOffsetForSegment[si] + free_vtx_components, s.numFreeEdges());

            // Compute the perturbations of the constrained vertex/theta variables.
            for (size_t lji = 0; lji < 2; ++lji) {
                size_t ji = s.joint(lji);
                if (ji == NONE) continue;
                const auto &js = *jointSensitivity[lji];
                const size_t jo = m_dofOffsetForJoint[ji];

                Eigen::Matrix<Real_, 4, 1> delta_e_theta = js.jacobian * v.template segment<6>(jo + 3);
                v_local.template segment<3>(3 * (js.j + 1)) = v.template segment<3>(jo) + (js.s_jX * 0.5) * delta_e_theta.template segment<3>(0);
                v_local.template segment<3>(3 * (js.j    )) = v.template segment<3>(jo) - (js.s_jX * 0.5) * delta_e_theta.template segment<3>(0);
                v_local[local_theta_offset + js.j] = delta_e_theta[3];
            }
        }
        else { v_local.head(s.rod.numDoF()).setZero(); }
        if (variableRestLen) {
            if (mask.restlen_in) {
                const size_t local_rl_offset = s.rod.restLenOffset();
                // Copy over the interior/free-end edge rest length perturbations.
                v_local.segment(local_rl_offset + s.hasStartJoint(), s.numFreeEdges())
                    = v.segment(m_restLenDoFOffsetForSegment[si], s.numFreeEdges());

                // Copy constrained terminal edges' rest length perturbations from their controlling joint.
                for (size_t lji = 0; lji < 2; ++lji) {
                    size_t ji = s.joint(lji);
                    if (ji == NONE) continue;
                    const auto &js = *jointSensitivity[lji];
                    v_local[local_rl_offset + js.j] = v[m_restLenDoFOffsetForJoint[ji] + (js.is_A ? 0 : 1)];
                }
            }
            else { v_local.tail(s.rod.numRestLengths()).setZero(); }
        }

        // Apply rod Hessian
        Hv_local.setZero(ndof_local);
        r.applyHessEnergy(v_local, Hv_local, variableRestLen, mask);

        // Apply dv_dr transpose (accumulate contribution to output gradient)
        if (mask.dof_out) {
            // Copy over the interior/free-end vertex and theta delta grad components
            const size_t free_vtx_components = 3 * s.numFreeVertices(),
                         local_theta_offset = s.rod.thetaOffset();
            Hv.segment(m_dofOffsetForSegment[si],                    free_vtx_components) = Hv_local.segment((3 * 2) * s.hasStartJoint(), free_vtx_components);
            Hv.segment(m_dofOffsetForSegment[si] + free_vtx_components, s.numFreeEdges()) = Hv_local.segment(local_theta_offset + s.hasStartJoint(), s.numFreeEdges());

            // Compute the perturbations of the constrained vertex/theta variables.
            for (size_t lji = 0; lji < 2; ++lji) {
                size_t ji = s.joint(lji);
                if (ji == NONE) continue;
                const auto &js = *jointSensitivity[lji];
                const size_t jo = m_dofOffsetForJoint[ji];

                Eigen::Matrix<Real_, 4, 1> delta_grad_e_theta;
                delta_grad_e_theta.template segment<3>(0) = (0.5 * js.s_jX) * (Hv_local.template segment<3>(3 * (js.j + 1)) - Hv_local.template segment<3>(3 * js.j));
                delta_grad_e_theta[3] = Hv_local[local_theta_offset + js.j];

                Hv.template segment<3>(jo    ) += Hv_local.template segment<3>(3 * (js.j + 1)) + Hv_local.template segment<3>(3 * js.j); // Joint position identity block
                Hv.template segment<6>(jo + 3) += js.jacobian.transpose() * delta_grad_e_theta;                                          // Joint orientation/angle/length Jacobian block
            }
        }
        if (variableRestLen && mask.restlen_out) {
            const size_t local_rl_offset = s.rod.restLenOffset();
            // Copy over the interior/free-end edge rest length delta grad components.
            Hv.segment(m_restLenDoFOffsetForSegment[si], s.numFreeEdges()) = Hv_local.segment(local_rl_offset + s.hasStartJoint(), s.numFreeEdges());

            // Accumulate constrained terminal edges' rest length delta grad components to their controlling joint.
            for (size_t lji = 0; lji < 2; ++lji) {
                size_t ji = s.joint(lji);
                if (ji == NONE) continue;
                const auto &js = *jointSensitivity[lji];
                Hv[m_restLenDoFOffsetForJoint[ji] + (js.is_A ? 0 : 1)] += Hv_local[local_rl_offset + js.j];
            }
        }

        // Compute joint Hessian term.
        if (mask.dof_in && mask.dof_out) {
            // typename ElasticRod_T<Real_>::Gradient sg(r);
            // sg.setZero();
            // Note: we only need the gradient with respect to the terminal
            // degrees of freedom, so we can ignore many of the energy contributions.
            const auto sg = r.template gradient<GradientStencilMaskTerminalsOnly>(updatedSource); // we never need the variable rest length gradient since the mapping from global to local rest lengths is linear

            // Accumulate contribution of the Hessian of e^j and theta^j wrt the joint parameters.
            //      dE/var^j (d^2 var^j / djoint_var_k djoint_var_l)
            for (size_t ji = 0; ji < 2; ++ji) {
                if (jointSensitivity[ji] == nullptr) continue;
                const auto &js = *jointSensitivity[ji];
                const size_t o = segmentJointDofOffset[ji] + 3; // DoF index for first component of omega
                Eigen::Matrix<Real_, 4, 1> dE_djointvar;
                dE_djointvar.template segment<3>(0) = (0.5 * js.s_jX) * (sg.gradPos(js.j + 1) - sg.gradPos(js.j));
                dE_djointvar[3]                     = sg.gradTheta(js.j);
                Hv.template segment<6>(o).noalias() += js.delta_jacobian.transpose() * dE_djointvar;
            }
        }
    };

    VecX result(VecX::Zero(v.size()));
#if MESHFEM_WITH_TBB
    RHALocalData<Real_> rhaLocalData;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, numSegments()), make_rod_hessian_applier(applyPerSegmentHessian, v.size(), rhaLocalData));

    for (const auto &data : rhaLocalData)
        result += data.Hv;
#else
    RodHessianApplierData<Real_> data;
    data.Hv.setZero(result.size());
    for (size_t si = 0; si < numSegments(); ++si)
        applyPerSegmentHessian(si, data);
    result = data.Hv;
#endif

    return result;
}

template<typename Real_>
auto RodLinkage_T<Real_>::applyHessianPerSegmentRestlen(const VecX &v, const HessianComputationMask &mask) const -> VecX {
    BENCHMARK_SCOPED_TIMER_SECTION timer("RodLinkage_T::applyHessianPSRL");
    const size_t ndof = numExtendedDoFPSRL();
    if (size_t(v.size()) != ndof) throw std::runtime_error("Input vector size mismatch");

    VecX vPerEdge(numExtendedDoF());
    vPerEdge.head(numDoF()) = v.head(numDoF());
    if (mask.restlen_in) m_segmentRestLenToEdgeRestLenMapTranspose.applyRaw(v.tail(numSegments()).data(), vPerEdge.tail(numRestLengths()).data(), /* transpose */ true);
    else                 vPerEdge.tail(numRestLengths()).setZero();
    auto HvPerEdge = applyHessian(vPerEdge, true, mask);

    VecX result(v.size());
    result.head(numDoF()) = HvPerEdge.head(numDoF());
    if (mask.restlen_out) m_segmentRestLenToEdgeRestLenMapTranspose.applyRaw(HvPerEdge.tail(numRestLengths()).data(), result.tail(numSegments()).data(), /* no transpose */ false);
    else                  result.tail(numSegments()).setZero();

    return result;
}

#endif /* end of include guard: RODLINKAGEHESSVEC_INL */
