////////////////////////////////////////////////////////////////////////////////
// PeriodicRod.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  This class represents a closed loop formed by gluing together the ends of
//  an elastic rod. To ensure stretching, bending, and twisting elastic
//  energies are calculated properly, the first and last ends of the rod are
//  constrained to overlap, and the stretching stiffness of this rod is halved
//  to avoid double-counting. An additional twist variable is introduced to allow a
//  twist discontinuity at the joint. This variable specifies the offset in frame
//  angle theta between the two overlapping edges (measured from the last edge
//  to the first). By fixing this angle to a constant, nonzero twist can be
//  maintained in the rod.
//
//  The ends are glued together with the following simple equality constraints on
//  the deformation variables:
//      x_{nv - 2}     = x_0
//      x_{nv - 1}     = x_1
//      theta_{ne - 1} = theta_0 + twist
//
//  We implement these constraints efficiently with a change of deformation
//  variables from "unreduced" variables
//      [x_0, ..., x_{nv - 1}, theta_0, ..., theta_{ne - 1}]
//  to "reduced" variables
//      [x_0, ..., x_{nv - 3}, theta_0, ..., theta_{ne - 2}, twist]
//
//  In matrix form, this linear change of variables looks like:
//                  [I_6 0             0 0           0][ x_0 \\ x_1              ]
//                  [0   I_{3(nv - 2)} 0 0           0][ x_2 .. x_{nv - 3}       ]
//  unreducedVars = [I_6 0             0 0           0][ theta_0                 ]
//                  [0   0             1 0           0][ theta_1..theta_{ne - 2} ]
//                  [0   0             0 I_{ne - 1}  0][ twist                   ]
//                  [0   0             1 0           1] 
//                  \_______________ J ______________/ \______ reducedVars _____/
//  where `J` is the sparse Jacobian matrix of the change of variables consisting of
//  an arrangement of Identity blocks.
//
//  The periodic rod's elastic energy gradient is obtained by applying J^T to
//  the underlying rod's gradient.
//  The periodic rod's elastic energy Hessian is obtained from the underlying
//  rod's Hessian H as:
//      H_reduced = J^T H J
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  06/07/2021 13:42:56
////////////////////////////////////////////////////////////////////////////////
#ifndef PERIODICROD_HH
#define PERIODICROD_HH
#include "ElasticRod.hh"
#include "/home/mvidulis/dev/ElasticRods/3rdparty/MeshFEM/src/lib/MeshFEM/../MeshFEM/Geometry.hh"  // TODO KNOTS: fix include
#include <algorithm>

// Templated to support automatic differentiation types.
template<typename Real_>
struct PeriodicRod_T;

using PeriodicRod = PeriodicRod_T<Real>;

template<typename Real_>
struct PeriodicRod_T {
    using Rod    = ElasticRod_T<Real_>;
    using Pt3    = Pt3_T<Real_>;
    using Vec2   = Vec2_T<Real_>;
    using VecX   = VecX_T<Real_>;
    using CSCMat = CSCMatrix<SuiteSparse_long, Real_>;
    using StdVectorVector2D = std::vector<Vec2, Eigen::aligned_allocator<Vec2>>; // Work around alignment issues.
    using EnergyType = typename Rod::EnergyType;

    // Assumption: `rod` is an ElasticRod initialized in the following way:
    // - reference frame is zero-twist (we choose a random normal director for edge 0 and we parallel-transport it in space);
    // - material frame coincides with reference frame (all thetas are zero).
    PeriodicRod_T(const std::vector<Pt3> &points, bool zeroRestCurvature = false)
        : rod(points)
    {
        const size_t nv = rod.numVertices();
        if (((points[0] - points[nv - 2]).norm() > 1e-12) ||
            ((points[1] - points[nv - 1]).norm() > 1e-12)) throw std::runtime_error("First and last edge must overlap!");

        // Overwrite final edge's reference frame with the 
        auto restDirectors = rod.restDirectors();
        restDirectors.back() = restDirectors.front();
        rod.setRestDirectors(restDirectors);

        // Recompute rest curvature's decomposition in the updated frame (though it could simply be rotated...)
        // or reset it to zero if requested.
        StdVectorVector2D restKappa(nv, Vec2::Zero());
        if (!zeroRestCurvature) {
            for (size_t i = 1; i < nv - 1; ++i) {
                auto kb = curvatureBinormal(rod.restEdgeVector(i - 1).normalized(), rod.restEdgeVector(i).normalized());
                restKappa[i] = Vec2(0.5 * kb.dot(restDirectors[i - 1].d2 + restDirectors[i].d2),
                                   -0.5 * kb.dot(restDirectors[i - 1].d1 + restDirectors[i].d1));
            }
        }

        rod.setRestKappas(restKappa);
        rod.deformedConfiguration().initialize(rod);
    }

    PeriodicRod_T(const ElasticRod &rod, Real twist)
        : rod(rod), m_twist(twist) 
    {  
        const size_t nv = rod.numVertices();
        const std::vector<Pt3> &points = rod.deformedPoints();
        if (((points[0] - points[nv - 2]).norm() > 1e-12) ||
            ((points[1] - points[nv - 1]).norm() > 1e-12)) throw std::runtime_error("First and last edge must overlap!");
    }

    // Converting constructor from another floating point type (e.g., double to autodiff)
    template<typename Real2>
    PeriodicRod_T(const PeriodicRod_T<Real2> &pr) : rod(pr.rod) { }

    // Set a homogeneous material for the rod
    void setMaterial(const RodMaterial &mat) {
        rod.setMaterial(mat);
        // Avoid double-counting stiffness/mass for the overlapping edge.
        rod.density(0) = 0.5;
        rod.density(rod.numEdges() - 1) = 0.5;
    }

    size_t numDoF()      const { return rod.numDoF() - 6; } // we remove the last two endpoint position variables.
    size_t numElems()    const { return rod.numVertices() - 2; } // #nodes == #edges in periodic rods
    size_t numVertices(bool countGhost = false) const { return countGhost ? numElems() + 2 : numElems(); }
    size_t numEdges   (bool countGhost = false) const { return countGhost ? numElems() + 1 : numElems(); }
    size_t thetaOffset() const { return 3 * (rod.numVertices() - 2); }

    VecX getDoFs() const {
        VecX result(numDoF());
        VecX unreducedDoFs = rod.getDoFs();
        result.head   (3 * (rod.numVertices() - 2))                     = unreducedDoFs.head   (3 * (rod.numVertices() - 2));
        result.segment(3 * (rod.numVertices() - 2), rod.numEdges() - 1) = unreducedDoFs.segment(3 * rod.numVertices(), rod.numEdges() - 1);
        result[result.size() - 1] = m_twist;
        return result;
    }

    Pt3 getNode(size_t ni) const { assert(ni < numVertices()); return rod.deformedConfiguration().point(ni); }

    VecX applyJacobian(const Eigen::Ref<const VecX> &dofs) const {
        if (size_t(dofs.size()) != numDoF()) throw std::runtime_error("DoF vector has incorrect length.");
        VecX unreducedDoFs(rod.numDoF());
        unreducedDoFs.head   (3 * (rod.numVertices() - 2))               = dofs.head   (3 * (rod.numVertices() - 2));
        unreducedDoFs.template segment<6>(3 * (rod.numVertices() - 2))   = dofs.template head<6>();
        unreducedDoFs.segment(3 * rod.numVertices(), rod.numEdges() - 1) = dofs.segment(3 * (rod.numVertices() - 2), rod.numEdges() - 1);
        unreducedDoFs[unreducedDoFs.size() - 1] = unreducedDoFs[rod.thetaOffset()] + dofs[dofs.size() - 1];
        return unreducedDoFs;
    }

    void setDoFs(const Eigen::Ref<const VecX> &dofs) {
        m_twist = dofs[dofs.size() - 1];
        rod.setDoFs(applyJacobian(dofs));
    }

    Real_ twist()    const { return m_twist; }
    void setTwist(Real_ t) { m_twist = t; setDoFs(getDoFs()); }

    Real_ energy(EnergyType etype = EnergyType::Full) const { return rod.energy(etype); }
    Real_ energyStretch() const { return rod.energyStretch(); }
    Real_ energyBend()    const { return rod.energyBend(); }
    Real_ energyTwist()   const { return rod.energyTwist(); }
    VecX gradient(bool updatedSource = false, EnergyType etype = EnergyType::Full) const {
        auto unreducedGradient = rod.gradient(updatedSource, etype, /* variableDesignParameters = */ false, /* designParameterOnly = */ false);
        VecX result(numDoF());

        // Apply the transposed Jacobian
        // First two column blocks of J^T
        result.head   (3 * (rod.numVertices() - 2))                     = unreducedGradient.head   (3 * (rod.numVertices() - 2));
        result.segment(3 * (rod.numVertices() - 2), rod.numEdges() - 1) = unreducedGradient.segment(3 * rod.numVertices(), rod.numEdges() - 1);

        // Third column block of J^T
        result.template segment<6>(0) += unreducedGradient.template segment<6>(3 * (rod.numVertices() - 2));

        // Column blocks 4 and 5 of J^T
        Real_ gradLastTwist = unreducedGradient[unreducedGradient.size() - 1];

        // Column block 6 of J^T
        result[3 * (rod.numVertices() - 2)] += gradLastTwist;
        result[result.size() - 1]            = gradLastTwist;

        return result;
    }

    // Hout = J^T H J. This has the effect of rewriting the
    // row/column indices for all triplets, except for the last row/column of
    // H, whose indices get duplicated into a +/- copy.
    template<class SPMat>
    void reduceHessian(const CSCMat &H, SPMat &Hout) const {
        const size_t nv = rod.numVertices();
        const size_t reducedPosVars = 3 * (nv - 2);
        const size_t unreducedPosVars = 3 * nv;
        const size_t unreducedVars = rod.numDoF();
        const size_t firstReducedTheta = reducedPosVars;

        // rewrite unreduced index i to its (first) corresponding reduced index
        auto reducedVarIdx = [&](size_t i) -> size_t { 
            if (i < reducedPosVars)    return i;
            if (i < unreducedPosVars)  return i - reducedPosVars; // first 6 displacement variables
            if (i < unreducedVars - 1) return i - unreducedPosVars + firstReducedTheta;
            return firstReducedTheta;
        };

        const size_t lastTheta = unreducedVars - 1;
        const size_t  twistVar = numDoF() - 1;

        auto emitNZ = [&](size_t i, size_t j, Real_ v) {
            if (i > j) return; // omit entries in the lower triangle
            Hout.addNZ(i, j, v);
        };

        for (const auto &t : H) {
            // Note: triplet `t` is in the upper triangle of H; we want to generate
            // the upper triangle of Hout = J^T H J.
            int ri = reducedVarIdx(t.i), rj = reducedVarIdx(t.j);
            emitNZ(ri, rj, t.v);
            if (t.i != t.j) {
                emitNZ(rj, ri, t.v);
                // Generate the extra triplets produced by the dependency of the
                // unreduced theta variable on m_twist.
                if (t.i == lastTheta) { emitNZ(twistVar, rj, t.v); emitNZ(rj, twistVar, t.v); }
                if (t.j == lastTheta) { emitNZ(twistVar, ri, t.v); emitNZ(ri, twistVar, t.v); }
            }
            else if (t.i == lastTheta) {
                // Generate the extra diagonal entry produced by the dependency of the
                // unreduced theta variable on m_twist.
                emitNZ(ri, twistVar, t.v);
                emitNZ(twistVar, twistVar, t.v);
            }
        }
    }

    // Optimizers like Knitro and Ipopt need to know all Hessian entries that
    // could ever possibly be nonzero throughout the course of optimization.
    // The current Hessian may be missing some of these entries.
    // Knowing the fixed sparsity pattern also allows us to more efficiently construct the Hessian.
    CSCMat hessianSparsityPattern(Real_ val = 0.0) const {
        if (m_cachedHessianSparsityPattern.m == 0) {
            auto Hsp = rod.hessianSparsityPattern(false, 1.0);

            TripletMatrix<Triplet<Real_>> Htrip(numDoF(), numDoF());
            Htrip.symmetry_mode = TripletMatrix<Triplet<Real_>>::SymmetryMode::UPPER_TRIANGLE;
            Htrip.reserve(Hsp.nnz());
            reduceHessian(Hsp, Htrip);
            m_cachedHessianSparsityPattern = CSCMat(Htrip);
        }

        m_cachedHessianSparsityPattern.fill(val);
        return m_cachedHessianSparsityPattern;
    }

    void hessian(CSCMat &H, EnergyType etype = EnergyType::Full) const {
        CSCMat H_unreduced = m_getCachedUnreducedHessianSparsityPattern();
        rod.hessian(H_unreduced, etype, /* variableDesignParameters = */ false);

        H = hessianSparsityPattern(0.0);
        reduceHessian(H_unreduced, H);
    }

    CSCMat hessian(EnergyType etype = EnergyType::Full) const {
        CSCMat H;
        hessian(H, etype);
        return H;
    }

    // Note: the "lumped mass matrix" is not perfectly diagonal due to the "twist" variable's
    // coupling with the first theta variable.
    void massMatrix(CSCMat &M, bool updatedSource = false, bool useLumped = false) const {
        CSCMat M_unreduced = m_getCachedUnreducedHessianSparsityPattern();
        rod.massMatrix(M_unreduced, updatedSource, useLumped);

        M = hessianSparsityPattern(0.0);
        reduceHessian(M_unreduced, M);
    }

    CSCMat massMatrix() const {
        CSCMat M;
        massMatrix(M);
        return M;
    }

    // Set a new deformed configuration without affecting the current opening angle (m_twist)
    void setDeformedConfiguration(const std::vector<Pt3> &points, const std::vector<Real> &thetas) {
        if (points.size() != numVertices()) throw std::runtime_error("setDeformedConfiguration: number of deformed points (" + std::to_string(points.size()) + ") should match number of vertices (" + std::to_string(numVertices()) + ").");
        if (thetas.size() != numEdges())    throw std::runtime_error("setDeformedConfiguration: number of deformed thetas (" + std::to_string(thetas.size()) + ") should match number of edges (" + std::to_string(numEdges()) + ").");
        std::vector<Pt3>  extendedPoints(points.begin(), points.end());
        std::vector<Real> extendedThetas(thetas.begin(), thetas.end());
        extendedPoints.push_back(points[0]);
        extendedPoints.push_back(points[1]);
        extendedThetas.push_back(m_twist + thetas[0]);
        rod.setDeformedConfiguration(extendedPoints, extendedThetas);
    }

    // Set a new deformed configuration changing the opening angle (m_twist)
    void setDeformedConfiguration(const std::vector<Pt3> &points, const std::vector<Real> &thetas, Real twist) {
        m_twist = twist;
        setDeformedConfiguration(points, thetas);
    }

    // Compute the total twist, namely the number of "turns" that the material frame does compared to a zero-twist (natural) frame.
    // The output corresponds to int_0^L u_3(s) ds, where u_3 is the component of the Darboux vector of the material frame in the tangential direction.
    // The integral coincides with the cumulated twist of the reference frame plus the total opening angle
    Real totalTwistAngle() const { return totalReferenceTwistAngle() + m_twist; }

    // Cumulated twist angle in reference frame
    Real totalReferenceTwistAngle() const {
        const std::vector<Real> &refTwist = rod.deformedConfiguration().referenceTwist; // twist in reference frame, always of the form [0, tw_1, ..., tw_nv, 0]
        return std::accumulate(refTwist.begin() + 1, refTwist.end() - 1, 0.0);
    }

    // Angle \Phi \in [0, 2*pi).
    // Defines the rotation R_{\Phi} that makes the material frame of the first edge coincide with the one of the last edge.
    Real openingAngle() const {
        Real dummy_integer_part;
        Real v = 2*M_PI * std::modf(m_twist / (2*M_PI), &dummy_integer_part);  // in (-2*pi, 2*pi), if x < 0 then std::modf(x) < 0
        return v >= 0 ? v : v + 2*M_PI;
    }

    // Compute the writhe of the piecewise linear centerline. 
    // Use the approach from [Swigon et al. 1998] and [Klenin and Langowski 2000].
    // Note: quadratic complexity in the number of nodes, can be slow for thousands of nodes.
    Real writhe() const {
        Real tol = 1e-10;
        const size_t n = numVertices(); // == numEdges()
        std::vector<Pt3> nodes = deformedPoints();

        std::vector<Eigen::Vector3d> edges(n);
        std::vector<Eigen::Vector3d> tangents(n);
        for (size_t i = 0; i < n-1; i++) {
            edges[i] = nodes[i+1] - nodes[i];
            tangents[i] = edges[i].normalized();
        }
        edges[n-1] = nodes[0] - nodes[n-1];
        tangents[n-1] = edges[n-1].normalized();

        auto sign = [](const Real &x){ return (Real(0) < x) - (x < Real(0)); };
        auto mu = [&](const Eigen::Vector3d &ti, const Eigen::Vector3d &dmk, const Eigen::Vector3d &tj) { // see [Swigon et al. 1998]
            const Eigen::Vector3d ti_cross_tj = ti.cross(tj);
            assert(ti_cross_tj.norm() > tol);
            if (dmk.norm() < tol)
                return M_PI/2;
            const Eigen::Vector3d a = (ti.cross(dmk)).normalized();
            const Eigen::Vector3d b = (dmk.cross(tj)).normalized();
            Real v = a.dot(b);
            v = std::min(1.0, std::max(v, -1.0)); // clamp e.g. 1 + 1e-12 (numerical noise) to 1
            int pm1 = sign(ti_cross_tj.dot(dmk));
            return pm1*acos(v);
        };

        Real W = 0;
        for (size_t i = 0; i < n-2; i++) {
            for (size_t j = i+2; j < n; j++) {  // we start at i+2 since W_{i, i+1} = 0
                if (i == 0 && j == n-1)  // last edge is adjacent to the first one
                    continue;

                const Eigen::Vector3d &ti = tangents[i];
                const Eigen::Vector3d &tj = tangents[j];
                if (std::abs(ti.dot(tj)) < 1 - tol)  // we skip this term if edges are parallel (the writhe integrand is zero)
                    W += 1/(4*M_PI) * (
                          mu(ti, nodes[i]   - nodes[j]  , tj)
                        - mu(ti, nodes[i+1] - nodes[j]  , tj)
                        - mu(ti, nodes[i]   - nodes[j+1], tj)
                        + mu(ti, nodes[i+1] - nodes[j+1], tj)
                    );
            }
        }
        W *= 2;
        return W;
    }

    // Get the binormals from the reference frame.
    // In case two (or more) consecutive edges are collinear, 
    // the binormal can be set equal to the closest previous valid binormal
    // (i.e. it is parallely transported along a straight segment).
    std::vector<Eigen::Vector3d> binormals(bool normalize = true, bool transport_on_straight = false) const {
        Real tol = 1e-10;
        const std::vector<Eigen::Vector3d> &kb = rod.deformedConfiguration().kb;
        std::vector<Eigen::Vector3d> binormals = std::vector<Eigen::Vector3d>(kb.begin()+1, kb.end()-1);  // kb[1:-1], binormals[i] is actually the binormal at node i+1
        std::vector<Real> b_norms(numVertices());
        for (size_t i = 0; i < numVertices(); i++)
                b_norms[i] = binormals[i].norm();

        // Normalize if norm > 0; otherwise set to zero
        if (normalize) {
            for (size_t i = 0; i < numVertices(); i++) {
                if (b_norms[i] > tol)
                    binormals[i] /= b_norms[i];
                else
                    binormals[i] = Eigen::Vector3d::Zero();
            }
        }

        if (transport_on_straight) {
            // Along straight segments, use the binormal defined at the previous node.
            // For segments made of n edges, the binormal replacement needs to be done iteratively (n times).
            for (size_t i = 0; i < numVertices(); i++) {
                if (b_norms[i] > tol)
                    continue;
                int j = (i == 0) ? numVertices()-1 : i-1;
                while(b_norms[j] < tol)
                    j = (j == 0) ? numVertices()-1 : j-1;
                binormals[i] = binormals[j];
            }
        }
        std::vector<Eigen::Vector3d> binormals_zero_based(numVertices());  // do an np.roll(binormals, 1)
        for (size_t i = 0; i < numVertices()-1; i++)
            binormals_zero_based[i+1] = binormals[i];
        binormals_zero_based[0] = binormals[numVertices()-1];
        return binormals_zero_based;
    }

    std::vector<Pt3> deformedPoints() const {
        std::vector<Pt3> pts = rod.deformedPoints();
        pts.pop_back(); pts.pop_back(); // drop the last two (duplicated) nodes
        return pts;
    }
    std::vector<Pt3> edgesMidpoints() const {
        std::vector<Pt3> pts = rod.deformedPoints();
        std::vector<Pt3> emp(numEdges());
        for(size_t ei = 0; ei < numEdges(); ei++)
            emp[ei] = 0.5 * (pts[ei] + pts[ei+1]);
        return emp;
    }
    std::vector<Real> thetas() const { 
        std::vector<Real> thetas = rod.thetas();
        thetas.pop_back(); // drop last edge, superimposed to the first
        return thetas;
    }
    std::vector<Real> deformedLengths() const { 
        std::vector<Real> lengths = rod.deformedConfiguration().len;
        lengths.pop_back(); // drop last edge, superimposed to the first
        return lengths;
    }
    std::vector<Real> restLengths() const { 
        std::vector<Real> lengths = rod.restLengths();
        lengths.pop_back(); // drop last edge, superimposed to the first
        return lengths;
    }
    Eigen::VectorXd maxBendingStresses() const { 
        Eigen::VectorXd mbs = rod.maxBendingStresses();
        mbs[0] = mbs[numVertices()]; // substitute value on node x_0 (always zero) with value on node x_{nv}
        mbs.conservativeResize(numVertices()); // drop the last two (duplicated) nodes
        return mbs;
    }

    // Different kinds of curvature discretizations 
    // (cfr. Carroll et al. 2013, "A Survey of the Differential Geometry of Discrete Curves")
    enum  class CurvatureDiscretizationType { Tangent, Sine, Angle };

    // Compute discrete curvature at rod's vertices as in [Bergou 2008] (\kappa_i is the *integrated* curvature):
    // \kappa_i / (\bar{l}/2) = 2 F(\Phi_i/2) / (\bar{l}_i/2)
    // where F(.) can be tan(.), sin(.), or the identity, and \Phi_i is the angle between tangents {i, i+1}. 
    // The choice F(.) = sin(.) corresponds to computing the circumscribed circle to the three nodes {i-1, i, i+1};
    // the choice F(.) = tan(.) corresponds to computing the inscribed circle passing through the mid-points of the two edges and having its center on the bisecting line of the angle formed by edges i and i+1;
    // the choice F(.) = Id(.)  returns the turning angle between the tangents on edges {i-1, i}.
    // (cfr. Carroll et al. 2013, "A Survey of the Differential Geometry of Discrete Curves").
    // TODO: use deformed lengths in place of rest ones?
    Eigen::VectorXd curvature(const CurvatureDiscretizationType &discretization = CurvatureDiscretizationType::Sine, bool pointwise = true) const { 
        Eigen::VectorXd curv(numVertices());
        if (discretization == CurvatureDiscretizationType::Tangent) { // 2 * tan(Phi/2)
            std::vector<Eigen::Vector3d> kb = rod.deformedConfiguration().kb; // per vertex kb, including first and last nodes (kb = 0)
            for(size_t i = 1; i < numVertices(); i++)
                curv[i] = kb[i].norm();
            curv[0] = kb[numVertices()].norm();
        }
        else if (discretization == CurvatureDiscretizationType::Sine) { // 2 * sin(Phi/2)
            for(size_t i = 1; i < numVertices(); i++) {
                const Eigen::Vector3d &eim1 = rod.deformedPoint(i)   - rod.deformedPoint(i-1);
                const Eigen::Vector3d &ei   = rod.deformedPoint(i+1) - rod.deformedPoint(i);
                Real norm_prod = eim1.norm() * ei.norm();
                Real dot_prod = eim1.dot(ei);
                assert(norm_prod - dot_prod > -1e-12);  // numerical errors can turn a 0 into a small negative value: this should be the only case in which we can have negative values
                Real numerator = std::abs(norm_prod - dot_prod);
                curv[i] = 2.0 * sqrt(numerator / (2.0 * norm_prod));
            }
            const Eigen::Vector3d &e0   = rod.deformedPoint(1) - rod.deformedPoint(0);
            const Eigen::Vector3d &enm1 = rod.deformedPoint(0) - rod.deformedPoint(numVertices()-1);
            Real norm_prod = e0.norm() * enm1.norm();
            Real dot_prod = e0.dot(enm1);
            assert(norm_prod - dot_prod > -1e-12);
            Real numerator = std::abs(norm_prod - dot_prod);
            curv[0] = 2.0 * sqrt(numerator / (2.0 * norm_prod));
        }
        else if (discretization == CurvatureDiscretizationType::Angle) { // Phi
            for(size_t i = 1; i < numVertices(); i++) {
                const Eigen::Vector3d &eim1 = rod.deformedPoint(i)   - rod.deformedPoint(i-1);
                const Eigen::Vector3d &ei   = rod.deformedPoint(i+1) - rod.deformedPoint(i);
                Real norm_prod = eim1.norm() * ei.norm();
                Real argument = eim1.dot(ei) / norm_prod;
                assert((argument > -1 - 1e-12) && (argument < 1 + 1e-12));  // numerical errors can push values close to +\-1 out of [-1.0, 1.0] bounds
                argument = std::max(-1.0, std::min(argument, 1.0));  // std::clamp(dot_prod, -1.0, 1.0);
                curv[i] = acos(argument);
            }
            const Eigen::Vector3d &e0   = rod.deformedPoint(1) - rod.deformedPoint(0);
            const Eigen::Vector3d &enm1 = rod.deformedPoint(0) - rod.deformedPoint(numVertices()-1);
            Real norm_prod = e0.norm() * enm1.norm();
            Real argument = e0.dot(enm1) / norm_prod;
            assert((argument > -1 - 1e-12) && (argument < 1 + 1e-12));
            argument = std::max(-1.0, std::min(argument, 1.0));
            curv[0] = acos(argument);
        }
        else
            throw std::runtime_error("Unknown discretization.");

        // To compute the pointwise curvature we normalize the integrated quantity 
        // using the Voronoi area of the edges (at rest: TODO use current config instead?)
        if (pointwise) {
            std::vector<Real> rl = rod.restLengths(); // first edge = last edge
            for(size_t i = 1; i < numVertices(); i++)
                curv[i] /= ((rl[i-1] + rl[i])/2.0);
            curv[0] /= ((rl[numVertices()-1] + rl[numVertices()])/2.0);
        }

        return curv;
    }
    // TODO
    Eigen::VectorXd torsion(const CurvatureDiscretizationType &discretization = CurvatureDiscretizationType::Sine, bool pointwise = true) const { 
        std::vector<Eigen::Vector3d> b = binormals(/*normalize*/true, /*transport_on_straight*/true);
        Eigen::VectorXd tor(numEdges());
        if (discretization == CurvatureDiscretizationType::Tangent) { // 2 * tan(Psi/2)
            for(size_t i = 0; i < numEdges(); i++) {
                const size_t ip1 = (i == numEdges()-1) ? 0 : i+1;
                tor[i] = (b[i].cross(b[ip1]) * (2.0 / (1 + b[i].dot(b[ip1])))).norm();
            }
        }
        else if (discretization == CurvatureDiscretizationType::Sine) { // 2 * sin(Psi/2)
            for(size_t i = 0; i < numEdges(); i++) {
                const size_t ip1 = (i == numEdges()-1) ? 0 : i+1;
                tor[i] = 2.0 * sqrt((1 - b[i].dot(b[ip1])) / 2.0);
            }
        }
        else if (discretization == CurvatureDiscretizationType::Angle) { // Psi \in [-pi, pi]
            for(size_t i = 0; i < numEdges(); i++) {
                const auto ti = (rod.deformedConfiguration().point(i+1) - rod.deformedConfiguration().point(i)).normalized();
                const size_t ip1 = (i == numEdges()-1) ? 0 : i+1;
                tor[i] = angle(ti, b[i], b[ip1]);
            }
        }
        else
            throw std::runtime_error("Unknown discretization.");

        // To compute the pointwise torsion we normalize the integrated quantity 
        // using the Voronoi area of the edge (at rest: TODO use current config instead?)
        if (pointwise) {
            std::vector<Real> rl = rod.restLengths(); // first edge = last edge
            for(size_t i = 0; i < numEdges(); i++)
                tor[i] /= rl[i];
        }

        return tor;
    }
    Real crossSectionHeight(size_t ei) const { assert(ei < numEdges()); return rod.crossSectionHeight(ei); }
    Real restLength() const { return rod.restLength() - rod.restLengths()[0]; }
    Real restLengthForEdge(size_t ei) const { assert(ei < numEdges()); return rod.restLengthForEdge(ei); }

    bool elementsAreNeighbors(int i, int j, int d = 1) const {
        if (j < i)
            std::swap(i, j);   // sort
        if (std::abs(i - j) <= d || // i == j included
            std::abs(i - ((j + d) % int(numElems()) - d)) <= d)  // account for periodicity
            return true;
        else
            return false;
    }

    // Additional methods required by compute_equilibrium
    void updateSourceFrame() { return rod.updateSourceFrame(); }
    void updateRotationParametrizations() { return rod.updateRotationParametrizations(); }
    Real_ characteristicLength() const { return rod.characteristicLength(); }
    Real_ initialMinRestLength() const { return rod.initialMinRestLength(); }
    std::vector<size_t> lengthVars(bool variableRestLen) const { return rod.lengthVars(variableRestLen); }
    Real_ approxLinfVelocity(const VecX &paramVelocity) const {
        return rod.approxLinfVelocity(applyJacobian(paramVelocity));
    }
    void writeDebugData(const std::string &path) const { rod.writeDebugData(path); }

    // ------------------------------------------------------------------------------------
    // Visualization
    // ------------------------------------------------------------------------------------

    // Visualise the mesh of the underlying ElasticRod (two additional nodes, one additional edge)
    void visualizationGeometry(std::vector<MeshIO::IOVertex > &vertices,
                               std::vector<MeshIO::IOElement> &quads,
                               const bool averagedMaterialFrames = false) const {
        this->rod.visualizationGeometry(vertices, quads, averagedMaterialFrames);
    }

    void saveVisualizationGeometry(const std::string &path, const bool averagedMaterialFrames = false) const { rod.saveVisualizationGeometry(path, averagedMaterialFrames); }

    // TODO KNOTS: can we easily have visualization fields, too?

    // template<typename Derived>
    // Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Derived::ColsAtCompileTime>
    // visualizationField(const std::vector<Derived> &perRodFields) const {
    //     if (perRodFields.size() != numRods()) throw std::runtime_error("Invalid per-rod-field size");
    //     using FieldStorage = Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Derived::ColsAtCompileTime>;
    //     std::vector<FieldStorage> perRodVisualizationFields;
    //     perRodVisualizationFields.reserve(numRods());
    //     int fullSize = 0;
    //     const int cols = perRodFields.at(0).cols();
    //     for (size_t ri = 0; ri < numRods(); ++ri) {
    //         if (cols != perRodFields[ri].cols()) throw std::runtime_error("Mixed field types forbidden.");
    //         perRodVisualizationFields.push_back(m_rods[ri]->rod.visualizationField(perRodFields[ri]));
    //         fullSize += perRodVisualizationFields.back().rows();
    //     }
    //     FieldStorage result(fullSize, cols);
    //     int offset = 0;
    //     for (const auto &vf : perRodVisualizationFields) {
    //         result.block(offset, 0, vf.rows(), cols) = vf;
    //         offset += vf.rows();
    //     }
    //     assert(offset == fullSize);
    //     return result;
    // }

    // // Provide a single vector containing the fields for all the rods stored in sequence
    // // Note: the field should already contain duplicates to be visualized on duplicated edges or nodes of each PeriodicRod
    // // TODO: this function is not called for vector fields, meaning there is no check on matching values for overlapping edges/nodes. Fix it.
    // template<typename Derived>
    // Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Derived::ColsAtCompileTime>
    // visualizationField(const Derived &field) const {
    //     size_t gi = 0; // global index
    //     std::vector<Derived> perRodFields(numRods());
    //     if ((size_t)field.size() == numEdges(/*countGhost*/true)) {
    //         for (size_t ri = 0; ri < numRods(); ri++) {
    //             perRodFields[ri] = field.segment(gi, m_rods[ri]->numEdges(/*countGhost*/true));
    //             gi += m_rods[ri]->numEdges(/*countGhost*/true);
    //         }
    //         for (const auto &f : perRodFields)
    //             if (f[0] != f[f.size()-1])
    //                 throw std::runtime_error("Values on first and last edges must match.");
    //         return visualizationField(perRodFields);
    //     }
    //     if ((size_t)field.size() == numVertices(/*countGhost*/true)) {
    //         for (size_t ri = 0; ri < numRods(); ri++) {
    //             perRodFields[ri] = field.segment(gi, m_rods[ri]->numVertices(/*countGhost*/true));
    //             gi += m_rods[ri]->numVertices(/*countGhost*/true);
    //         }
    //         for (const auto &f : perRodFields) 
    //             if (f[0] != f[f.size()-2] || f[1] != f[f.size()-1])
    //                 throw std::runtime_error("Values on first and second nodes must match second-last and last ones.");
    //         return visualizationField(perRodFields);
    //     }
    //     else
    //         throw std::runtime_error("Invalid field size (note that the field should already contain duplicates to be visualized on ghost edges or nodes).");
    // }

    // using RMPins = typename Rod::RMPins;
    // typename RMPins::PinInfo prepareRigidMotionPins() { return rod.prepareRigidMotionPins(); }  // TODO KNOTS: delete? Should be part of the new MeshFEM_dev features

    Rod rod;
private:
    Real_ m_twist = 0.0;   // TODO KNOTS: rename!

    CSCMat &m_getCachedUnreducedHessianSparsityPattern() const {
        if (m_cachedUnreducedHessianSparsityPattern.m == 0)
            m_cachedUnreducedHessianSparsityPattern = rod.hessianSparsityPattern(0.0);
        return m_cachedUnreducedHessianSparsityPattern;
    }

    // ------------------------------------------------------------------------------------
    // TODO KNOTS: these utility functions are from MeshFEM_dev, can I include them here?

    // Get the sin of the signed angle from v1 to v2 around axis "a". Uses right hand rule
    // as the sign convention: clockwise is positive when looking along vector.
    // Assumes all vectors are normalized.
    Real_ sinAngle(const Vec3_T<Real_> &a, const Vec3_T<Real_> &v1, const Vec3_T<Real_> &v2) const {
        return v1.cross(v2).dot(a);
    }

    // Get the signed angle from v1 to v2 around axis "a". Uses right hand rule
    // as the sign convention: clockwise is positive when looking along vector.
    // Assumes all vectors are normalized **and perpendicular to a**
    // Return answer in the range [-pi, pi]
    Real_ angle(const Vec3_T<Real_> &a, const Vec3_T<Real_> &v1, const Vec3_T<Real_> &v2) const {
        Real_ s = std::max(Real_(-1.0), std::min(Real_(1.0), sinAngle(a, v1, v2)));
        Real_ c = std::max(Real_(-1.0), std::min(Real_(1.0), v1.dot(v2)));
        return atan2(s, c);
    }

    // Compute the curvature binormal for a vertex between two edges with tangents
    // e0 and e1, respectively
    // (edge tangent vectors not necessarily normalized)
    Vec3_T<Real_> curvatureBinormal(const Vec3_T<Real_> &e0, const Vec3_T<Real_> &e1) {
        return e0.cross(e1) * (2.0 / (e0.norm() * e1.norm() + e0.dot(e1)));
    }
    // ------------------------------------------------------------------------------------

    mutable CSCMat m_cachedHessianSparsityPattern,
                   m_cachedUnreducedHessianSparsityPattern;
};

#endif /* end of include guard: PERIODICROD_HH */
