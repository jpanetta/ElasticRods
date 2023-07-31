////////////////////////////////////////////////////////////////////////////////
// PeriodicRod.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  This class represents a closed loop formed by gluing together the ends of
//  an elastic rod. To ensure stretching, bending, and twisting elastic
//  energies are calculated properly, the first and last ends of the rod are
//  constrained to overlap, and the stretching stiffness of this rod is halved
//  to avoid double-counting. An additional opening angle variable is introduced 
//  to allow a twist discontinuity at the joint. This variable specifies the 
//  offset in frame angle theta between the two overlapping edges (measured from 
//  the last edge to the first). By fixing this angle to a constant, nonzero 
//  twist can be maintained in the rod.
//
//  The ends are glued together with the following simple equality constraints on
//  the deformation variables:
//      x_{nv - 2}     = x_0
//      x_{nv - 1}     = x_1
//      theta_{ne - 1} = theta_0 + totalOpeningAngle
//
//  We implement these constraints efficiently with a change of deformation
//  variables from "unreduced" variables
//      [x_0, ..., x_{nv - 1}, theta_0, ..., theta_{ne - 1}]
//  to "reduced" variables
//      [x_0, ..., x_{nv - 3}, theta_0, ..., theta_{ne - 2}, totalOpeningAngle]
//
//  In matrix form, this linear change of variables looks like:
//                  [I_6 0             0 0           0][ x_0 \\ x_1              ]
//                  [0   I_{3(nv - 2)} 0 0           0][ x_2 .. x_{nv - 3}       ]
//  unreducedVars = [I_6 0             0 0           0][ theta_0                 ]
//                  [0   0             1 0           0][ theta_1..theta_{ne - 2} ]
//                  [0   0             0 I_{ne - 1}  0][ totalOpeningAngle       ]
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
#include <MeshFEM/Geometry.hh>
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

    PeriodicRod_T(const std::vector<Pt3> &points, bool zeroRestCurvature = false)
        : rod(points)
    {
        const size_t nv = rod.numVertices();
        if (((points[0] - points[nv - 2]).norm() > 1e-12) ||
            ((points[1] - points[nv - 1]).norm() > 1e-12)) throw std::runtime_error("First and last edge must overlap!");

        // Overwrite final edge's reference frame with the reference frame of the first edge.
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

    PeriodicRod_T(const ElasticRod &rod, Real totalOpeningAngle)
        : rod(rod), m_totalOpeningAngle(totalOpeningAngle) 
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

    size_t numDoF()   const { return rod.numDoF() - 6; }   // remove the last two endpoint position variables.
    size_t numElems() const { return rod.numVertices() - 2; }   // #nodes == #edges in periodic rods
    size_t numVertices(bool countGhost = false) const { return countGhost ? numElems() + 2 : numElems(); }
    size_t numEdges   (bool countGhost = false) const { return countGhost ? numElems() + 1 : numElems(); }
    size_t thetaOffset() const { return 3 * (rod.numVertices() - 2); }

    VecX getDoFs() const {
        VecX result(numDoF());
        VecX unreducedDoFs = rod.getDoFs();
        result.head   (3 * (rod.numVertices() - 2))                     = unreducedDoFs.head   (3 * (rod.numVertices() - 2));
        result.segment(3 * (rod.numVertices() - 2), rod.numEdges() - 1) = unreducedDoFs.segment(3 * rod.numVertices(), rod.numEdges() - 1);
        result[result.size() - 1] = m_totalOpeningAngle;
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
        m_totalOpeningAngle = dofs[dofs.size() - 1];
        rod.setDoFs(applyJacobian(dofs));
    }

    Real_ totalOpeningAngle() const { return m_totalOpeningAngle; }
    void setTotalOpeningAngle(Real_ angle) { m_totalOpeningAngle = angle; setDoFs(getDoFs()); }

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
        Real_ gradLastTotOpenAngle = unreducedGradient[unreducedGradient.size() - 1];

        // Column block 6 of J^T
        result[3 * (rod.numVertices() - 2)] += gradLastTotOpenAngle;
        result[result.size() - 1]            = gradLastTotOpenAngle;

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
        const size_t totOpenAngleVar = numDoF() - 1;

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
                // unreduced theta variable on m_totalOpeningAngle.
                if (t.i == lastTheta) { emitNZ(totOpenAngleVar, rj, t.v); emitNZ(rj, totOpenAngleVar, t.v); }
                if (t.j == lastTheta) { emitNZ(totOpenAngleVar, ri, t.v); emitNZ(ri, totOpenAngleVar, t.v); }
            }
            else if (t.i == lastTheta) {
                // Generate the extra diagonal entry produced by the dependency of the
                // unreduced theta variable on m_totalOpeningAngle.
                emitNZ(ri, totOpenAngleVar, t.v);
                emitNZ(totOpenAngleVar, totOpenAngleVar, t.v);
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

    // Note: the "lumped mass matrix" is not perfectly diagonal due to the "totalOpeningAngle" variable's
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

    // Set a new deformed configuration without affecting the current opening angle
    void setDeformedConfiguration(const std::vector<Pt3> &points, const std::vector<Real> &thetas) {
        if (points.size() != numVertices()) throw std::runtime_error("setDeformedConfiguration: number of deformed points (" + std::to_string(points.size()) + ") should match number of vertices (" + std::to_string(numVertices()) + ").");
        if (thetas.size() != numEdges())    throw std::runtime_error("setDeformedConfiguration: number of deformed thetas (" + std::to_string(thetas.size()) + ") should match number of edges (" + std::to_string(numEdges()) + ").");
        std::vector<Pt3>  extendedPoints(points.begin(), points.end());
        std::vector<Real> extendedThetas(thetas.begin(), thetas.end());
        extendedPoints.push_back(points[0]);
        extendedPoints.push_back(points[1]);
        extendedThetas.push_back(m_totalOpeningAngle + thetas[0]);
        rod.setDeformedConfiguration(extendedPoints, extendedThetas);
    }

    // Set a new deformed configuration changing the opening angle
    void setDeformedConfiguration(const std::vector<Pt3> &points, const std::vector<Real> &thetas, Real totalOpeningAngle) {
        m_totalOpeningAngle = totalOpeningAngle;
        setDeformedConfiguration(points, thetas);
    }

    std::vector<Pt3>    deformedPoints() const { return m_ERtoPRonVertices(rod.deformedPoints()); }
    std::vector<Real>           thetas() const { return m_ERtoPRonEdges(rod.thetas()); }
    std::vector<Real>  deformedLengths() const { return m_ERtoPRonEdges(rod.deformedConfiguration().len); }
    std::vector<Real>      restLengths() const { return m_ERtoPRonEdges(rod.restLengths()); }
    Eigen::VectorXd maxBendingStresses() const { 
        Eigen::VectorXd mbs = rod.maxBendingStresses();
        mbs[0] = mbs[numVertices()]; // replace value on node x_0 (always zero) with value on node x_{nv}
        mbs.conservativeResize(numVertices()); // drop the last two duplicated nodes
        return mbs;
    }

    // Compute the binormals of the centerline curve.
    // In case two (or more) consecutive edges are collinear — and the Frenet binormal is thus not defined —
    // the binormal can be set equal to the closest previous valid binormal
    // (i.e. binormals can be parallely transported along straight segments).
    std::vector<Eigen::Vector3d> binormals(bool normalize = true, bool transport_on_straight = false) const {
        Real tol = 1e-10;
        const std::vector<Eigen::Vector3d> &kb = rod.deformedConfiguration().kb;
        std::vector<Eigen::Vector3d> binormals = std::vector<Eigen::Vector3d>(kb.begin()+1, kb.end()-1);  // kb[1:-1], binormals[i] is currently the binormal at node i+1
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

    enum  class CurvatureDiscretizationType { Tangent, Sine, Angle };  // see "A Survey of the Differential Geometry of Discrete Curves" [Carroll et al. 2013]

    // Compute the discrete curvature at rod's vertices \kappa_i / (\bar{l}/2) = 2 F(\Phi_i/2) / (\bar{l}_i/2),
    // where \kappa_i is the *integrated* curvature as in [Bergou 2008], F(.) can be tan(.), sin(.), or the identity, and \Phi_i \in [0, pi] is the turning angle between the tangents {i-1, i}. 
    // The choice F(.) = sin(.) corresponds to computing the circumscribed circle to the three nodes {i-1, i, i+1};
    // the choice F(.) = tan(.) corresponds to computing the inscribed circle passing through the mid-points of the two edges and having its center on the bisecting line of the angle formed by edges i-1 and i;
    // the choice F(.) = Id(.)  returns the turning angle between the tangents on edges {i-1, i}.
    // See "A Survey of the Differential Geometry of Discrete Curves" [Carroll et al. 2013]).
    // The choice F(.) = tan(.) is the one used in [Bergou 2008].
    Eigen::VectorXd curvature(const CurvatureDiscretizationType &discretization = CurvatureDiscretizationType::Angle, bool pointwise = true) const { 
        Eigen::VectorXd curv(numVertices());
        if (discretization == CurvatureDiscretizationType::Tangent) { // 2 * tan(Phi/2) \in [0, +inf]
            std::vector<Eigen::Vector3d> kb = rod.deformedConfiguration().kb; // per vertex kb, including first and last nodes (kb = 0)
            for(size_t i = 1; i < numVertices(); i++)
                curv[i] = kb[i].norm();
            curv[0] = kb[numVertices()].norm();
        }
        else if (discretization == CurvatureDiscretizationType::Sine) { // 2 * sin(Phi/2) \in [0, 2]
            for(size_t i = 1; i < numVertices(); i++) {
                const Eigen::Vector3d &eim1 = rod.deformedPoint(i)   - rod.deformedPoint(i-1);
                const Eigen::Vector3d &ei   = rod.deformedPoint(i+1) - rod.deformedPoint(i);
                Real norm_prod = eim1.norm() * ei.norm();
                Real dot_prod = eim1.dot(ei);
                assert(norm_prod - dot_prod > -1e-12);  // floating point arithmetic can turn a 0 into a small negative value: this is the only case in which we can have negative values
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
        else if (discretization == CurvatureDiscretizationType::Angle) { // Phi \in [0, pi]
            for(size_t i = 1; i < numVertices(); i++) {
                const Eigen::Vector3d &eim1 = rod.deformedPoint(i)   - rod.deformedPoint(i-1);
                const Eigen::Vector3d &ei   = rod.deformedPoint(i+1) - rod.deformedPoint(i);
                Real norm_prod = eim1.norm() * ei.norm();
                Real argument = eim1.dot(ei) / norm_prod;
                assert((argument > -1 - 1e-12) && (argument < 1 + 1e-12));  // floating point arithmetic can push values close to +\-1 out of [-1.0, 1.0] bounds
                argument = std::max(-1.0, std::min(argument, 1.0));
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
        // using the Voronoi length of the edges in the current deformed configuration
        if (pointwise) {
            std::vector<Real> len = rod.deformedConfiguration().len;
            for(size_t i = 1; i < numVertices(); i++)
                curv[i] /= ((len[i-1] + len[i])/2.0);
            curv[0] /= ((len[numVertices()-1] + len[numVertices()])/2.0);
        }

        return curv;
    }
    // Compute the discrete torsion at rod's edges \tau_i = F(\Psi_{i+1} - \Psi_i),
    // where F(.) can be tan(.), sin(.), or the identity, and \Psi_i \in [-pi, pi] is the angle between the binormals {i, i+1}. 
    // This is similar in spirit to the discrete twist m_i = \theta_i - \theta_{i-1} defined in [Bergou 2008],
    // but the discrete torsion is an *intrinsic* quantity that only depends only on the curve, not on the framing.
    // See "A Survey of the Differential Geometry of Discrete Curves" [Carroll et al. 2013]).
    Eigen::VectorXd torsion(const CurvatureDiscretizationType &discretization = CurvatureDiscretizationType::Angle, bool pointwise = true) const { 
        std::vector<Eigen::Vector3d> b = binormals(/*normalize*/true, /*transport_on_straight*/true);
        Eigen::VectorXd tor(numEdges());
        auto sign = [](const Real &x){ return (Real(0) < x) - (x < Real(0)); };
        
        if (discretization == CurvatureDiscretizationType::Tangent) { // 2 * tan(Psi/2) \in [-inf, inf]
            for(size_t i = 0; i < numEdges(); i++) {
                const size_t ip1 = (i == numEdges()-1) ? 0 : i+1;
                const Eigen::Vector3d cross = b[i].cross(b[ip1]);
                const Eigen::Vector3d ti = (rod.deformedPoint(i+1) - rod.deformedPoint(i)).normalized();
                tor[i] = sign(cross.dot(ti)) * (cross * (2.0 / (1 + b[i].dot(b[ip1])))).norm();
            }
        }
        else if (discretization == CurvatureDiscretizationType::Sine) { // 2 * sin(Psi/2) \in [-2, 2]
            for(size_t i = 0; i < numEdges(); i++) {
                const size_t ip1 = (i == numEdges()-1) ? 0 : i+1;
                const Eigen::Vector3d cross = b[i].cross(b[ip1]);
                const Eigen::Vector3d ti = (rod.deformedPoint(i+1) - rod.deformedPoint(i)).normalized();
                tor[i] = sign(cross.dot(ti)) * 2.0 * sqrt((1 - b[i].dot(b[ip1])) / 2.0);
            }
        }
        else if (discretization == CurvatureDiscretizationType::Angle) { // Psi \in [-pi, pi]
            for(size_t i = 0; i < numEdges(); i++) {
                const auto ti = (rod.deformedPoint(i+1) - rod.deformedPoint(i)).normalized();
                const size_t ip1 = (i == numEdges()-1) ? 0 : i+1;
                tor[i] = angle(ti, b[i], b[ip1]);
            }
        }
        else
            throw std::runtime_error("Unknown discretization.");

        // To compute the pointwise curvature we normalize the integrated quantity 
        // using the Voronoi length of the edges in the current deformed configuration
        if (pointwise) {
            std::vector<Real> len = rod.deformedConfiguration().len;
            for(size_t i = 0; i < numEdges(); i++)
                tor[i] /= len[i];
        }

        return tor;
    }
    Real crossSectionHeight(size_t ei) const { assert(ei < numEdges()); return rod.material(ei).crossSectionHeight; }
    Real crossSectionArea  (size_t ei) const { assert(ei < numEdges()); return rod.material(ei).area; }
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

    // -------------------------------------------------------
    // Geometric quantities related by Călugăreanu's Theorem:     
    //          Link + Phi / 2pi = Twist + Writhe
    // -------------------------------------------------------

    // Compute the number of turns the material frame does compared to a zero-twist (natural) frame.
    // The value corresponds to int_0^L u_3(s) ds / (2pi), where u_3 is the component of the Darboux vector of the material frame in the tangential direction.
    Real twist() const { return totalTwistAngle() / (2*M_PI); }

    // Compute int_0^L u_3(s) ds.
    // The integral coincides with the cumulated twist of the reference frame plus the total opening angle.
    Real totalTwistAngle() const { return totalReferenceTwistAngle() + m_totalOpeningAngle; }

    // Compute the cumulated twist angle of the reference frame
    Real totalReferenceTwistAngle() const {
        const std::vector<Real> &refTwist = rod.deformedConfiguration().referenceTwist; // twist in reference frame, always of the form [0, tw_1, ..., tw_nv, 0]
        return std::accumulate(refTwist.begin() + 1, refTwist.end() - 1, 0.0);
    }

    // Compute the rotation R_{\Phi}, with \Phi \in [0, 2*pi), that makes the material frame of the first edge coincide with the one of the last edge.
    // It corresponds to the total opening angle modulo 2*pi (plus, if negative, a shift to make it positive).
    Real openingAngle() const {
        Real dummy_integer_part;
        Real v = 2*M_PI * std::modf(m_totalOpeningAngle / (2*M_PI), &dummy_integer_part);  // in (-2*pi, 2*pi), if x < 0 then std::modf(x) < 0
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
                if (i == 0 && j == n-1)  // last and first edges are adjacent
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

    Real link() const { return std::floor(twist() + writhe()); }

    Rod rod;
private:
    Real_ m_totalOpeningAngle = 0.0;

    CSCMat &m_getCachedUnreducedHessianSparsityPattern() const {
        if (m_cachedUnreducedHessianSparsityPattern.m == 0)
            m_cachedUnreducedHessianSparsityPattern = rod.hessianSparsityPattern(0.0);
        return m_cachedUnreducedHessianSparsityPattern;
    }

    template<typename T>
    std::vector<T> m_ERtoPRonEdges(const std::vector<T> &values) const {
        assert(values.size() == rod.numEdges());
        std::vector<T> valuesPR(values.begin(), values.end());
        valuesPR.pop_back();  // drop last value
        return valuesPR;
    }

    template<typename T>
    std::vector<T> m_ERtoPRonVertices(const std::vector<T> &values) const {
        assert(values.size() == rod.numVertices());
        std::vector<T> valuesPR(values.begin(), values.end());
        valuesPR.pop_back(); valuesPR.pop_back();  // drop last two values
        return valuesPR;
    }

    mutable CSCMat m_cachedHessianSparsityPattern,
                   m_cachedUnreducedHessianSparsityPattern;
};

#endif /* end of include guard: PERIODICROD_HH */
