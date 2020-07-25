////////////////////////////////////////////////////////////////////////////////
// ElasticRod.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Represents a discrete elastic rod as proposed in [Bergou 2010] (Discrete
//  Viscous Threads). Uses the time-parallel frame.
//
//  Note: no boundary conditions have been applied, so the Hessian will be
//  rank deficient.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  03/05/2018 11:01:34
////////////////////////////////////////////////////////////////////////////////
#ifndef ELASTICROD_HH
#define ELASTICROD_HH

#include <MeshFEM/Types.hh>
#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/Fields.hh>
#include <stdexcept>
#include <numeric>

// Forward declare IO types.
namespace MeshIO {
    class IOVertex;
    class IOElement;
}

#include <Eigen/StdVector> // Work around alignment issues with std::vector

#include "RodMaterial.hh"
#include "TriDiagonalSystem.hh"
#include "TemplatedTypes.hh"

// Flags to indicate whether certain blocks of the Hessian computation should
// be skipped. "*_in" indicates whether the vectors to which the Hessian
// will be applied have nonzero dof/reslen components.
// "*_out" indicates whether the dof/restlen components of H * v should
// be computed.
// Whether the
struct HessianComputationMask {
    bool     dof_in = true,     dof_out = true;
    bool restlen_in = true, restlen_out = true;
};

////////////////////////////////////////////////////////////////////////////////
// Functionality for omitting contributions from certain gradient stencils,
// whether for reducing computation (e.g., to speed up RodLinkage::applyHessian)
// or for inspecting internal elastic forces acting between parts of the rod.
////////////////////////////////////////////////////////////////////////////////
struct GradientStencilMaskIncludeAll {
    static constexpr bool includeEdgeStencil(size_t /* ne */, size_t /* j */) { return true; }
    static constexpr bool includeVtxStencil (size_t /* nv */, size_t /* i */) { return true; }
};

// Include stencils that influence the two outermost vertices and outermost edge
// variable.
struct GradientStencilMaskTerminalsOnly {
    static constexpr bool includeEdgeStencil(size_t ne, size_t j) { return (j <= 1) || (j >= ne - 2); }
    static constexpr bool includeVtxStencil (size_t nv, size_t i) { return (i <= 2) || (i >= nv - 3); }
};

struct GradientStencilMaskCustom {
    std::vector<bool> edgeStencilMask, vtxStencilMask;
    // Include all by default (unless the stencil has been modified)
    bool includeEdgeStencil(size_t /* ne */, size_t j) const { if (edgeStencilMask.empty()) return true; return edgeStencilMask.at(j); }
    bool includeVtxStencil (size_t /* nv */, size_t i) const { if ( vtxStencilMask.empty()) return true; return  vtxStencilMask.at(i); }
};

// Templated to support automatic differentiation types.
template<typename Real_>
struct ElasticRod_T;

using ElasticRod = ElasticRod_T<Real>;

// An elastic rod is a sequence of points connected with edges.
// Templated to support automatic differentiation types.
template<typename Real_>
struct ElasticRod_T {
    using Vec3 = Vec3_T<Real_>;
    using Pt3  =  Pt3_T<Real_>;
    using Vec2 = Vec2_T<Real_>;
    using VecX = VecX_T<Real_>;
    using Mat2 = Mat2_T<Real_>;
    using CSCMat = CSCMatrix<SuiteSparse_long, Real_>;

    using TMatrix = TripletMatrix<Triplet<Real_>>;
    using StdVectorVector2D = std::vector<Vec2, Eigen::aligned_allocator<Vec2>>; // Work around alignment issues.
    using StdVectorMatrix2D = std::vector<Mat2, Eigen::aligned_allocator<Mat2>>; // Work around alignment issues.
    struct Directors;
    struct DeformedState;

    ElasticRod_T(const std::vector<Pt3> &points) {
        setRestConfiguration(points);
        m_deformedStates.reserve(2); // Typically we will need to track only two deformed states (for line search)
        setMaterial(RodMaterial());
    }

    // Converting constructor from another floating point (e.g., double to autodiff)
    template<typename Real2>
    ElasticRod_T(const ElasticRod_T<Real2> &r);

    enum class EnergyType { Full, Bend, Twist, Stretch };
    // Configure the bending energy expression (Bergou2010 or Bergou2008).
    enum class BendingEnergyType { Bergou2010, Bergou2008 };
    void setBendingEnergyType(BendingEnergyType type) { m_bendingEnergyType = type; }
    BendingEnergyType bendingEnergyType() const { return m_bendingEnergyType; }

    void     setRestConfiguration(const std::vector<Pt3> &points);
    // Get/set the active deformed state.
    const DeformedState &deformedConfiguration() const { return m_deformedStates.back(); }
          DeformedState &deformedConfiguration()       { return m_deformedStates.back(); }
    void setDeformedConfiguration(const std::vector<Pt3> &points, const std::vector<Real_> &thetas);
    void setDeformedConfiguration(const DeformedState &deformedState) { m_deformedStates.back() = deformedState; }

    // Set the current adapted curve frame as the source for parallel transport.
    // Note: this effectively changes the elastic energy function (but it preserves
    // the value and gradient at the current configuration)
    // The Hessian and gradEnergy*(updatedSource = true) calculations will
    // only be accurate for the current configuration after this method has
    // been called.
    // Typically, this method should be called at the start of each
    // optimization/simulation iteration
    void updateSourceFrame() { deformedConfiguration().updateSourceFrame(); }

    // For compatibility with RodLinkage; nop
    void updateRotationParametrizations() { }

    // Since the energy is path-dependent (due to the parallel transport), we
    // need to be able to save and restore the deformed state during a line
    // search; we use a stack for this purpose.
    void pushDeformedConfiguration(const std::vector<Pt3> &points, const std::vector<Real_> &thetas) {
        m_deformedStates.push_back(m_deformedStates.back()); // Clone the active deformed state
        m_deformedStates.back().update(points, thetas);      // Apply the new deformed configuration
    }
    void  popDeformedConfiguration() {
        if (m_deformedStates.size() == 1) throw std::runtime_error("Cannot pop the only remaining state!");
        m_deformedStates.pop_back();
    }

    const std::vector<Pt3> &deformedPoints() const { return m_deformedStates.back().points(); }
    const std::vector<Real_>            &thetas() const { return m_deformedStates.back().thetas(); }
    const Pt3       &deformedPoint(size_t i) const { return m_deformedStates.back().point(i); }
    Real_                         theta(size_t j) const { return m_deformedStates.back().theta(j); }

    // Edge's edge vector in the rest configuration (i.e., the non-unit tangent vector)
    Vec3 restEdgeVector(size_t j) const {
        assert(j < numEdges());
        return m_restPoints[j + 1] - m_restPoints[j];
    }

    const std::vector<Directors> &restDirectors() const { return m_restDirectors; }
    const StdVectorVector2D      &restKappas   () const { return m_restKappa    ; }
    const std::vector<Real_    > &restTwists   () const { return m_restTwist    ; }
    void setRestDirectors(const std::vector<Directors> &val) { m_restDirectors = val; }
    void setRestKappas   (const StdVectorVector2D      &val) { m_restKappa     = val; }
    void setRestTwists   (const std::vector<Real_    > &val) { m_restTwist     = val; }

    Vec3 restTangent(size_t j) const { return restEdgeVector(j).normalized(); }

    // Directly modify the rest lengths while preserving the (integrated) rest curvatures
    // Note: this *doesn't* affect the points stored in "m_restPoints"
    void setRestLengths(const std::vector<Real_> &l) { m_restLen = l; }
    const std::vector<Real_> &restLengths() const { return m_restLen; }
    const std::vector<Real_> &    lengths() { return m_deformedStates.back().len; }
    size_t numRestLengths() const { return numEdges(); }
    Real_  minRestLength() const { return *std::min_element(m_restLen.begin(), m_restLen.end()); }

    // The minimum rest length as of the last "setRestConfiguration". This is
    // useful to get a reasonable lower bound on the rest/deformed length variables.
    // (We can't use the current minRestLength() because the rest length optimization could be
    //  run in a loop, allowing the minimum rest length to shrink by a constant factor with each run.)
    Real_ initialMinRestLength() const { return m_initMinRestLen; }

    Real_  restLengthForEdge(size_t j) const { return m_restLen[j]; }
    Real_ &restLengthForEdge(size_t j)       { return m_restLen[j]; }

    // Total rest length of this rod.
    Real_ restLength() const { return std::accumulate(m_restLen.begin(), m_restLen.end(), Real_(0.0)); }
    // The rest-length defines the characteristic length scale of this rod.
    // (For purposes of determining reasonable descent velocites).
    Real_ characteristicLength() const { return restLength(); }

    const Vec3 &restMaterialFrameD2(size_t j) const {
        assert(j < numEdges());
        // Note: assume the rest configuration has no twist (all rest thetas are zero).
        return m_restDirectors.at(j).d2;
    }
    const Vec3 &deformedMaterialFrameD2(size_t j) const {
        return m_deformedStates.back().materialFrame.at(j).d2;
    }

    const Vec3 &restMaterialFrameD1(size_t j) const {
        assert(j < numEdges());
        // Note: assume the rest configuration has no twist (all rest thetas are zero).
        return m_restDirectors.at(j).d1;
    }
    const Vec3 &deformedMaterialFrameD1(size_t j) const {
        return m_deformedStates.back().materialFrame.at(j).d1;
    }

    // Get/set the degrees of freedom corresponding to the current deformed
    // configuration. The variable ordering is:
    //      flattened centerline positions (x1, y1, z1, x2, y2, ...)
    //      followed by thetas
    size_t   posOffset() const { return 0; }
    size_t thetaOffset() const { return 3 * numVertices(); }
    VecX getDoFs() const;

    void setDoFs(const Eigen::Ref<const VecX> &dofs);
    void setDoFs(const std::vector<Real_> &dofs) { setDoFs(Eigen::Map<const VecX>(dofs.data(), dofs.size())); }

    // Extended degrees of freedom: positions, thetas, and rest lengths
    // (To be used when optimizing the rest length)
    size_t restLenOffset() const { return thetaOffset() + numEdges(); }
    VecX getExtendedDoFs() const {
        VecX result(numExtendedDoF());
        result.segment(0, numDoF()) = getDoFs();
        result.segment(restLenOffset(), m_restLen.size()) = Eigen::Map<const VecX>(m_restLen.data(), m_restLen.size());
        return result;
    }

    void setExtendedDoFs(const Eigen::Ref<const VecX> &dofs) {
        if (size_t(dofs.size()) != numExtendedDoF()) throw std::runtime_error("Extended DoF vector size mismatch");
        setDoFs(dofs.segment(0, numDoF()));
        Eigen::Map<VecX>(m_restLen.data(), m_restLen.size()) = dofs.segment(restLenOffset(), m_restLen.size());
    }

    // Rest length solve interface (for compatibility with RodLinkage)
    size_t numRestlenSolveDof()                                                                const { return numExtendedDoF(); }
    size_t numRestlenSolveRestLengths()                                                        const { return numEdges(); }
    VecX getRestlenSolveDoF()                                                                  const { return getExtendedDoFs(); }
    void setRestlenSolveDoF(const VecX &params)                                                      { return setExtendedDoFs(params); }
    VecX restlenSolveGradient(bool updatedSource = false, EnergyType eType = EnergyType::Full) const { return gradient(updatedSource, eType, true); }
    CSCMat restlenSolveHessianSparsityPattern()                                                const { return hessianSparsityPattern(true); }
    void restlenSolveHessian(CSCMat &H, EnergyType etype = EnergyType::Full)                   const { hessian(H, etype, true); }
    std::vector<size_t> restlenSolveLengthVars()                                               const { return lengthVars(true); }

    // Push a new deformed configuration with the specified DoFs
    void pushDoFs(const std::vector<Real_> &dofs) {
        m_deformedStates.push_back(m_deformedStates.back()); // Clone the active deformed state
        setDoFs(dofs);
    }

    const std::vector<Pt3> &restPoints() const { return m_restPoints; }

    // Determine the material frame vector D2 for edge "j" that corresponds to
    // angle "theta" after the edge has been transformed to have the new edge vector eNew
    // (i.e., after the reference directors have been updated with parallel transport).
    Vec3 materialFrameD2ForTheta(Real_ theta, const Vec3 &eNew, size_t j) const;

    // Determine frame rotation angle "theta" for edge "j" from material frame
    // vector, "d2". Because we usually want thetas for a new deformed
    // configuration, we have the user specify the new edge vector "eNew" which
    // allows us to compute the theta with respect to the new reference
    // directors that will be obtained by parallel transport.
    // (This is the rotation that takes the second reference director to "d2".
    // We remove the integer-multiple-of-2Pi ambiguity in one of two ways:
    //  if spatialCoherence == true, we chose the angle that minimizes twisting energy for one of the incident vertices.
    //  if spatialCoherence == false, we opt for temporal coherence, minimizing the change in angle from the deformedConfiguration.
    //  To remove the integer-multiple-of-2Pi ambiguity, we choose the angle that minimizes twisting energy for one
    //  of the incident vertices).
    Real_ thetaForMaterialFrameD2(Vec3 d2 /* copy modified inside */, const Vec3 &eNew, size_t j, bool spatialCoherence = false) const;

    void setMaterial(const RodMaterial &material) {
        m_edgeMaterial.assign(1, material);
        m_density            .assign(numEdges(),                           1.0);
        m_stretchingStiffness.assign(numEdges(),  material.stretchingStiffness);
        m_twistingStiffness  .assign(numVertices(), material.twistingStiffness);
        m_bendingStiffness   .assign(numVertices(),  material.bendingStiffness);
    }

    void setMaterial(const std::vector<RodMaterial> &edgeMaterials) {
        if (edgeMaterials.size() > 1) {
            throw std::runtime_error("Unimplemented");
            // TODO: set (interpolated) stiffnesses accordingly
        }
        else { setMaterial(edgeMaterials[0]); }
    }

    // Access the material for a particular edge.
    const RodMaterial &material(size_t j = 0) const { if (m_edgeMaterial.size() == 1) return m_edgeMaterial[0]; else return m_edgeMaterial.at(j); }
          RodMaterial &material(size_t j = 0)       { if (m_edgeMaterial.size() == 1) return m_edgeMaterial[0]; else return m_edgeMaterial.at(j); }
    const std::vector<RodMaterial> &edgeMaterials() const { return m_edgeMaterial; }

    // Allow manual modification of stretching stiffness
    Real_ &stretchingStiffness(size_t j) { return m_stretchingStiffness[j]; }

    // Allow manual modification of the density
    Real_ &density(size_t j)       { assert(j < numEdges()); return m_density[j]; }
    Real_  density(size_t j) const { assert(j < numEdges()); return m_density[j]; }
    const std::vector<Real_> &densities() const { return m_density; }
    void setDensities(const std::vector<Real_> &d) { m_density = d; }

    // Access a given vertex's bending and twisting stiffness.
          RodMaterial::BendingStiffness  &bendingStiffness(size_t i)       { assert(i < numVertices()); return  m_bendingStiffness[i]; }
    const RodMaterial::BendingStiffness  &bendingStiffness(size_t i) const { assert(i < numVertices()); return  m_bendingStiffness[i]; }
    Real_ &                              twistingStiffness(size_t i)       { assert(i < numVertices()); return m_twistingStiffness[i]; }
    Real_                                twistingStiffness(size_t i) const { assert(i < numVertices()); return m_twistingStiffness[i]; }
    const std::vector<RodMaterial::BendingStiffness> &   bendingStiffnesses() const { return    m_bendingStiffness; }
    const std::vector<Real_                        > &  twistingStiffnesses() const { return   m_twistingStiffness; }
    const std::vector<Real_                        > &stretchingStiffnesses() const { return m_stretchingStiffness; }
    void    setBendingStiffnesses(const std::vector<RodMaterial::BendingStiffness> &vals) { if (vals.size() != numVertices()) throw std::runtime_error("Invalid bending    stiffnesses size: " + std::to_string(vals.size()) + " vs " + std::to_string(numVertices())); m_bendingStiffness    = vals; }
    void   setTwistingStiffnesses(const std::vector<Real_                        > &vals) { if (vals.size() != numVertices()) throw std::runtime_error("Invalid twisting   stiffnesses size: " + std::to_string(vals.size()) + " vs " + std::to_string(numVertices())); m_twistingStiffness   = vals; }
    void setStretchingStiffnesses(const std::vector<Real_                        > &vals) { if (vals.size() != numEdges   ()) throw std::runtime_error("Invalid stretching stiffnesses size: " + std::to_string(vals.size()) + " vs " + std::to_string(numEdges   ())); m_stretchingStiffness = vals; }

    Real_ energyStretch() const;
    Real_ energyBend()    const;
    Real_ energyTwist()   const;
    Real_ energy()        const;

    Real_ energy(EnergyType type) const {
        switch (type) {
            case EnergyType::   Full: return energy();
            case EnergyType::   Bend: return energyBend();
            case EnergyType::  Twist: return energyTwist();
            case EnergyType::Stretch: return energyStretch();
            default: throw std::runtime_error("Unknown energy type");
        }
    }

    // Per-vertex bending energy for visualization
    VecX energyBendPerVertex() const;

    struct Gradient : public VecX {
        using Base = VecX;
        using Base::Base; // Needed for pybind11
        // Construct zero-initialized gradient
        // If hasVariableRestLen is true, then the gradient also stores partial
        // derivatives with respect to the rest length.
        Gradient(const ElasticRod_T &e, bool hasVariableRestLen = false)
            : Base(Base::Zero(3 * e.numVertices() + e.numEdges() + hasVariableRestLen * e.numEdges())),
              thetaOffset(e.thetaOffset()),
              restLenOffset(e.thetaOffset() + e.numEdges()),
              variableRestLen(hasVariableRestLen)
        { }

        // Accessors for gradient with respect to centerline positions and material frame angles.
        auto   gradPos    (size_t i) const { return this->template segment<3>(3 * i); }
        auto   gradPos    (size_t i)       { return this->template segment<3>(3 * i); }
        Real_  gradTheta  (size_t j) const { return (*this)[thetaOffset + j]; }
        Real_ &gradTheta  (size_t j)       { return (*this)[thetaOffset + j]; }
        Real_  gradRestLen(size_t j) const { assert(variableRestLen); return (*this)[restLenOffset + j]; }
        Real_ &gradRestLen(size_t j)       { assert(variableRestLen); return (*this)[restLenOffset + j]; }

        size_t thetaOffset = 0;
        size_t restLenOffset = 0;
        bool variableRestLen = false;
    };

    // If "updatedSource" is true, we use the more efficient gradient formulas
    // that are only accurate after a call to updateSourceFrame().
    // These are the formulas from the appendix in Bergou2010.
    // Otherwise, we use the exact gradient formulas for
    // configurations away from where the source frame was set.
    // If "restlenOnly" is passed, only the gradient components corresponding to rest lengths are computed; the
    // centerline position and theta components are left uninitialized.
    template<class StencilMask = GradientStencilMaskIncludeAll> Gradient gradEnergyStretch(                            bool variableRestLen = false, bool restlenOnly = false, const StencilMask &sm = StencilMask()) const;
    template<class StencilMask = GradientStencilMaskIncludeAll> Gradient gradEnergyBend   (bool updatedSource = false, bool variableRestLen = false, bool restlenOnly = false, const StencilMask &sm = StencilMask()) const;
    template<class StencilMask = GradientStencilMaskIncludeAll> Gradient gradEnergyTwist  (bool updatedSource = false, bool variableRestLen = false, bool restlenOnly = false, const StencilMask &sm = StencilMask()) const;
    template<class StencilMask = GradientStencilMaskIncludeAll> Gradient gradEnergy       (bool updatedSource = false, bool variableRestLen = false, bool restlenOnly = false, const StencilMask &sm = StencilMask()) const;

    template<class StencilMask = GradientStencilMaskIncludeAll>
    Gradient gradient(bool updatedSource = false, EnergyType eType = EnergyType::Full, bool variableRestLen = false, bool restlenOnly = false, const StencilMask &sm = StencilMask()) const {
        switch (eType) {
            case EnergyType::   Full: return gradEnergy       (updatedSource, variableRestLen, restlenOnly, sm);
            case EnergyType::   Bend: return gradEnergyBend   (updatedSource, variableRestLen, restlenOnly, sm);
            case EnergyType::  Twist: return gradEnergyTwist  (updatedSource, variableRestLen, restlenOnly, sm);
            case EnergyType::Stretch: return gradEnergyStretch(               variableRestLen, restlenOnly, sm);
            default: throw std::runtime_error("Unknown energy type");
        }
    }

    Gradient gradient(bool updatedSource = false, EnergyType eType = EnergyType::Full, bool variableRestLen = false, bool restlenOnly = false) const {
        return gradient<GradientStencilMaskIncludeAll>(updatedSource, eType, variableRestLen, restlenOnly, GradientStencilMaskIncludeAll());
    }

    // The number of non-zeros in the Hessian's sparsity pattern (a tight
    // upper bound for the number of non-zeros for any configuration).
    size_t hessianNNZ(bool variableRestLen = false) const;

    // Optimizers like Knitro and Ipopt need to know all Hessian entries that
    // could ever possibly be nonzero throughout the course of optimization.
    // The current Hessian may be missing some of these entries.
    // Knowing the fixed sparsity pattern also allows us to more efficiently construct the Hessian.
    CSCMat hessianSparsityPattern(bool variableRestLen = false, Real_ val = 0.0) const;

    TriDiagonalSystem<Real_> hessThetaEnergyTwist() const;

    void hessEnergyStretch(CSCMat &H, bool variableRestLen = false) const;
    void hessEnergyBend   (CSCMat &H, bool variableRestLen = false) const;
    void hessEnergyTwist  (CSCMat &H, bool variableRestLen = false) const;
    void hessEnergy       (CSCMat &H, bool variableRestLen = false) const;
    void hessian(CSCMat &H, EnergyType eType = EnergyType::Full, bool variableRestLen = false) const {
        const size_t ndof = variableRestLen ? numExtendedDoF() : numDoF();
        if ((size_t(H.m) != ndof) || (size_t(H.n) != ndof)) throw std::runtime_error("H size mismatch");
        switch (eType) {
            case EnergyType::   Full: hessEnergy       (H, variableRestLen); break;
            case EnergyType::   Bend: hessEnergyBend   (H, variableRestLen); break;
            case EnergyType::  Twist: hessEnergyTwist  (H, variableRestLen); break;
            case EnergyType::Stretch: hessEnergyStretch(H, variableRestLen); break;
            default: throw std::runtime_error("Unknown energy type");
        }
    }

    // Apply Hessian to "v," accumulate to "result"
    void applyHessEnergy(const VecX &v, VecX &result, bool variableRestLen = false, const HessianComputationMask &mask = HessianComputationMask()) const;

    VecX applyHessian(const VecX &v, bool variableRestLen = false, const HessianComputationMask &mask = HessianComputationMask()) const {
        VecX result(v.size());
        applyHessEnergy(v, result, variableRestLen, mask);
        return result;
    }

    TMatrix hessian(EnergyType eType = EnergyType::Full, bool variableRestLen = false) const {
        auto H = hessianSparsityPattern(variableRestLen);
        hessian(H, eType, variableRestLen);
        auto Htrip = H.getTripletMatrix();

        Htrip.symmetry_mode = TMatrix::SymmetryMode::UPPER_TRIANGLE;
        return Htrip;
    }

    TMatrix massMatrix() const {
        auto M = hessianSparsityPattern();
        massMatrix(M);
        return M.getTripletMatrix();
    }

    void massMatrix(CSCMat &M) const;

    // Provide an interface that's compatible with RodLinkage::massMatrix
    //     switch between lumped/full  mass matrix with boolean "useLumped"
    //     updatedSource does not affect the mass matrix of a single rod.
    void massMatrix(CSCMat &M, bool /* updatedSource */, bool useLumped = false) const {
        if (useLumped) M.setDiag(lumpedMassMatrix(), /* preserveSparsity = */ true);
        else           massMatrix(M);
    }

    // Get the diagonal lumped mass matrix approximation as a vector.
    // Each entry of this vector is the sum of the corresponding row
    // of the full mass matrix. Note that if certain DoFs are constrained
    // (meaning rows/columns of the full mass matrix are removed), this
    // this relationship to the full mass matrix no longer holds. In this
    // case, the lumped mass matrix we return stays positive where
    // the row sums of the constrained full mass matrix could go negative.
    // Physically, this lumped mass matrix neglects the material frame rotation
    // that perturbing a vertex can induce.
    VecX lumpedMassMatrix() const;

    // Approximate the greatest velocity of any point in the rod induced by
    // changing the parameters at rate paramVelocity.
    // ***Assumes that the source frame has been updated***.
    Real_ approxLinfVelocity(const VecX &paramVelocity) const;

    // Get the external forces acting on the rod due to gravity acceleration
    // vector "g", assuming the rod is made of a material with uniform mass
    // density rho.
    // Defaults to using the local gravity at EPFL in m/s^2, close to standard
    // gravity 9.80665. A note on units: g should be in m/s^2 if rho is in
    // kg/mm^3 and force is in Newtons (e.g., if we are using MPa for the
    // Young's modulus and mm for length units). Note here the length unit used
    // to express density should match the simulation length unit, but g is
    // separately expressed using meters for consistency with N = kg m/s^2.
    VecX gravityForce(Real_ rho, const Vec3 &g = Vec3(0.0, 0.0, 9.80635)) const;

    // 1D uniform Laplacian regularization energy for the rest length optimization.
    Real_   restLengthLaplacianEnergy()     const;
    VecX    restLengthLaplacianGradEnergy() const;
    TMatrix restLengthLaplacianHessEnergy() const;

    // Visualize the rod geometry assuming an elliptical cross-section
    // Appends this segment to the existing geometry in vertices/quads
    // If averagedMaterialFrames is true, we create smoother geometry that is guaranteed to be
    // connected if the same cross-section shape is used throughout
    void visualizationGeometry(std::vector<MeshIO::IOVertex> &vertices, std::vector<MeshIO::IOElement> &quads, const bool averagedMaterialFrames = false) const;
    void writeDebugData(const std::string &path) const;
    void saveVisualizationGeometry(const std::string &path, const bool averagedMaterialFrames = false) const;

    // Expand a per-edge or per-vertex field into a per-visualization-vertex or
    // per-visualization-quad field.
    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Derived::ColsAtCompileTime>
    visualizationField(const Derived &field) const {
        bool perVertex = false;
        if (size_t(field.rows()) == numVertices())
            perVertex = true;
        else if (size_t(field.rows()) == numEdges())
            perVertex = false;
        else throw std::runtime_error("Invalid field size " + std::to_string(field.rows()));

        // Predict result size
        size_t resultSize = 0;
        const size_t ne = numEdges();
        for (size_t j = 0; j < ne; ++j) {
            const size_t numCrossSectionPts   = material(j).crossSectionBoundaryPts.size();
            const size_t numCrossSectionEdges = material(j).crossSectionBoundaryEdges.size();
            resultSize += perVertex ? numCrossSectionPts * 2 : numCrossSectionEdges;
        }

        using FieldStorage = Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Derived::ColsAtCompileTime>;
        FieldStorage result(resultSize, field.cols());
        int outIdx = 0;
        for (size_t j = 0; j < ne; ++j) {
            const size_t numCrossSectionPts   = material(j).crossSectionBoundaryPts.size();
            const size_t numCrossSectionEdges = material(j).crossSectionBoundaryEdges.size();
            if (perVertex) {
                for (size_t i = 0; i < numCrossSectionPts; ++i) {
                    result.row(outIdx++) = field.row(j + 0);
                    result.row(outIdx++) = field.row(j + 1);
                }
            }
            else {
                for (size_t i = 0; i < numCrossSectionEdges; ++i)
                    result.row(outIdx++) = field.row(j);
            }
        }
        return result;
    }

    void stressVisualizationGeometry(std::vector<MeshIO::IOVertex> &vertices,
                                     std::vector<MeshIO::IOElement> &quads,
                                     Eigen::VectorXd &sqrtBendingEnergy,
                                     Eigen::VectorXd &stretchingStress,
                                     Eigen::VectorXd &maxBendingStress,
                                     Eigen::VectorXd &minBendingStress,
                                     Eigen::VectorXd &twistingStress) const;

    ////////////////////////////////////////////////////////////////////////////
    // Stress analysis
    ////////////////////////////////////////////////////////////////////////////
    Eigen::VectorXd stretchingStresses() const;
    Eigen::MatrixX2d   bendingStresses() const;
    Eigen::VectorXd   twistingStresses() const;

    Eigen::VectorXd maxBendingStresses() const { return bendingStresses().template  leftCols<1>(); }
    Eigen::VectorXd minBendingStresses() const { return bendingStresses().template rightCols<1>(); }

    size_t numVertices() const { return m_restPoints.size(); }
    size_t numEdges()    const { return (m_restPoints.size() > 0) ? m_restPoints.size() - 1 : 0; }

    // Number of degrees of freedom in the rod.
    size_t numDoF()      const { return numVertices() * 3 + numEdges(); }

    // Number of "extended" degrees of freedom in the rod (deformed configuration + rest lengths)
    size_t numExtendedDoF() const { return numDoF() + numEdges(); }

    // Indices of the variables that always need to be fixed during the rest length solve.
    // In the case of a single elastic rod, these should be specified manually
    // (so none are automatic). The user will likely end up fixing the two endpoints...
    std::vector<size_t> restLenFixedVars() const { return std::vector<size_t>(); }

    // Indices of all length quantity variables; we will want bound constraints
    // to keep these strictly positive.
    std::vector<size_t> lengthVars(bool variableRestLen = false) const {
        std::vector<size_t> result;
        if (!variableRestLen) return result;
        const size_t nrl = numRestLengths(),
                     rlo = restLenOffset();
        for (size_t i = 0; i < nrl; ++i)
            result.push_back(rlo + i);
        return result;
    }

    // For compatibility with RodLinkage
    std::vector<size_t> jointAngleDoFIndices() const { return std::vector<size_t>(); }
    void setAverageJointAngle(Real_ /* tgt */) { throw std::runtime_error("ElasticRod::setAverageJointAngle should not be called!"); }

    struct Directors {
        Directors(Vec3 _d1, Vec3 _d2) : d1(_d1), d2(_d2) { }
        Vec3 d1, d2;
        // Zero indexed: 0 ==> d1, 1 ==> d2
        const Vec3 &get(size_t i) const {
            if (i == 0) return d1;
            if (i == 1) return d2;
            throw std::runtime_error("Invalid director index: " + std::to_string(i));
        }
    };

    // Deformed configuration
    // We need more than just the deformed centerline positions and material frame
    // angles (theta) to specify the rod's state; there is also the "hidden state" of the
    // reference frame. We collect all of this state, as well as a cache for
    // derived quantities that are frequently used, in a class that can be
    // saved and restored.
    struct DeformedState {
        // Initialize/reset to the identity deformation for a particular rod.
        void initialize(const ElasticRod_T &rod);
        void update(const std::vector<Pt3> &points, const std::vector<Real_> &thetas);

        const std::vector<Pt3  > &points() const { return m_point; }
        const std::vector<Real_> &thetas() const { return m_theta; }

        const Pt3 &point(size_t i) const { return m_point[i]; }
        Real_      theta(size_t j) const { return m_theta[j]; }

        // Update the reference frame with parallel transport (in space) so
        // that the reference twist at each vertex is newTwist, adjusting
        // thetas so that the material frame/energy is unchanged.
        // This can be useful for debugging.
        void setReferenceTwist(Real_ newTwist = 0);

        std::vector<Directors> referenceDirectors; // Time-parallel reference frame (per edge)
        std::vector<Real_>     referenceTwist;     // Twist from one edge's current reference frame to the next (per vertex, derived from referenceDirectors)
        std::vector<Vec3>      tangent;            // Unit tangent vectors (per edge)
        std::vector<Directors> materialFrame;      // Material frame vectors (per edge)
        std::vector<Vec3>      kb;                 // Curvature binormal (per vertex)
        StdVectorVector2D      kappa;              // Curvature normal in vertex material coordinate system ("average" of edges' coordinate systems)
        std::vector<Real_>     len;                // Deformed edge lengths (per edge)

        StdVectorMatrix2D per_corner_kappa; // per_corner_kappa[i].col(j)[k] = per_corner_kappa[i](k, j) = (kappa_k)_i^(j + i - 1)

        // The source reference frame from which to construct the
        // current reference frames by parallel transport.
        // This should typically be the frame from the previous optimization step.
        // Note: the Hessian and gradEnergy*(updatedSource = true)
        // calculations will only be accurate if updateSourceFrame() has been
        // run since the last time the points/thetas were updated.
        std::vector<Vec3>      sourceTangent;
        std::vector<Directors> sourceReferenceDirectors;
        std::vector<Directors> sourceMaterialFrame;  // Material frame based on ***current thetas*** and source reference directors
        std::vector<Real_>     sourceTheta;          // Material frame angles to be used for enforcing temporal coherence when determining the angle from a material frame vector
        std::vector<Real_>     sourceReferenceTwist; // Reference twist used to enforce temporal coherence

        // Update the source quantities used to define the current reference
        // frame by parallel transport as well as to avoid jumps in angles when
        // resolving the multiple-of-2Pi ambiguity that arises when determining
        // angles from reference vectors.
        // Note: this function updates the energy landscape (but not the
        // current energy value) due to the change in the parallel transport
        // source frame. Changing sourceReferenceTwist also alters the
        // landscape, but only by shifting discontinuities (the landscape
        // within the region bounded by these discontinuities is unchanged).
        void updateSourceFrame() {
            sourceTangent 			 = tangent;
            sourceReferenceDirectors = referenceDirectors;
            sourceMaterialFrame      = materialFrame; // Note, this is also updated when current thetas change...
            sourceTheta              = m_theta;
            sourceReferenceTwist     = referenceTwist;
        }

    private:
        std::vector<Pt3  > m_point; // Current position of each vertex
        std::vector<Real_> m_theta;  // Angle from reference director to first material frame vector d1
    };

    ////////////////////////////////////////////////////////////////////////////
    // Accessors to be used for serialization only.
    ////////////////////////////////////////////////////////////////////////////
    void setInitialMinRestLen(Real_ val) { m_initMinRestLen = val; }

private:
    // Rest configuration
    std::vector<Pt3>       m_restPoints;    // Original position of each vertex
    std::vector<Directors> m_restDirectors; // Original reference frame (zero twist)
    StdVectorVector2D      m_restKappa;     // Rest curvature
    std::vector<Real_>     m_restTwist;     // Rest twist
    std::vector<Real_>     m_restLen;       // Rest length of each edge

    std::vector<DeformedState> m_deformedStates;

    // Constitutive parameters
    std::vector<Real_> m_density;             // per-edge; used to avoid double-counting stiffness/mass for edges shared at the joints. Note, this does not affect bending or twisting stiffness!
    std::vector<Real_> m_stretchingStiffness; // per-edge
    std::vector<Real_> m_twistingStiffness;   // per-vertex
    // Note: the constitutive parameters determining the bending response form a symmetric matrix. We assume the matrix has been
    // diagonalized so that the material frame vectors correspond to its eigenvectors, thus reducing to two parameters.
    std::vector<RodMaterial::BendingStiffness> m_bendingStiffness; // per-vertex

    // Representation of the rod material used for visualization/mass matrix construction.
    // When the rod is made of a homogeneous material, this vector holds a single material;
    // otherwise, it should hold a RodMaterial per edge.
    // Note that this is *decoupled* from the stiffness value (so a different,
    // possibly non-physical per-edge stiffness can be manually configured).
    std::vector<RodMaterial> m_edgeMaterial;

    BendingEnergyType m_bendingEnergyType = BendingEnergyType::Bergou2008;

    Real_ m_initMinRestLen = 0;
};

#endif /* end of include guard: ELASTICROD_HH */
