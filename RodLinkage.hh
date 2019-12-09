////////////////////////////////////////////////////////////////////////////////
// RodLinkage.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Represents a scissor linkage assembly of elastic rods. This assembly
//  consists of a collection of joints where two elastic rods meet and can
//  pivot relative to each other.
//
//  This network can be represented as a graph where each vertex has valence 1
//  or 4. Valence 1 vertices correspond to free ends of rods, while valence 4
//  vertices correspond to joints. The edges connecting vertices represent
//  an elastic rod segment.
//
//  There are two types of joints: free joints and constrained joints. Free
//  joints apply only the pin constraint to ensure the incident rods connect at
//  and pivot around an edge midpoint. A constrained joint fixes the joint's 3D
//  position and the orientation of the incident rods. Typically there will be
//  only one constrained joint per assembly to pin down the global orientation
//  and linkage opening (for structures with a single opening degree of
//  freedom).
//
//  Constraints are applied explicitly by using a reduced set of parameters. To
//  simplify this process, the elastic rod segments making up a full rod in the
//  assembly are individually modeled as separate discrete elastic rods. In
//  other words, the joints mark the start and end of the discrete elastic
//  rods. The parameters of the joint fully determine the first/last edges of
//  the four incident elastic rod segments. Conceptually, the incoming and
//  outgoing segments of a single rod share an overlapping edge at the joint.
//  This properly couples all rods with the only side-effect that the
//  stretching stiffnesses of the connection edges are double counted. This
//  side effect can be countered by halving the material "density" of these
//  connection edges.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  03/18/2018 12:52:41
////////////////////////////////////////////////////////////////////////////////
#ifndef RODLINKAGE_HH
#define RODLINKAGE_HH

#include "ElasticRod.hh"
#include "TemplatedTypes.hh"
#include "AutomaticDifferentiation.hh"
#include "RectangularBox.hh"
#include <rotation_optimization.hh>
#include <array>
#include <tuple>

template<typename Real_>
struct LinkageTerminalEdgeSensitivity; // Defined in LinkageTerminalEdgeSensitivity.hh

// Templated to support automatic differentiation types.
template<typename Real_>
struct RodLinkage_T;

using RodLinkage = RodLinkage_T<Real>;

template<typename Real_>
struct RodLinkage_T {
    using Vec3   = Vec3_T<Real_>;
    using Mat3   = Mat3_T<Real_>;
    using Pt3    =  Pt3_T<Real_>;
    using Vec2   = Vec2_T<Real_>;
    using VecX   = VecX_T<Real_>;
    using CSCMat = CSCMatrix<SuiteSparse_long, Real_>;
    using ropt   = rotation_optimization<Real_>;
    using Rod    = ElasticRod_T<Real_>;

    using TMatrix = TripletMatrix<Triplet<Real_>>;
    using EnergyType  = typename Rod::EnergyType;
    using BEnergyType = typename Rod::BendingEnergyType;

    static constexpr size_t defaultSubdivision = 10;
    static constexpr bool defaultConsistentAngle = true;
    static constexpr size_t NONE = std::numeric_limits<size_t>::max();
    struct RodSegment;
    struct Joint;

    // Construct empty linkage, to be initialized later by calling set.
    RodLinkage_T() { }

    // Forward all constructor arguments to set(...)
    template<typename... Args>
    RodLinkage_T(Args&&... args) {
        set(std::forward<Args>(args)...);
    }

    RodLinkage_T(const RodLinkage_T &linkage) { set(linkage); } // The above forwarding constructor confuses pybind11

    // Read the rod linkage from a line graph file.
    void read(const std::string &path, size_t subdivision = defaultSubdivision, bool initConsistentAngle = defaultConsistentAngle);

    // Initialize by copying from another linkage
    template<typename Real2_>
    void set(const RodLinkage_T<Real2_> &linkage) {
        set(linkage.joints(), linkage.segments(), linkage.material(), linkage.initialMinRestLength(), linkage.segmentRestLenToEdgeRestLenMapTranspose(), linkage.getPerSegmentRestLength());
    }

    // Initialize by copying the passed joints, segments, material, and initial rest length
    template<typename Real2_>
    void set(const std::vector<typename RodLinkage_T<Real2_>::Joint> &joints, const std::vector<typename RodLinkage_T<Real2_>::RodSegment> &segments,
             const RodMaterial &mat, Real2_ initMinRL, const typename RodLinkage_T<Real2_>::CSCMat &segmentRestLenToEdgeRestLenMapTranspose,
             const VecX_T<Real2_> &perSegmentRestLen) {
        m_joints.clear();
        m_joints.reserve(joints.size());
        for (const auto &j : joints) m_joints.emplace_back(j);

        m_segments.clear();
        m_segments.reserve(segments.size());
        for (const auto &s : segments) m_segments.emplace_back(s);

        m_material = mat;

        // Update the degree of freedom info/redirect the joints to point at this linkage
        m_buildDoFOffsets();
        for (auto &j : m_joints) { j.updateLinkagePointer(this); }

        m_initMinRestLen = initMinRL;
        m_segmentRestLenToEdgeRestLenMapTranspose = segmentRestLenToEdgeRestLenMapTranspose;
        m_perSegmentRestLen = perSegmentRestLen;
        m_clearCache();
        m_sensitivityCache.clear();
    }

    // Initialize the rod linkage from a line graph.
    void set(const std::string &path,
             size_t subdivision = defaultSubdivision, bool consistentAngle = defaultConsistentAngle) { read(path, subdivision, consistentAngle); }

    // Initialize the rod linkage from a line graph.
    void set(std::vector<MeshIO::IOVertex > vertices, // copy edited inside
             std::vector<MeshIO::IOElement> edges,    // copy edited inside
             size_t subdivision = defaultSubdivision, 
             bool consistentAngle = defaultConsistentAngle);

    void set(const Eigen::MatrixX3d &vertices,
             const Eigen::MatrixX2i &edges,
             size_t subdivision = defaultSubdivision, bool consistentAngle = defaultConsistentAngle) {
        std::vector<MeshIO::IOVertex > ioVertices;
        std::vector<MeshIO::IOElement> ioEdges;

        const size_t nv = vertices.rows(), ne = edges.rows();
        ioVertices.reserve(nv);
        ioEdges.reserve(ne);

        for (size_t i = 0; i < nv; ++i)
            ioVertices.emplace_back(vertices(i, 0), vertices(i, 1), vertices(i, 2));

        for (size_t i = 0; i < ne; ++i)
            ioEdges.emplace_back(edges(i, 0), edges(i, 1));

        set(ioVertices, ioEdges, subdivision, consistentAngle);
    }

    // Avoid accidentally copying linkages around for performance reasons;
    // explicitly use RodLinkage::set instead.
    // If we choose to offer this operator in the future, it should be
    // implemented as a call to set (the joint linkage pointers must be updated)
    RodLinkage_T &operator=(const RodLinkage_T &b) = delete;

    void setMaterial(const RodMaterial &material);
    RodMaterial material() const { return m_material; }
    void setStretchingStiffness(Real_ val);
    void setBendingEnergyType(BEnergyType betype) {
        for (auto &s : m_segments) s.rod.setBendingEnergyType(betype);
    }

    // Scale the bending and twisting stiffnesses of vertices falling within
    // the regions specified by "boxes"
    void stiffenRegions(const RectangularBoxCollection &boxes, Real factor) {
        for (auto &s : m_segments) {
            auto &r = s.rod;
            const size_t nv = r.numVertices();
            const auto &rp = r.restPoints();
            for (size_t i = 0; i < nv; ++i) {
                if (boxes.contains(stripAutoDiff(rp[i]))) {
                    r.bendingStiffness(i).lambda_1 *= factor;
                    r.bendingStiffness(i).lambda_2 *= factor;
                    r.twistingStiffness(i) *= factor;
                }
            }
        }
    }

    // Set the rest length of each rod edge to its current deformed length.
    void updateRestLength() {
        for (auto &s : m_segments)
            s.rod.setRestLengths(s.rod.lengths());
    }

    // Design optimization: currently we just optimize the rest lengths
    const VecX &getDesignParameters() { return m_perSegmentRestLen; }
    void setDesignParameters(const Eigen::Ref<const VecX> &p) {
        m_perSegmentRestLen = p;
        m_setRestLengthsFromPSRL();
    }
    size_t numDesignParams() const { return numSegments(); }
    // Gradient of elastic energy with respect to the design parameters
    VecX grad_design_parameters(bool updatedSource = false) {
        auto gPerEdgeRestLen = gradient(updatedSource, EnergyType::Full, true, /* only compute rest length components */ true);
        VecX result(numSegments());
        m_segmentRestLenToEdgeRestLenMapTranspose.applyRaw(gPerEdgeRestLen.tail(numRestLengths()).data(), result.data(), /* no transpose */ false);
        return result;
    }

    // The shortest rest-length of any rod in the linkage defines the characteristic lengthscale of this network.
    // (For purposes of determining reasonable descent velocites).
    Real_ characteristicLength() const {
        Real_ minLen = std::numeric_limits<float>::max();
        for (auto &s : m_segments) minLen = std::min(minLen, s.rod.characteristicLength());
        return minLen;
    }

    // Set the current adapted curve frame as the source for parallel transport.
    // See ElasticRod::updateSourceFrame for more discussion.
    // Also set each joint's source normal used to encourage temporal coherence
    // of normals as the linkage's opening angle reverses sign.
    void updateSourceFrame() {
        for (auto &s : m_segments) s.rod.updateSourceFrame();
        m_sensitivityCache.clear();
    }

    // Apply each joint's current rotation to its source frame, resetting the
    // joint rotation variables to the identity.
    // This could be done at every iteration of Newton's method to
    // speed up rotation gradient/Hessian calculation, or only when needed
    // as the rotation magnitude becomes too great (our rotation parametrization
    // has a singularity when the rotation angle hits pi).
    void updateRotationParametrizations() {
        for (auto &j : m_joints) j.updateParametrization();
        m_sensitivityCache.clear();
    }

    size_t numDoF() const;
    size_t numSegments() const { return segments().size(); }
    size_t numJoints()   const { return m_joints.size(); }

    // Parameter order: all segment parameters, followed by all joint parameters
    VecX getDoFs() const;
    void setDoFs(const Eigen::Ref<const VecX> &dofs, bool spatialCoherence = false);
    void setDoFs(const std::vector<Real_> &dofs) { setDoFs(Eigen::Map<const VecX>(dofs.data(), dofs.size())); }

    // Extended parameters: ordinary DoFs + rest lengths
    // Rest length parameter ordering: all rest lengths for segments' interior
    // and free end edges, followed by two rest lengths for each joint.
    size_t numExtendedDoF() const { return numDoF() + numRestLengths(); }
    size_t numExtendedDoFPSRL() const { return numDoF() + numSegments(); }
    size_t restLenOffset() const { return numDoF(); }

    size_t dofOffsetForJoint  (size_t ji) const { return m_dofOffsetForJoint  .at(ji); }
    size_t dofOffsetForSegment(size_t si) const { return m_dofOffsetForSegment.at(si); }
    size_t restLenDofOffsetForJoint  (size_t ji) const { return m_restLenDoFOffsetForJoint  .at(ji); }
    size_t restLenDofOffsetForSegment(size_t si) const { return m_restLenDoFOffsetForSegment.at(si); }

    // Get the index of the joint closest of the center of the structure.
    // This is usually a good choice for the joint used to constrain the
    // structures global rigid motion/drive it open.
    size_t centralJoint() const {
        Pt3 center(Pt3::Zero());
        for (const auto j : m_joints) { center += j.pos(); }
        center /= numJoints();
        Real_ closestDistSq = safe_numeric_limits<Real_>::max();
        size_t closestIdx = 0;
        for (size_t ji = 0; ji < numJoints(); ++ji) {
            Real_ distSq = (m_joints[ji].pos() - center).squaredNorm();
            if (distSq < closestDistSq) {
                closestDistSq = distSq;
                closestIdx = ji;
            }
        }
        return closestIdx;
    }

    size_t numRestLengths() const;
    VecX getRestLengths() const;
    Real_ minRestLength() const { return getRestLengths().minCoeff(); }
    CSCMat segmentRestLenToEdgeRestLenMapTranspose() const { return m_segmentRestLenToEdgeRestLenMapTranspose; }

    // The minimum rest length set at construction time. This is useful to get a reasonable
    // lower bound on the rest/deformed length variables.
    // (We can't use the current minRestLength() because the rest length optimization could be
    //  run in a loop, allowing the minimum rest length to shrink by a constant factor with each run.)
    Real_ initialMinRestLength() const { return m_initMinRestLen; }

    VecX getExtendedDoFs() const;
    void setExtendedDoFs(const VecX &params, bool spatialCoherence = false);

    VecX getExtendedDoFsPSRL() const;
    void setExtendedDoFsPSRL(const VecX &params, bool spatialCoherence = false);

    void setPerSegmentRestLength(const VecX &psrl) { m_perSegmentRestLen = psrl; m_setRestLengthsFromPSRL(); }
    VecX getPerSegmentRestLength() const { return m_perSegmentRestLen; }

    // Warning: this method could be dangerous--needs more thorough testing
    // (to ensure 2*pi twists aren't introduced).
    // Change to using the opposite (supplementary) choice of joint angles as the
    // opening angle definition. Do this while preserving the A/B labeling so
    // as not to invalidate m_segmentRestLenToEdgeRestLenMapTranspose.
    // Note: this flips the joint normals!
    void swapJointAngleDefinitions() {
        for (auto &j : m_joints) {
            j.swapAngleDefinition(); // side effect: swaps rod labels!
            j.swapRodLabels();       // revert to old labels, flipping the normals
        }

        // Flipping the normals twisted each rod's terminal edges by + or - pi;
        // Attempt to untwist the rods.
        setDoFs(getDoFs());
        for (auto &s : m_segments)
            s.setMinimalTwistThetas();
        updateSourceFrame();
    }

    // Compute the average over all joints of the joint opening angle.
    Real_ getAverageJointAngle() const {
        Real_ result = 0;
        for (const auto &j : m_joints) result += j.alpha();
        return result / numJoints();
    }

    // Change the average joint opening angle by uniformly scaling all joint openings.
    // (This only changes the angles/incident segment edges. No equilibrium solve is run.)
    void setAverageJointAngle(const Real_ alpha) {
        const Real_ curr = getAverageJointAngle();
        const Real_ scale = alpha / curr;
        for (auto &j : m_joints)
            j.set_alpha(j.alpha() * scale);
    }

    // Compute the average over all joints of the joint opening angle.
    Real_ getMinJointAngle() const {
        Real_ result = safe_numeric_limits<Real>::max();
        for (const auto &j : m_joints) result = std::min(result, j.alpha());
        return result;
    }

    // Rest length solve interface: use a rest length per RodSegment, not per edge.
    size_t numRestlenSolveDof()                                                                const { return numExtendedDoFPSRL(); }
    size_t numRestlenSolveRestLengths()                                                        const { return numSegments(); }
    VecX getRestlenSolveDoF()                                                                  const { return getExtendedDoFsPSRL(); }
    void setRestlenSolveDoF(const VecX &params)                                                      { return setExtendedDoFsPSRL(params); }
    VecX restlenSolveGradient(bool updatedSource = false, EnergyType eType = EnergyType::Full) const { return gradientPerSegmentRestlen(updatedSource, eType); }
    CSCMat restlenSolveHessianSparsityPattern()                                                const { return hessianPerSegmentRestlenSparsityPattern(); }
    void restlenSolveHessian(CSCMat &H, EnergyType etype = EnergyType::Full)                   const { hessianPerSegmentRestlen(H, etype); }
    std::vector<size_t> restlenSolveLengthVars() const {
        auto result = lengthVars(false); // Omit the per-edge rest length vars!
        const size_t rlo = restLenOffset();
        for (size_t si = 0; si < numSegments(); ++si) result.push_back(rlo + si);
        return result;
    }

    // Indices of the degrees of freedom controlling the joint center positions.
    std::vector<size_t> jointPositionDoFIndices() const {
        std::vector<size_t> result;
        const size_t nj = numJoints();
        result.reserve(3 * nj);
        for (size_t i = 0; i < nj; ++i) {
            size_t o = m_dofOffsetForJoint[i];
            result.push_back(o + 0);
            result.push_back(o + 1);
            result.push_back(o + 2);
        }
        return result;
    }

    // Indices of degrees of freedom controlling joint openings.
    std::vector<size_t> jointAngleDoFIndices() const {
        std::vector<size_t> result;
        result.reserve(numJoints());
        for (size_t j = 0; j < numJoints(); ++j)
            result.push_back(m_dofOffsetForJoint[j] + 6); // pos, omega, alpha <--
        return result;
    }

    // Indices of all degrees of freedom controlling joints.
    std::vector<size_t> jointDoFIndices() const {
        std::vector<size_t> result;
        for (size_t j = 0; j < numJoints(); ++j) {
            size_t o = m_dofOffsetForJoint[j];
            size_t ndof = m_joints[j].numDoF();
            for (size_t d = 0; d < ndof; ++d)
                result.push_back(o + d);
        }
        return result;
    }

    // Indices of all variables that must be fixed during the rest length solve
    // This consists of all joint positions, plus the free rod endpoints.
    std::vector<size_t> restLenFixedVars() const {
        std::vector<size_t> result = jointPositionDoFIndices();
        // Fix the free ends
        for (size_t si = 0; si < numSegments(); ++si) {
            const auto &s = segment(si);
            if (!s.hasStartJoint()) {
                size_t o = m_dofOffsetForSegment[si];
                result.push_back(o); result.push_back(o + 1); result.push_back(o + 2);
            }
            if (!s.hasEndJoint()) {
                size_t o = m_dofOffsetForSegment[si] + 3 * (s.rod.numVertices() - 1 - 2 * s.hasStartJoint());
                result.push_back(o); result.push_back(o + 1); result.push_back(o + 2);
            }
        }

        return result;
    }

    // Indices of all length quantity variables; we will want bound constraints
    // to keep these strictly positive.
    std::vector<size_t> lengthVars(bool variableRestLen = false) const {
        std::vector<size_t> result;
        // The two variables for each joint...
        for (size_t ji = 0; ji < numJoints(); ++ji) {
            result.push_back(m_dofOffsetForJoint[ji] + 7);
            result.push_back(m_dofOffsetForJoint[ji] + 8);
        }
        // ... and all the rest lengths, if requested
        if (variableRestLen) {
            const size_t nrl = numRestLengths(),
                         rlo = restLenOffset();

            for (size_t i = 0; i < nrl; ++i)
                result.push_back(rlo + i);
        }
        return result;
    }

    // Get a list of all the centerline positions in the network.
    std::vector<Pt3> deformedPoints() const {
        std::vector<Pt3> result;
        for (const auto &s : m_segments) {
            const auto &dp = s.rod.deformedPoints();
            result.insert(result.end(), dp.begin(), dp.end());
        }
        return result;
    }

    // The joint positions flattened in x0, y0, z0, x1, y1, z1, ... order
    VecX jointPositions() const {
        VecX result(3 * numJoints());
        for (size_t ji = 0; ji < numJoints(); ++ji)
            result.template segment<3>(3 * ji) = m_joints[ji].pos();
        return result;
    }

    // Elastic energy stored in the linkage
    Real_ energy() const;
    Real_ energyStretch() const;
    Real_ energyBend() const;
    Real_ energyTwist() const;

    // The maximum elastic energy stored in any individual rod.
    Real_ maxRodEnergy() const {
        Real_ result = 0;
        for (const auto &s : m_segments) result = std::max(result, s.rod.energy());
        return result;
    }

    Real_ energy(EnergyType type) const {
        switch (type) {
            case EnergyType::   Full: return energy();
            case EnergyType::   Bend: return energyBend();
            case EnergyType::  Twist: return energyTwist();
            case EnergyType::Stretch: return energyStretch();
            default: throw std::runtime_error("Unknown energy type");
        }
    }

    // The maximum magnitude strain appearing anywhere in the linkage
    Real_ maxStrain() const;

    // Compute the generalized forces acting from the "A" rods on the "B" rods.
    // This indicates the equal and opposite forces acting at each joint rivet.
    // Note: the rotation parametrization should be updated before calling
    // this method!
    VecX rivetForces(EnergyType eType = EnergyType::Full) const;

    // Get the net force and torque acting from the "A" rods at the center of each joint.
    // Returns a Jx6 matrix where the first three columns hold the net force on each
    // joint and the last three columns hold the net torque.
    Eigen::MatrixXd rivetNetForceAndTorques(EnergyType eType = EnergyType::Full) const;

    // Gradient of the linkage's elastic energy with respect to all degrees of freedom.
    // If "updatedSource" is true, we use the more efficient gradient formulas
    // that are only accurate after a call to updateSourceFrame().
    // If "skipBRods" is true, only the contributions from "A rods" are
    // accumulated; see rivetForces.
    VecX gradient(bool updatedSource = false, EnergyType eType = EnergyType::Full, bool variableRestLen = false, bool restlenOnly = false, const bool skipBRods = false) const;
    VecX gradientPerSegmentRestlen(bool updatedSource = false, EnergyType eType = EnergyType::Full) const;

    // The number of non-zeros in the Hessian's sparsity pattern (a tight
    // upper bound for the number of non-zeros for any configuration).
    size_t hessianNNZ(bool variableRestLen = false) const;

    // Optimizers like Knitro and Ipopt need to know all Hessian entries that
    // could ever possibly be nonzero throughout the course of optimization.
    // The current Hessian may be missing some of these entries.
    // Knowing the fixed sparsity pattern also allows us to more efficiently construct the Hessian.
    CSCMat hessianSparsityPattern(bool variableRestLen = false, Real_ val = 0.0) const;

    // Accumulate the Hessian into the sparse matrix "H," which must already be initialized
    // with the sparsity pattern.
    void hessian(CSCMat &H, EnergyType eType = EnergyType::Full, const bool variableRestLen = false) const;

    // Hessian of the linkage's elastic energy with respect to all degrees of freedom.
    TMatrix hessian(EnergyType eType = EnergyType::Full, const bool variableRestLen = false) const;

    CSCMat hessianPerSegmentRestlenSparsityPattern(Real_ val = 0.0) const;
    void hessianPerSegmentRestlen(CSCMat &H, EnergyType etype = EnergyType::Full) const;

    VecX applyHessian(const VecX &v, bool variableRestLen = false, const HessianComputationMask &mask = HessianComputationMask()) const;
    VecX applyHessianPerSegmentRestlen(const VecX &v, const HessianComputationMask &mask = HessianComputationMask()) const;

    // useLumped: whether to use the rods' diagonal lumped mass matrices.
    // The linkage's mass matrix will be non-diagonal in either case because the joint
    // parameters control multiple rod centerline point/theta variables.
    void massMatrix(CSCMat &M, bool updatedSource = false, bool useLumped = false) const;
    TMatrix massMatrix(bool updatedSource = false, bool useLumped = false) const {
        auto M = hessianSparsityPattern();
        massMatrix(M, updatedSource, useLumped);
        return M.getTripletMatrix();
    }
    // Probably not useful: this matrix is usually not positive definite
    VecX lumpedMassMatrix(bool updatedSource = false) const;

    // Approximate the greatest velocity of any point in the rod induced by
    // changing the parameters at rate paramVelocity.
    // ***Assumes that the source frame has been updated***.
    Real_ approxLinfVelocity(const VecX &paramVelocity) const;

    // 1D uniform Laplacian regularization energy for the rest length optimization.
    // Note: gradient is of size numRestLengths(),
    //       Hessian is (numRestLengths() x numRestLengths())
    Real_   restLengthLaplacianEnergy()     const;
    VecX    restLengthLaplacianGradEnergy() const;
    TMatrix restLengthLaplacianHessEnergy() const;

    void visualizationGeometry(std::vector<MeshIO::IOVertex > &vertices,
                               std::vector<MeshIO::IOElement> &quads,
                               const bool averagedMaterialFrames = false) const;

    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Derived::ColsAtCompileTime>
    visualizationField(const std::vector<Derived> &perRodSegmentFields) const {
        if (perRodSegmentFields.size() != numSegments()) throw std::runtime_error("Invalid field size");
        using FieldStorage = Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Derived::ColsAtCompileTime>;
        std::vector<FieldStorage> perRodVisualizationFields;
        perRodVisualizationFields.reserve(perRodSegmentFields.size());
        int fullSize = 0;
        const int cols = perRodSegmentFields.at(0).cols();
        for (size_t si = 0; si < numSegments(); ++si) {
            if (cols != perRodSegmentFields[si].cols()) throw std::runtime_error("Mixed field types forbidden.");
            perRodVisualizationFields.push_back(segment(si).rod.visualizationField(perRodSegmentFields[si]));
            fullSize += perRodVisualizationFields.back().rows();
        }
        FieldStorage result(fullSize, cols);
        int offset = 0;
        for (const auto &vf : perRodVisualizationFields) {
            result.block(offset, 0, vf.rows(), cols) = vf;
            offset += vf.rows();
        }
        assert(offset == fullSize);
        return result;
    }

    void saveVisualizationGeometry(const std::string &path, const bool averagedMaterialFrames = false) const;
    void saveStressVisualization(const std::string &path) const;

    // Output a line mesh holding each rod's centerline and data fields
    // describing the rods' internal states.
    void writeRodDebugData(const std::string &path, const size_t singleRod = NONE) const;
    // Output a line mesh representing the linkage graph with information about the joints.
    void writeLinkageDebugData(const std::string &path) const;

    // originJoint[i]: which joint vertex i was created from (NONE if vertex "i" did not originate from a joint).
    void triangulation(std::vector<MeshIO::IOVertex> &vertices, std::vector<MeshIO::IOElement> &tris, std::vector<size_t> &originJoint) const;
    void writeTriangulation(const std::string &path) const {
        std::vector<MeshIO::IOVertex > vertices;
        std::vector<MeshIO::IOElement> elements;
        std::vector<size_t> dummy;
        triangulation(vertices, elements, dummy);
        MeshIO::save(path, vertices, elements);
    }

    const std::vector<RodSegment> &segments() const { return m_segments; }
          std::vector<RodSegment> &segments()       { return m_segments; }
    const std::vector<Joint>      &joints()   const { return m_joints; }
          std::vector<Joint>      &joints()         { return m_joints; }

    const Joint &joint(size_t i) const { return m_joints.at(i); }
          Joint &joint(size_t i)       { return m_joints.at(i); }

    const RodSegment &segment(size_t i) const { return m_segments.at(i); }
          RodSegment &segment(size_t i)       { return m_segments.at(i); }

    template<typename Getter>
    std::vector<Eigen::VectorXd> collectRodScalarFields(const Getter &getter) const {
        std::vector<Eigen::VectorXd> result;
        result.reserve(numSegments());
        for (size_t si = 0; si < numSegments(); ++si)
            result.push_back(getter(segment(si).rod));
        return result;
    }
    std::vector<Eigen::VectorXd> sqrtBendingEnergies() const { return collectRodScalarFields([](const Rod &r) { return stripAutoDiff(r.energyBendPerVertex()).array().sqrt().eval(); }); }
    std::vector<Eigen::VectorXd>  stretchingStresses() const { return collectRodScalarFields([](const Rod &r) { return r.stretchingStresses(); }); }
    std::vector<Eigen::VectorXd>  maxBendingStresses() const { return collectRodScalarFields([](const Rod &r) { return r.maxBendingStresses(); }); }
    std::vector<Eigen::VectorXd>  minBendingStresses() const { return collectRodScalarFields([](const Rod &r) { return r.minBendingStresses(); }); }
    std::vector<Eigen::VectorXd>    twistingStresses() const { return collectRodScalarFields([](const Rod &r) { return r.  twistingStresses(); }); }

    // Partition the segment indices into list of segments making up each rod (i.e., polylines)
    // Within each rod, segment indices are listed in order. We attempt to pick a
    // consistent direction for all "A" rods and all "B" rods (with A and B oriented oppositely).
    std::vector<std::tuple<bool, std::vector<size_t>>> traceRods() const;

    // Get the "stress" (sqrt bending energy) values for each vertex along a full continuous rod.
    // Also get the "texture coordinates" of these stress values, where
    // the rod parameterization puts the joints at integer parameter values.
    // The ordering of the rods matches the return value of traceRods()
    // return: (vector holding joint index   vector for each rod,
    //          vector holding stress sample vector for each rod,
    //          vector holding parametric coordinate vector of stress samples for each rod)
    std::tuple<std::vector<std::vector<size_t>>,
               std::vector<std::vector<double>>,
               std::vector<std::vector<double>>> rodStresses() const;

    void florinVisualizationGeometry(std::vector<std::vector<size_t>> &polylinesA,
                                     std::vector<std::vector<size_t>> &polylinesB,
                                     std::vector<Point3D> &points, std::vector<Vector3D> &normals,
                                     std::vector<Real> &stresses) const;

    ////////////////////////////////////////////////////////////////////////////
    // Entity representations
    ////////////////////////////////////////////////////////////////////////////
    // Represent a joint between rod A and rod B
    // (where rod segments A0, A1, B0, and B1 meet)
    struct Joint {
        // Converting constructor from a different floating point type.
        // Unfortunately, we can't just template this on the floating point type
        // of the surrounding RodLinkage_T struct, since then the compiler isn't
        // able to deduce the template parameter...
        template<typename Joint2>
        Joint(const Joint2 &j)
            : m_linkage(nullptr) { m_setState<decltype(j.alpha())>(j.getState()); }

        Joint(RodLinkage_T *l, const Pt3 &p, const Vec3 &eA, const Vec3 &eB,
              const std::array<size_t, 2> segmentsA, const std::array<size_t, 2> segmentsB,
              const std::array<bool  , 2>  isStartA, const std::array<bool  , 2>  isStartB);

        constexpr size_t  numDoF() const { return 9; }
        size_t valence() const { return numSegmentsA() + numSegmentsB(); }
        size_t numSegmentsA() const { return 1 + (m_segmentsA[1] != NONE); }
        size_t numSegmentsB() const { return 1 + (m_segmentsB[1] != NONE); }

        // Call "f(ji, si, AB)" for each adjacent joint index ji.
        // Also pass "si", the index of the segment connecting to joint "ji",
        //       and "AB", whether this segment belongs to rod A (0) or B (1) of joint "ji"
        // If restrict_AB is 0 or 1, then only visit joints attached via rod A or B, respectively
        template<class F>
        void visitNeighbors(const F &f, const size_t restrict_AB = 2) const;

        // Set the joint parameters from a collection of global variables (DoFs)
        template<class Derived>
        void setParameters(const Eigen::DenseBase<Derived> &vars);

        // Extract the joint parameters, storing them in a collection of global variables (DoFs)
        template<class Derived>
        void getParameters(Eigen::DenseBase<Derived> &vars) const;

        // Flip orientation to best align with joint b's normal.
        // This is done by swapping the A and B labels (we want to preserve alpha's sign)
        void makeNormalConsistent(const Joint &b) {
            if (m_source_normal.dot(b.m_source_normal) < 0)
                swapRodLabels();
        }

        // +-----------------------+
        // |  B     A      A    -B |
        // |   \.-./        \   /  |
        // |    \ /          \ /   |
        // |     X     ==>  ( X    |
        // |    / \          / \   |
        // |   /   \        /   \  |
        // |               B       |
        // +-----------------------+
        // Change the definition of alpha, replacing it with its complement.
        // This also exchanges the labels of A and B (changing B's sign to avoid flipping the normal)
        // and rotates the bisector by pi/2.
        void swapAngleDefinition();

        // Exchange the roles of Rod A and B; this allows us to flip the joint
        // normal without changing alpha.
        void swapRodLabels() {
            std::swap(m_segmentsA, m_segmentsB);
            std::swap(m_isStartA, m_isStartB);
            std::swap(m_len_A, m_len_B);
            // Leave m_sign_B unchanged (since we don't have a mechanism for flipping A),
            // but if m_sign_B is negative, we flip to controlling the other
            // half of the X (i.e., (B, -A) instead of (A, -B)).
            // In this case, the angle bisector sign must be flipped.
            if (m_sign_B < 0) m_source_t *= -1.0;
            m_source_normal *= -1.0;
            m_update();
        }

        // Update the deformed points/material frames of the incident rod edges,
        // as stored in the full network's "points" and "thetas" arrays.
        void applyConfiguration(const std::vector<RodSegment>   &rodSegments,
                                std::vector<std::vector<Pt3>>   &networkPoints,
                                std::vector<std::vector<Real_>> &networkThetas,
                                bool spatialCoherence = false) const;

        // Determine the sensitivity of the incident terminal edge vector of segment "si"
        // to changes to eA and eB. This is represented by the pair of scalars
        // (s_jA, s_jB) where, e.g., s_jA is:
        //      0 if edge vector A doesn't control terminal edge j,
        //      1 if edge vector A gives terminal edge j's vector directly,
        //     -1 if edge vector A gives the negative of terminal edge j's vector
        std::tuple<double, double, bool> terminalEdgeIdentification(size_t si) const {
            // Note: m_e_A points from segment A0 to segment A1.
            double s_jA = 0, s_jB = 0;
            bool isStart = false;
            if (m_segmentsA[0] == si) { isStart = m_isStartA[0]; s_jA = isStart ? -1.0 :  1.0; } // m_e_A points out of segment A0
            if (m_segmentsA[1] == si) { isStart = m_isStartA[1]; s_jA = isStart ?  1.0 : -1.0; } // m_e_A points into   segment A1
            if (m_segmentsB[0] == si) { isStart = m_isStartB[0]; s_jB = isStart ? -1.0 :  1.0; } // m_e_B points out of segment B0
            if (m_segmentsB[1] == si) { isStart = m_isStartB[1]; s_jB = isStart ?  1.0 : -1.0; } // m_e_B points into   segment B1

            return std::make_tuple(s_jA, s_jB, isStart);
        }

        const Vec3 &  pos() const { return m_pos; }
        const Vec3 &omega() const { return m_omega; }
        Real_       alpha() const { return m_alpha; }
        Real_       len_A() const { return m_len_A; }
        Real_       len_B() const { return m_len_B; }

        const Vec3 &   e_A() const { return m_e_A; }
        const Vec3 &   e_B() const { return m_e_B; }
        const Vec3 &normal() const { return m_normal; }

        // Rod A,B's edge tangents (before rotation by omega)
        Vec3 source_t_A() const { return            ropt::rotated_vector((-0.5 * m_alpha) * m_source_normal, m_source_t); }
        Vec3 source_t_B() const { return m_sign_B * ropt::rotated_vector(( 0.5 * m_alpha) * m_source_normal, m_source_t); }
        const Vec3 &source_normal() const { return m_source_normal; }

        void set_pos  (const Vec3 &v)     { m_pos = v; }
        void set_omega(const Vec3 &omega) { m_omega = omega; m_update(); }
        void set_alpha(Real_ alpha)       { m_alpha = alpha; m_update(); }
        void set_len_A(Real_ l)           { m_len_A = l; }
        void set_len_B(Real_ l)           { m_len_B = l; }

        // Change the rotation parametrization to be the tangent space of SO(3) at at the current rotation.
        // Note: this changes omega (and consequently the linkage DoF values).
        void updateParametrization() {
            m_source_t      = ropt::rotated_vector(m_omega, m_source_t);
            m_source_normal = ropt::rotated_vector(m_omega, m_source_normal);
            m_omega.setZero();
        }

        // Use with care!
        void updateLinkagePointer(RodLinkage_T *ptr) { m_linkage = ptr; }

        // Sadly cannot be deduced from getState()'s return (with auto)
        using SerializedState = std::tuple<Pt3, Vec3, Real_, Real_, Real_, Real_, Vec3, Vec3,
                                           std::array<size_t, 2>, std::array<size_t, 2>, std::array<bool, 2>, std::array<bool, 2>>;
        // Construct from state output by getState
        Joint(const SerializedState &state) : m_linkage(nullptr) { m_setState<Real_>(state); }
        // Get full state of this Joint, e.g. for serialization
        SerializedState getState() const {
            return std::make_tuple(m_pos, m_omega, m_alpha, m_len_A, m_len_B, m_sign_B, m_source_t, m_source_normal,
                                   m_segmentsA, m_segmentsB, m_isStartA, m_isStartB);
        }

        // Get the two distinct rest lengths for the edges this joint controls (one for the rods of segment A, one for B)
        Vec2 getRestLengths() const {
            assert(m_linkage);
            auto getLen = [&](size_t sidx, bool isStart) {
                const auto &r = m_linkage->segment(sidx).rod;
                return r.restLengthForEdge(isStart ? 0 : (r.numEdges() - 1));
            };

            Vec2 result(getLen(m_segmentsA[0], m_isStartA[0]),
                        getLen(m_segmentsB[0], m_isStartB[0]));

            if ((m_segmentsA[1] != NONE) && (getLen(m_segmentsA[1], m_isStartA[1]) != result[0])) throw std::runtime_error("Segment A rest length mismatch at joint");
            if ((m_segmentsB[1] != NONE) && (getLen(m_segmentsB[1], m_isStartB[1]) != result[1])) throw std::runtime_error("Segment B rest length mismatch at joint");

            return result;
        }

        // Set the two distinct rest lengths for the edges this joint controls (one for the rods of segment A, one for B)
        void setRestLengths(const Vec2 &rlens) {
            assert(m_linkage);
            auto setLen = [&](size_t sidx, bool isStart, Real_ val) {
                if (sidx == NONE) return;
                auto &r = m_linkage->segment(sidx).rod;
                r.restLengthForEdge(isStart ? 0 : (r.numEdges() - 1)) = val;
            };

            for (size_t i = 0; i < 2; ++i) {
                setLen(m_segmentsA[i], m_isStartA[i], rlens[0]);
                setLen(m_segmentsB[i], m_isStartB[i], rlens[1]);
            }
        }

        // Figure out whether segment "si" is part of rod A or rod B at this joint.
        // 0 ==> A, 1 ==> B, NONE ==> si not incident this joint.
        size_t segmentABOffset(size_t si) const {
            if ((si == m_segmentsA[0]) || (si == m_segmentsA[1])) return 0;
            if ((si == m_segmentsB[0]) || (si == m_segmentsB[1])) return 1;
            return NONE;
        }

        const std::array<size_t, 2> &segmentsA() const { return m_segmentsA; }
        const std::array<size_t, 2> &segmentsB() const { return m_segmentsB; }

        const std::array<bool, 2> &isStartA() const { return m_isStartA; }
        const std::array<bool, 2> &isStartB() const { return m_isStartB; }

        // The index of the segment that segment "si" connects with
        // at this joint.
        size_t continuationSegment(size_t si) const {
            if (si == m_segmentsA[0]) return m_segmentsA[1];
            if (si == m_segmentsA[1]) return m_segmentsA[0];
            if (si == m_segmentsB[0]) return m_segmentsB[1];
            if (si == m_segmentsB[1]) return m_segmentsB[0];
            throw std::runtime_error("Segment " + std::to_string(si) + " is not incident");
        }

        // Segment enumeration (for looping over all incident segments):
        // Gets the segment corresponding to local index "lsi" in {0, 1, 2, 3}.
        // 0, 1 correspond to rod A; 2, 3 to rod B.
        size_t segment(size_t lsi) const {
            assert(lsi < 4);
            if (lsi < 2) { return m_segmentsA[lsi]; }
            return m_segmentsB[lsi - 2];
        }

        // Call "visitor(idx)" for each global independent vertex/theta of
        // freedom index "idx" influenced by the joint's variable "var"
        // (i.e. that appears in the joint var's column of the Hessian).
        // Note: global degrees of freedom are *not* visited in order.
        // restLenVar: whether "var" selects an ordinary joint variable or
        // a joint rest length variable.
        template<class F>
        void visitInfluencedSegmentVars(const size_t var, F &visitor, bool restLenVar = false) const;

    private:
        // Pointer to the linkage containing this joint; needed for accessing
        // the segments controlled by this joint.
        // This is a pointer rather than a reference to enable the RodLinkage's
        // copy construction (though care should be taken when copying joints).
        RodLinkage_T *m_linkage;

        // Joint parameters:
        Pt3   m_pos;             // Position of edges' common midpoint
        Vec3  m_omega;           // The axis-angle representation of the joint's rotation (in tangent space of SO(3))
        Real_ m_alpha;           // The opening angle of the joint
        Real_ m_len_A, m_len_B;  // The length of each incident edge

        // Derived quantities:
        Vec3 m_e_A, m_e_B;  // The incident edge vector for rods A/B, computed from the parameters above.
                            // Orientation is chosen so that m_e_A points from segment A0 to segment A1
                            //                               m_e_B points from segment B0 to segment B1
        Vec3 m_normal;      // The joint normal, computed from the "source" normal and the joint rotation.

        // The "reference" rotation around which the joint's rotation is
        // parametrized is given by
        // (m_source_t | m_source_normal x m_source_t | m_source_normal).
        // The current tangent and normal are the rotation of m_source_t and
        // m_source_normal by the rotation described by axis/angle m_omega.
        Vec3 m_source_t, m_source_normal;
        Real_ m_sign_B;          // The sign of rod B's edge vector; chosen at construction time to define alpha.

        // Connectivity information
        std::array<size_t, 2> m_segmentsA, m_segmentsB; // The two segments that comprise each rod passing through this joint. (second entries can be "NONE")
        std::array<bool,   2> m_isStartA,  m_isStartB;  // Whether this joint is at the start (or end) of the incident rod segments.
                                                        // (This depends on the input graph's edge orientation, and 0, 1, or 2 segments
                                                        //  of a rod could originate at this joint.)

        // Set the cached state, e.g., for serialization (use with care!)
        template<typename Real2_>
        void m_setState(const typename RodLinkage_T<Real2_>::Joint::SerializedState &state) {
            m_pos           = std::get< 0>(state);
            m_omega         = std::get< 1>(state);
            m_alpha         = std::get< 2>(state);
            m_len_A         = std::get< 3>(state);
            m_len_B         = std::get< 4>(state);
            m_sign_B        = std::get< 5>(state);
            m_source_t      = std::get< 6>(state);
            m_source_normal = std::get< 7>(state);
            m_segmentsA     = std::get< 8>(state);
            m_segmentsB     = std::get< 9>(state);
            m_isStartA      = std::get<10>(state);
            m_isStartB      = std::get<11>(state);
            m_update();
        }

        // Update cached edge vectors/normals; to be called whenever the parameters change.
        void m_update();
    };

    // RodSegment parameters are the underlying elastic rod's centerline
    // positions and material frame angles (thetas). The first and last edge of
    // the rod only have degrees of freedom if that rod end is not part of a
    // joint.
    struct RodSegment {
        size_t startJoint = NONE, endJoint = NONE;
        Rod rod;

        // Converting constructor from a different floating point type.
        // Unfortunately, we can't just template this on the floating point type
        // of the surrounding RodLinkage_T struct, since then the compiler isn't
        // able to deduce the template parameter...
        template<typename RodSegment2>
        RodSegment(const RodSegment2 &s)
            : startJoint(s.startJoint), endJoint(s.endJoint), rod(s.rod) { }

        // Construct a rod segment with endpoint pivots at startPt and endPt.
        // Because the pivots occur at an edge midpoint, the rod will extend
        // half an edgelength past the start and end points.
        // TODO: change nsubdiv to an edge length parameter!
        RodSegment(const Pt3 &startPt, const Pt3 &endPt, size_t nsubdiv);

        RodSegment(size_t startJoint, size_t endJoint, Rod &&rod) :
            startJoint(startJoint), endJoint(endJoint), rod(std::move(rod)) { }

        // Read the rod parameters from a collection of global variables (DoFs),
        // storing them in "points" and "thetas" arrays. Only the entries of "points"
        // and "thetas" that are controlled by this rod segment's parameters are
        // altered (terminal edge quantities controlled by a joint are left unchanged).
        template<class Derived>
        void unpackParameters(const Eigen::DenseBase<Derived> &vars,
                              std::vector<Pt3>   &points,
                              std::vector<Real_> &thetas) const;

        // Set the rod parameters from a collection of global variables (DoFs)
        template<class Derived>
        void setParameters(const Eigen::DenseBase<Derived> &vars);

        // Extract the rod parameters, storing them in a collection of global variables (DoFs)
        template<class Derived>
        void getParameters(Eigen::DenseBase<Derived> &vars) const;

        bool hasStartJoint() const { return startJoint != NONE; }
        bool hasEndJoint()   const { return   endJoint != NONE; }
        size_t numJoints()   const { return hasStartJoint() + hasEndJoint(); }
        // Access start/end joint indices by label 0/1.
        size_t joint(size_t i) const { if (i == 0) return startJoint;
                                       if (i == 1) return endJoint;
                                       throw std::runtime_error("Out of bounds.");
        }

        // Determine the number of degrees of freedom belonging to this segment's rod
        // (after joint constraints have been applied).
        size_t numDoF() const {
            // Joints (if present) each determine the position of two vertices
            // and the material axis of one edge.
            return rod.numDoF() - numJoints() * (2 * 3 + 1);
        }

        // Number of internal/free end vertices and edges
        size_t numFreeVertices() const { return rod.numVertices() - 2 * (numJoints()); }
        size_t numFreeEdges()    const { return rod.numEdges()    -      numJoints() ; }

        // Determine the number of positional degrees of freedom belonging to this
        // segment's rod after joint constraints have been applied. This effectively
        // gives us the offset into the segment's reduced DoFs of the first
        // material frame variable.
        size_t numPosDoF() const { return 3 * (rod.numVertices() - 2 * hasStartJoint() - 2 * hasEndJoint()); }

        // Number of degrees of freedom in the underlying rod (i.e. before
        // joint constraints have been applied).
        size_t fullDoF() const { return rod.numDoF(); }
        // This is actually pretty good.

        // Update the unconstrained thetas (internal edges + free ends) to
        // minimize twisting energy
        void setMinimalTwistThetas(bool verbose = false);
    };

    ////////////////////////////////////////////////////////////////////////////
    // Expose TerminalEdgeSensitivity for debugging
    ////////////////////////////////////////////////////////////////////////////
    enum class TerminalEdge : int { Start = 0, End = 1 };
    const LinkageTerminalEdgeSensitivity<Real_> &getTerminalEdgeSensitivity(size_t si, TerminalEdge which, bool updatedSource, bool evalHessian);

private:
    std::vector<Joint> m_joints;
    std::vector<RodSegment> m_segments;

    // Material used to initialize rods (useful if they are recreated by
    // RodLinkage::set after the linkage's material has been configured).
    RodMaterial m_material;

    // Offset in the full list of linkage DoFs of the DoFs for each segment/joint
    std::vector<size_t> m_dofOffsetForSegment,
                        m_dofOffsetForJoint,
                        m_restLenDoFOffsetForSegment,
                        m_restLenDoFOffsetForJoint;

    Real_ m_initMinRestLen = 0;

    void m_buildDoFOffsets();

    CSCMat m_segmentRestLenToEdgeRestLenMapTranspose;
    void m_constructSegmentRestLenToEdgeRestLenMapTranspose(const VecX &segmentRestLenGuess);
    void m_setRestLengthsFromPSRL();
    VecX m_perSegmentRestLen; // cached for ease so we don't have to reconstruct from the linkage's per-edge rest length.

    // Cache to avoid memory allocation in setDoFs
    std::vector<std::vector<Pt3  >> m_networkPoints;
    std::vector<std::vector<Real_>> m_networkThetas;

    struct SensitivityCache {
        SensitivityCache();

        // Cache of constrained terminal edges' Jacobians and Hessians
        // (to accelerate repeated calls to elastic energy Hessian/gradient).
        // The entries for segment si's two ends are at
        // sensitivityForTerminalEdge[2 * si + 0] and sensitivityForTerminalEdge[2 * si + 1]
        // (entries for free ends are left uninitialized).
        std::vector<LinkageTerminalEdgeSensitivity<Real_>> sensitivityForTerminalEdge;

        bool evaluatedWithUpdatedSource = true;
        bool evaluatedHessian = false;
        void update(const RodLinkage_T &l, bool updatedSource, bool evalHessian);
        // Compute directional derivative of Jacobian ("delta_jacobian") instead of the full Hessian
        void update(const RodLinkage_T &l, bool updatedSource, const VecX &delta_params);
        const LinkageTerminalEdgeSensitivity<Real_> &lookup(size_t si, TerminalEdge which) const {
            return sensitivityForTerminalEdge.at(2 * si + static_cast<int>(which));
        }
        bool filled() const { return !sensitivityForTerminalEdge.empty(); }

        void clear();
        ~SensitivityCache();
    };
    mutable SensitivityCache m_sensitivityCache;

    ////////////////////////////////////////////////////////////////////////////
    // Cache for hessian sparsity patterns
    ////////////////////////////////////////////////////////////////////////////
    mutable std::unique_ptr<CSCMat> m_cachedHessianSparsity, m_cachedHessianVarRLSparsity, m_cachedHessianPSRLSparsity;
    void m_clearCache() { m_cachedHessianSparsity.reset(), m_cachedHessianVarRLSparsity.reset(), m_cachedHessianPSRLSparsity.reset(); }
};

#endif /* end of include guard: RODLINKAGE_HH */
