#include "ElasticRod.hh"
#include <stdexcept>
#include <cmath>
#include <MeshFEM/GlobalBenchmark.hh>
#include "VectorOperations.hh"
#include "SparseMatrixOps.hh"
#include <MeshFEM/unused.hh>
#include "AutomaticDifferentiation.hh"

////////////////////////////////////////////////////////////////////////////////
// Geometric operations
////////////////////////////////////////////////////////////////////////////////
// Compute the unit tangents (normalized edge vectors) for the chain of edges
// connecting "pts"
template<typename Real_>
void unitTangents(const std::vector<Pt3_T<Real_>> &pts, std::vector<Vec3_T<Real_>> &result) {
    const size_t ne = pts.size() - 1;
    result.clear();
    result.reserve(ne);
    for (size_t j = 0; j < ne; ++j) {
        result.emplace_back(pts[j + 1] - pts[j]);
        result.back().normalize();
    }
}

template<typename Real_>
std::vector<Vec3_T<Real_>> unitTangents(const std::vector<Pt3_T<Real_>> &pts) {
    std::vector<Vec3_T<Real_>> result;
    unitTangents(pts, result);
    return result;
}

////////////////////////////////////////////////////////////////////////////////
// Elastic Rod Accessors
////////////////////////////////////////////////////////////////////////////////
template<typename Real_>
void ElasticRod_T<Real_>::setRestConfiguration(const std::vector<Pt3_T<Real_>> &points) {
    if (points.size() < 3) throw std::runtime_error("Must have at least three points (two edges)");

    m_restPoints = points;

    const size_t nv = points.size();
    const size_t ne = nv - 1;

    auto unit_tangents = unitTangents(m_restPoints);

    std::vector<Vec3> rest_d1;
    rest_d1.reserve(ne);

    // Choose first rest reference director arbitrarily.
    rest_d1.emplace_back(getPerpendicularVector(unit_tangents[0]));

    // Parallel transport this reference vector to the remaining rod edges
    for (size_t i = 1; i < ne; ++i) {
        rest_d1.emplace_back(
            parallelTransportNormalized(unit_tangents[i - 1], unit_tangents[i],
                                        rest_d1.back()));
    }

    // Compute the orthogonal reference director (d2) for each edge
    m_restDirectors.clear();
    m_restDirectors.reserve(ne);
    for (size_t j = 0; j < ne; ++j) {
        m_restDirectors.emplace_back(rest_d1[j],
                                     unit_tangents[j].cross(rest_d1[j]));
    }

    m_restLen.clear();
    m_restLen.reserve(ne);
    for (size_t j = 0; j < ne; ++j)
        m_restLen.push_back((m_restPoints[j + 1] - m_restPoints[j]).norm());

    // Compute rest curvature (in material frame)
    m_restKappa.assign(nv, Vec2::Zero());
    for (size_t i = 1; i < nv - 1; ++i) {
        auto kb = curvatureBinormal(unit_tangents[i - 1], unit_tangents[i]);
        m_restKappa[i] = Vec2(0.5 * kb.dot(m_restDirectors[i - 1].d2 + m_restDirectors[i].d2),
                             -0.5 * kb.dot(m_restDirectors[i - 1].d1 + m_restDirectors[i].d1));
    }

    // For now, assume the rod is untwisted in its rest configuration
    m_restTwist.assign(nv, 0);

    m_initMinRestLen = minRestLength();

    m_deformedStates.resize(1);
    deformedConfiguration().initialize(*this);
}

template<typename Real_>
void ElasticRod_T<Real_>::setDeformedConfiguration(const std::vector<Pt3_T<Real_>> &points, const std::vector<Real_> &thetas) {
    if (points.size() != numVertices()) throw std::logic_error("Invalid number of points");
    if (thetas.size() != numEdges())    throw std::logic_error("Invalid number of material frame rotations");

    deformedConfiguration().update(points, thetas);
}

template<typename Real_>
VecX_T<Real_> ElasticRod_T<Real_>::getDoFs() const {
    VecX dofs(numDoF());
    const size_t nv = numVertices(), ne = numEdges();

    const auto &points = deformedConfiguration().points();
    const auto &thetas = deformedConfiguration().thetas();

    for (size_t i = 0; i < nv; ++i) dofs.template segment<3>(3 * i) = points[i];
    for (size_t j = 0; j < ne; ++j) dofs[3 * nv + j]       = thetas[j];

    return dofs;
}

template<typename Real_>
void ElasticRod_T<Real_>::setDoFs(const Eigen::Ref<const VecX_T<Real_>> &dofs) {
    if (size_t(dofs.size()) != numDoF()) throw std::runtime_error("DoF vector has incorrect length.");
    const size_t nv = numVertices(), ne = numEdges();

    std::vector<Vec3>  points(nv);
    std::vector<Real_> thetas(ne);

    for (size_t i = 0; i < nv; ++i) points[i] = dofs.template segment<3>(3 * i);
    for (size_t j = 0; j < ne; ++j) thetas[j] = dofs[3 * nv + j];

    deformedConfiguration().update(points, thetas);
}

template<typename Real_>
void ElasticRod_T<Real_>::DeformedState::initialize(const ElasticRod_T &rod) {
    // Initialize source reference directors and tangents since they'll be
    // needed for parallel transport in the update method
    sourceReferenceDirectors = rod.m_restDirectors;
    unitTangents(rod.m_restPoints, sourceTangent);
    sourceTheta.assign(rod.numEdges(), 0);
    sourceReferenceTwist.assign(rod.numVertices(), 0);

    update(rod.m_restPoints, sourceTheta);
}

// Constructor for cached deformed quantities.
template<typename Real_>
void ElasticRod_T<Real_>::DeformedState::update(const std::vector<Pt3_T<Real_>> &points, const std::vector<Real_> &thetas) {
    const size_t nv = points.size(),
                 ne = thetas.size();
    m_point = points;
    m_theta = thetas;

    // Compute edge lengths and new unit tangents, parallel transporting the
    // material frame to the new tangent.
    len.resize(ne);
    tangent.resize(ne);
    referenceDirectors.clear(); referenceDirectors.reserve(ne);
    for (size_t j = 0; j < ne; ++j) {
        tangent[j] = (points[j + 1] - points[j]);
        len[j] = tangent[j].norm();
        tangent[j] /= len[j];

        referenceDirectors.emplace_back(parallelTransportNormalized(sourceTangent[j], tangent[j], sourceReferenceDirectors[j].d1),
                                        parallelTransportNormalized(sourceTangent[j], tangent[j], sourceReferenceDirectors[j].d2));
    }

    // Compute twist of the transported reference directors from one edge to the next
    referenceTwist.resize(nv);
    referenceTwist.front() = referenceTwist.back() = 0;
    for (size_t i = 1; i < nv - 1; ++i) {
        // Finite rotation angle needed to take the parallel transported copy
        // of the previous edge's reference director to the current edge's
        // reference director.
        Vec3 prevDirectorTransported = parallelTransportNormalized(tangent[i - 1], tangent[i], referenceDirectors[i - 1].d1);
        referenceTwist[i] = angle(tangent[i], prevDirectorTransported, referenceDirectors[i].d1);

        // Temporal coherence: to avoid jumps in the material frame twist of 2 pi,
        // choose the 2 Pi offset to minimize change from source reference twist.
        // (so the new reference twist is always in the interval [source - pi, source + pi]
        Real_ diff = (sourceReferenceTwist[i] - referenceTwist[i]) / (2 * M_PI);
        referenceTwist[i] += 2 * M_PI * std::round(stripAutoDiff(diff)); // TODO: strip autodiff?
    }

    // Compute material frame using reference directors and thetas.
    // Also update the "source material frame" based on the new thetas.
    materialFrame.clear(), materialFrame.reserve(ne);
    sourceMaterialFrame.clear(), sourceMaterialFrame.reserve(ne);
    for (size_t j = 0; j < ne; ++j) {
        Real_ cosTheta = cos(thetas[j]),
              sinTheta = sin(thetas[j]);
        materialFrame.emplace_back(cosTheta * referenceDirectors[j].d1 + sinTheta * referenceDirectors[j].d2,
                                  -sinTheta * referenceDirectors[j].d1 + cosTheta * referenceDirectors[j].d2);
        sourceMaterialFrame.emplace_back(cosTheta * sourceReferenceDirectors[j].d1 + sinTheta * sourceReferenceDirectors[j].d2,
                                        -sinTheta * sourceReferenceDirectors[j].d1 + cosTheta * sourceReferenceDirectors[j].d2);
    }

    // Compute curvature binormals
    kb.resize(nv);
    kb.front() = kb.back() = Vec3::Zero();
    for (size_t i = 1; i < nv - 1; ++i)
        kb[i] = curvatureBinormal(tangent[i - 1], tangent[i]);

    // Compute curvature normal in material coordinate system.
    kappa.resize(nv);
    per_corner_kappa.resize(nv);
    kappa.front() = kappa.back() = Vec2::Zero();
    for (size_t i = 1; i < nv - 1; ++i) {
        per_corner_kappa[i](0, 0) =  kb[i].dot(materialFrame[i - 1].d2);
        per_corner_kappa[i](0, 1) =  kb[i].dot(materialFrame[i    ].d2);
        per_corner_kappa[i](1, 0) = -kb[i].dot(materialFrame[i - 1].d1);
        per_corner_kappa[i](1, 1) = -kb[i].dot(materialFrame[i    ].d1);

        kappa[i] = 0.5 * per_corner_kappa[i].rowwise().sum();
    }
}

template<typename Real_>
void ElasticRod_T<Real_>::DeformedState::setReferenceTwist(Real_ newTwist) {
    // Compute twist of the transported reference directors from one edge to the next
    const size_t nv = m_point.size();
    Real_ referenceRotation = 0;
    for (size_t i = 1; i < nv - 1; ++i) {
        Vec3 prevD1Transported = parallelTransportNormalized(tangent[i - 1], tangent[i], referenceDirectors[i - 1].d1),
             prevD2Transported = parallelTransportNormalized(tangent[i - 1], tangent[i], referenceDirectors[i - 1].d2);

        referenceDirectors[i].d1 = rotatedVectorAngle(tangent[i], newTwist, prevD1Transported);
        referenceDirectors[i].d2 = rotatedVectorAngle(tangent[i], newTwist, prevD2Transported);

        // This modification should preserve the twisting strain, meaning:
        //   theta(i) - theta(i - 1) + newTwist == thetaOld(i) - thetaOld(i - 1) + referenceTwistOld
        //   theta(i) = thetaOld(i) - thetaOld(i - 1) + theta(i - 1) + referenceTwistOld - newTwist
        //   ==> thetaChange = prevThetaChange + referenceTwistOld - newTwist

        // Subtract off the accumulated rotation of reference frame i to preserve the material frame.
        referenceRotation += newTwist - referenceTwist[i];
        m_theta[i]        -= referenceRotation;
        referenceTwist[i]  = newTwist;
    }
}

// Determine the material frame vector D2 for edge "j" that corresponds to
// angle "theta" after the edge has been transformed to have the new edge vector eNew
// (i.e., after the reference directors have been updated with parallel transport).
template<typename Real_>
Vec3_T<Real_> ElasticRod_T<Real_>::materialFrameD2ForTheta(Real_ theta, const Vec3_T<Real_> &eNew, size_t j) const {
    // Determine parallel-transported directors (from source tangent vector to eNew)
    const auto &dc = deformedConfiguration();
    Vec3 refd1 = parallelTransportNormalized(dc.sourceTangent[j], eNew.normalized(), dc.sourceReferenceDirectors[j].d1);
    Vec3 refd2 = parallelTransportNormalized(dc.sourceTangent[j], eNew.normalized(), dc.sourceReferenceDirectors[j].d2);
    // std::cout << "sourceTangent: " << dc.sourceTangent[j].transpose() << std::endl;
    // std::cout << "refd1: " << refd1.transpose() << std::endl;
    // std::cout << "refd2: " << refd2.transpose() << std::endl;

    Real_ cosTheta = cos(theta),
          sinTheta = sin(theta);
    return cosTheta * refd2 - sinTheta * refd1;
}

// Determine frame rotation angle "theta" for edge "j" from material frame vector "d2".
// (This is the rotation that takes the first reference director to "d2".
// We remove the integer-multiple-of-2Pi ambiguity in one of two ways:
//  if spatialCoherence == true, we chose the angle that minimizes twisting energy for one of the incident vertices.
//  if spatialCoherence == false, we opt for temporal coherence, minimizing the change in angle from the source configuration.
//  To remove the integer-multiple-of-2Pi ambiguity, we choose the angle that minimizes twisting energy for one
//  of the incident vertices).
template<typename Real_>
Real_ ElasticRod_T<Real_>::thetaForMaterialFrameD2(Vec3_T<Real_> d2, const Vec3_T<Real_> &eNew, size_t j, bool spatialCoherence) const {
    const auto &dc = deformedConfiguration();
    Real_ parallelComp = d2.dot(eNew);
    if (std::abs(stripAutoDiff(parallelComp)) > 1e-8) {
        std::cerr << "WARNING: non-perpendicular normal" << std::endl;
        d2 -= (parallelComp / eNew.squaredNorm()) * eNew;
        d2.normalize();
    }

    // Determine parallel-transported directors (from source tangent vector to eNew)
    Vec3 tNew = eNew.normalized();
    Vec3 refd1 = parallelTransportNormalized(dc.sourceTangent[j], tNew, dc.sourceReferenceDirectors[j].d1);
    Vec3 refd2 = parallelTransportNormalized(dc.sourceTangent[j], tNew, dc.sourceReferenceDirectors[j].d2);

    Real_ cosTheta =  d2.dot(refd2);
    Real_ sinTheta = -d2.dot(refd1);
    Real_ theta = atan2(sinTheta, cosTheta);

    if (spatialCoherence) {
        // Choose 2 Pi offset for theta to minimize twisting energy
        // Note: dc.referenceTwist[j] was computed for the old reference directors, not the new ones parallel-transported
        // onto eNew. However, this shouldn't be an issue as long as eNew is not too different from the old tangent.
        Real_ twistDeviation;
        if (j > 0) twistDeviation = theta - dc.theta(j - 1) + dc.referenceTwist[j] - m_restTwist[j];
        else       twistDeviation = theta - dc.theta(    1) - dc.referenceTwist[1] + m_restTwist[1];


        // This probably could be implemented with an fmod...
        while (twistDeviation >  M_PI) { theta -= 2 * M_PI; twistDeviation -= 2 * M_PI; }
        while (twistDeviation < -M_PI) { theta += 2 * M_PI; twistDeviation += 2 * M_PI; }
    }
    else {
        // Temporal coherence: choose 2 Pi offset to minimize change from previous theta.
        Real_ diff = (dc.sourceTheta[j] - theta) / (2 * M_PI);
        theta += 2 * M_PI * std::round(stripAutoDiff(diff));
    }

    if (std::isnan(stripAutoDiff(theta))) {
        std::cerr << "NaN theta encountered" << std::endl;
        std::cerr << "\td2: " << d2.transpose() << std::endl;
        std::cerr << "\trefd1: " << refd1.transpose() << std::endl;
        std::cerr << "\trefd2: " << refd2.transpose() << std::endl;
        std::cerr << "\ttNew: " << tNew.transpose() << std::endl;
        std::cerr << "\tsourceTangent: " << dc.sourceTangent[j].transpose() << std::endl;
        std::cerr << "\tsourceReferenceDirectors: " << dc.sourceReferenceDirectors[j].d1.transpose() << ", " << dc.sourceReferenceDirectors[j].d1.transpose() << std::endl;
        std::cerr << "\tsourceTheta: " << dc.sourceTheta[j] << std::endl;
    }

    return theta;
}

////////////////////////////////////////////////////////////////////////////////
// Elastic energy
////////////////////////////////////////////////////////////////////////////////
template<typename Real_>
Real_ ElasticRod_T<Real_>::energyStretch() const {
    const size_t ne = numEdges();
    Real_ result = 0;
    for (size_t j = 0; j < ne; ++j) {
        Real_ strainj = deformedConfiguration().len[j] / m_restLen[j] - 1.0;
        result += density(j) * m_stretchingStiffness[j] * strainj * strainj * m_restLen[j];
    }
    return 0.5 * result;
}

template<typename Real_>
Real_ ElasticRod_T<Real_>::energyBend() const {
    const size_t nv = numVertices();
    const auto &dc = deformedConfiguration();
    Real_ result = 0;

    if (m_bendingEnergyType == BendingEnergyType::Bergou2010) {
        for (size_t i = 1; i < nv - 1; ++i) {
            Real_ libar2 = m_restLen[i - 1] + m_restLen[i];

            Vec2 kappaDiff = dc.kappa[i] - m_restKappa[i];
            Real_ contrib = m_bendingStiffness[i].lambda_1 * kappaDiff[0] * kappaDiff[0]
                          + m_bendingStiffness[i].lambda_2 * kappaDiff[1] * kappaDiff[1];

            result += contrib / libar2;
        }
    }
    else if (m_bendingEnergyType == BendingEnergyType::Bergou2008) {
        for (size_t i = 1; i < nv - 1; ++i) {
            Real_ inv_libar2 = 1.0 / (m_restLen[i - 1] + m_restLen[i]);

            for (size_t adj_edge = 0; adj_edge < 2; ++adj_edge) {
                Vec2 kappaDiff = dc.per_corner_kappa[i].col(adj_edge) - m_restKappa[i];

                Real_ contrib = m_bendingStiffness[i].lambda_1 * kappaDiff[0] * kappaDiff[0]
                              + m_bendingStiffness[i].lambda_2 * kappaDiff[1] * kappaDiff[1];

                result += contrib * m_restLen[(i - 1) + adj_edge] * inv_libar2 * inv_libar2;
            }
        }
    }
    else { assert(false); }

    return result;
}

template<typename Real_>
Real_ ElasticRod_T<Real_>::energyTwist() const {
    const size_t nv = numVertices();
    const auto &dc = deformedConfiguration();
    Real_ result = 0;
    for (size_t i = 1; i < nv - 1; ++i) {
        Real_ libar2 = m_restLen[i - 1] + m_restLen[i];
        Real_ twistDeviation = dc.theta(i) - dc.theta(i - 1) + dc.referenceTwist[i] - m_restTwist[i];
        result += (m_twistingStiffness[i] / libar2) * twistDeviation * twistDeviation;
    }
    return result;
}

template<typename Real_>
Real_ ElasticRod_T<Real_>::energy() const { return energyStretch() + energyBend() + energyTwist(); }

// Per-vertex bending energy for visualization
template<typename Real_>
VecX_T<Real_> ElasticRod_T<Real_>::energyBendPerVertex() const {
    const size_t nv = numVertices();
    const auto &dc = deformedConfiguration();
    VecX_T<Real_> result(nv);
    result[0] = result[nv - 1] = 0.0;

    if (m_bendingEnergyType == BendingEnergyType::Bergou2010) {
        for (size_t i = 1; i < nv - 1; ++i) {
            Real_ libar2 = m_restLen[i - 1] + m_restLen[i];

            Vec2 kappaDiff = dc.kappa[i] - m_restKappa[i];
            Real_ contrib = m_bendingStiffness[i].lambda_1 * kappaDiff[0] * kappaDiff[0]
                          + m_bendingStiffness[i].lambda_2 * kappaDiff[1] * kappaDiff[1];

            result[i] = contrib / libar2;
        }
    }
    else if (m_bendingEnergyType == BendingEnergyType::Bergou2008) {
        for (size_t i = 1; i < nv - 1; ++i) {
            Real_ inv_libar2 = 1.0 / (m_restLen[i - 1] + m_restLen[i]);

            for (size_t adj_edge = 0; adj_edge < 2; ++adj_edge) {
                Vec2 kappaDiff = dc.per_corner_kappa[i].col(adj_edge) - m_restKappa[i];

                Real_ contrib = m_bendingStiffness[i].lambda_1 * kappaDiff[0] * kappaDiff[0]
                              + m_bendingStiffness[i].lambda_2 * kappaDiff[1] * kappaDiff[1];

                result[i] = contrib * m_restLen[(i - 1) + adj_edge] * inv_libar2 * inv_libar2;
            }
        }
    }
    else { assert(false); }

    return result;
}

////////////////////////////////////////////////////////////////////////////////
// Elastic energy gradients
////////////////////////////////////////////////////////////////////////////////
template<typename Real_>
template<typename StencilMask>
typename ElasticRod_T<Real_>::Gradient ElasticRod_T<Real_>::gradEnergyStretch(bool variableRestLen, bool restlenOnly, const StencilMask &sm)  const {
    // d 0.5 sum_j ks^j (l^j / rl^j - 1)^2 rl^j
    //     = sum_j ks^j (l^j / rl^j - 1) * dl^j
    // d l^j = +/- tangent if x is +/- endpoint
    const size_t ne = numEdges();
    Gradient result(*this, variableRestLen);
    // Stretching energy independent of twist: gradTheta = 0

    const auto &dc = deformedConfiguration();
    for (size_t j = 0; j < ne; ++j) {
        if (!sm.includeEdgeStencil(ne, j)) continue;
        Real_ fracLen = dc.len[j] / m_restLen[j];
        Real_ coeff = density(j) * m_stretchingStiffness[j] * (fracLen - 1.0);
        if (!restlenOnly) {
            result.gradPos(j    ) -= coeff * dc.tangent[j];
            result.gradPos(j + 1) += coeff * dc.tangent[j];
        }

        if (variableRestLen)
            result.gradRestLen(j) = density(j) * m_stretchingStiffness[j] * 0.5 * (1.0 - fracLen * fracLen);
    }

    return result;
}

// grad 0.5 sum_i 1/libar_i * (B11_i * (kappa1_i - kappabar1_i)^2 + B22_i * (kappa2_i - kappabar2_i)^2)
template<typename Real_>
template<class StencilMask>
typename ElasticRod_T<Real_>::Gradient ElasticRod_T<Real_>::gradEnergyBend(bool updatedSource, bool variableRestLen, bool restlenOnly, const StencilMask &sm) const {
    const size_t nv = numVertices();
    Gradient result(*this, variableRestLen);

    using M32d = Eigen::Matrix<Real_, 3, 2>;

    const auto &dc = deformedConfiguration();
    // Compute gradient of each vertex's contribution to the bending energy.
    for (size_t i = 1; i < nv - 1; ++i) {
        if (!sm.includeVtxStencil(nv, i)) continue;

        Real_ inv_2libar = 1.0 / (m_restLen[i - 1] + m_restLen[i]);

        const auto &ti   = dc.tangent[i],
                   &tim1 = dc.tangent[i - 1],
                   &kb   = dc.kb[i];
        const Real_ inv_chi = 1.0 / (1.0 + tim1.dot(ti)); // 1 / chi in paper
        Vec3 tilde_t = inv_chi * (ti + tim1);

        std::array<Real_, 2> inv_len{{1.0 / dc.len[i - 1],
                                      1.0 / dc.len[i    ]}};
        M32d dE_de(M32d::Zero());

        for (size_t adj_edge = 0; adj_edge < 2; ++adj_edge) { // 0 ==> i - 1, 1 ==> i
            const size_t j = (i - 1) + adj_edge;

            // Derivative of the bending energy with respect to (kappa_k)_i^j
            std::array<Real_, 2> dE_dkappa_k_j;
            if (m_bendingEnergyType == BendingEnergyType::Bergou2010) {
                const Vec2 kappaDiff = dc.kappa[i] - m_restKappa[i];
                if (variableRestLen) {
                    Real_ contrib = m_bendingStiffness[i].lambda_1 * kappaDiff[0] * kappaDiff[0]
                                  + m_bendingStiffness[i].lambda_2 * kappaDiff[1] * kappaDiff[1];
                    result.gradRestLen(j) -= inv_2libar * inv_2libar * contrib;
                    if (restlenOnly) continue;
                }
                dE_dkappa_k_j = {{inv_2libar * m_bendingStiffness[i].lambda_1 * kappaDiff[0],
                                  inv_2libar * m_bendingStiffness[i].lambda_2 * kappaDiff[1]}};
            }
            else if (m_bendingEnergyType == BendingEnergyType::Bergou2008) {
                const Vec2 kappaDiff = dc.per_corner_kappa[i].col(adj_edge) - m_restKappa[i];
                if (variableRestLen) {
                    Real_ contrib = m_bendingStiffness[i].lambda_1 * kappaDiff[0] * kappaDiff[0]
                                  + m_bendingStiffness[i].lambda_2 * kappaDiff[1] * kappaDiff[1];
                    const size_t jother = i - adj_edge; // 0 ==> i, 1 ==> i - 1
                    result.gradRestLen(j     ) += (inv_2libar * inv_2libar * inv_2libar) * (m_restLen[jother] - m_restLen[j]) * contrib;
                    result.gradRestLen(jother) -= (inv_2libar * inv_2libar * inv_2libar) * (                2 * m_restLen[j]) * contrib;
                    if (restlenOnly) continue;
                }
                dE_dkappa_k_j = {{2.0 * m_restLen[j] * inv_2libar * inv_2libar * m_bendingStiffness[i].lambda_1 * kappaDiff[0],
                                  2.0 * m_restLen[j] * inv_2libar * inv_2libar * m_bendingStiffness[i].lambda_2 * kappaDiff[1]}};
            }
            else { assert(false); }

            // Accumulate energy dependence through (kappa_k)_i^j
            // Compute material frame angle (theta) dependence
            result.gradTheta(j) += dE_dkappa_k_j[0] * dc.per_corner_kappa[i](1, adj_edge) - dE_dkappa_k_j[1] * dc.per_corner_kappa[i](0, adj_edge);
            for (size_t k = 0; k < 2; ++k) {
                // Compute centerline position dependence
                // First, compute variation of kappa_k^j with respect to a perturbation of the edge tangents.
                const size_t kother = (k + 1) % 2;
                Real_ sign = (k == 0) ? 1 : -1; // Infinitesimal transport kappa_2^j term is just like kappa_1, except d2 is replaced with -d1.
                M32d d_kappa_k_j_de;
                d_kappa_k_j_de.col(0) = inv_len[0] * (( 2 * sign * inv_chi) *   ti.cross(dc.materialFrame[j].get(kother)) - dc.per_corner_kappa[i].col(adj_edge)[k] * tilde_t);
                d_kappa_k_j_de.col(1) = inv_len[1] * ((-2 * sign * inv_chi) * tim1.cross(dc.materialFrame[j].get(kother)) - dc.per_corner_kappa[i].col(adj_edge)[k] * tilde_t);

                if (!updatedSource) {
                    const auto &ts = dc.sourceTangent[j],
                               &t  = dc.tangent[j];
                    const auto &ds = dc.sourceMaterialFrame[j].get(k);
                    Vec3 kb_cross_ts = kb.cross(ts);
                    Real_ inv_chi_hat = 1.0 / (1.0 + ts.dot(t));
                    Real_ ds_dot_t = ds.dot(t), kb_cross_ts_dot_t = t.dot(kb_cross_ts);
                    Vec3 finite_xport_contrib =
                        inv_chi_hat * (ds_dot_t * kb_cross_ts + ds * kb_cross_ts_dot_t)
                        - (inv_chi_hat * inv_chi_hat * ds_dot_t * kb_cross_ts_dot_t) * ts
                        + ds.cross(kb);

                    // We only have a finite transport term for the derivative with respect to edge "j"
                    d_kappa_k_j_de.col(adj_edge) += inv_len[adj_edge] * (finite_xport_contrib - t * t.dot(finite_xport_contrib));
                }

                dE_de += dE_dkappa_k_j[k] * d_kappa_k_j_de;
            }
        }
        // Second, compute variation of bending energy with respect to the centerline positions
        result.gradPos(i - 1) -= dE_de.col(0);
        result.gradPos(i    ) += dE_de.col(0) - dE_de.col(1);
        result.gradPos(i + 1) += dE_de.col(1);
    }

    return result;
}

// grad 0.5 sum_i 1/libar_i * beta_i (m_i - mbar_i)^2
template<typename Real_>
template<class StencilMask>
typename ElasticRod_T<Real_>::Gradient ElasticRod_T<Real_>::gradEnergyTwist(bool updatedSource, bool variableRestLen, bool restlenOnly, const StencilMask &sm) const {
    const size_t nv = numVertices();
    Gradient result(*this, variableRestLen);

    const auto &dc = deformedConfiguration();
    // Compute gradient of each vertex's contribution to the twisting energy.
    for (size_t i = 1; i < nv - 1; ++i) {
        if (!sm.includeVtxStencil(nv, i)) continue;

        const Real_ inv_libar   = 2.0 / (m_restLen[i - 1] + m_restLen[i]);
        const Real_ inv_len_i   = 1.0 / dc.len[i    ],
                    inv_len_im1 = 1.0 / dc.len[i - 1];
        const Real_ deltaTwist = (dc.theta(i) - dc.theta(i - 1) + dc.referenceTwist[i] - m_restTwist[i]);

        if (variableRestLen) {
            Real_ dE_dljbar = -0.25 * m_twistingStiffness[i] * inv_libar * inv_libar * deltaTwist * deltaTwist;
            result.gradRestLen(i - 1) += dE_dljbar;
            result.gradRestLen(i    ) += dE_dljbar;
            if (restlenOnly) continue;
        }

        const Real_ dE_dm = inv_libar * m_twistingStiffness[i] * deltaTwist;
        // std::cout << "referenceTwist: " << dc.referenceTwist[i] << std::endl;
        // std::cout << "m_restTwist: " << m_restTwist[i] << std::endl;
        // std::cout << "dE_dm: " << dE_dm << std::endl;

        // Compute material frame angle (theta) dependence
        result.gradTheta(i - 1) -= dE_dm;
        result.gradTheta(i    ) += dE_dm;

        // Compute centerline position dependence
        Vec3 d_m_de_i   = 0.5 * inv_len_i   * dc.kb[i],
             d_m_de_im1 = 0.5 * inv_len_im1 * dc.kb[i];

        // If the source reference frame hasn't been updated to the current
        // reference frame, we need additional terms in the gradient
        // (that are not mentioned in Bergou2010's appendix)
        if (!updatedSource) {
            const auto &ti     = dc.tangent[i    ],
                       &tim1   = dc.tangent[i - 1],
                       &tsi    = dc.sourceTangent[i    ],
                       &tsim1  = dc.sourceTangent[i - 1],
                       &d1i    = dc.materialFrame[i    ].d1,
                       &d2i    = dc.materialFrame[i    ].d2,
                       &d1im1  = dc.materialFrame[i - 1].d1,
                       &d2im1  = dc.materialFrame[i - 1].d2,
                       &ds1i   = dc.sourceMaterialFrame[i    ].d1,
                       &ds1im1 = dc.sourceMaterialFrame[i - 1].d1;
            {
                Real_ inv_chi_hat = 1.0 / (1.0 + tsim1.dot(tim1));
                Real_ ds1im1_dot_tim1 = ds1im1.dot(tim1),
                      d2im1_dot_tsim1 = d2im1.dot(tsim1);

                Vec3 d_m_de_im1_contrib = inv_chi_hat * (ds1im1_dot_tim1 * d1im1.cross(tsim1) + d2im1_dot_tsim1 * ds1im1)
                                        - (inv_chi_hat * inv_chi_hat * ds1im1_dot_tim1 * d2im1_dot_tsim1) * tsim1
                                        + ds1im1.cross(d1im1);
                d_m_de_im1 += inv_len_im1 * (d_m_de_im1_contrib - tim1 * tim1.dot(d_m_de_im1_contrib));
            }
            {
                Real_ inv_chi_hat = 1.0 / (1.0 + tsi.dot(ti));
                Real_ ds1i_dot_ti = ds1i.dot(ti),
                      d2i_dot_tsi = d2i.dot(tsi);

                Vec3 d_m_de_i_contrib = inv_chi_hat * (ds1i_dot_ti * d1i.cross(tsi) + d2i_dot_tsi * ds1i)
                                        - (inv_chi_hat * inv_chi_hat * ds1i_dot_ti * d2i_dot_tsi) * tsi
                                        + ds1i.cross(d1i);
                d_m_de_i -= inv_len_i * (d_m_de_i_contrib - ti * ti.dot(d_m_de_i_contrib));
            }
        }

        result.gradPos(i - 1) -= dE_dm *  d_m_de_im1;
        result.gradPos(i    ) += dE_dm * (d_m_de_im1 - d_m_de_i);
        result.gradPos(i + 1) += dE_dm * d_m_de_i;
    }

    return result;
}

template<typename Real_>
template<class StencilMask>
typename ElasticRod_T<Real_>::Gradient ElasticRod_T<Real_>::gradEnergy(bool updatedSource, bool variableRestLen, bool restlenOnly, const StencilMask &sm) const {
    auto result = gradEnergyStretch(             variableRestLen, restlenOnly, sm);
    result +=     gradEnergyBend(updatedSource,  variableRestLen, restlenOnly, sm);
    result +=     gradEnergyTwist(updatedSource, variableRestLen, restlenOnly, sm);
    return result;
}

////////////////////////////////////////////////////////////////////////////////
// Elastic energy Hessians
////////////////////////////////////////////////////////////////////////////////
// The number of non-zeros in the Hessian's sparsity pattern (a tight
// upper bound for the number of non-zeros in the upper triangle for any
// configuration).
template<typename Real_>
size_t ElasticRod_T<Real_>::hessianNNZ(bool variableRestLen) const {
    const size_t nv = numVertices(), ne = numEdges();
    if (nv < 3) throw std::runtime_error("Too few vertices");
    size_t diag    = 6 * nv + ne;             // Size of diagonal blocks in x-x and theta-theta parts
    size_t odiagxx = 9 * (2 * (nv - 2) + 1);  // Size of off-diagonal blocks in the x-x part (the last vertex has no off-diag block, second-to-last has one, all the rest have 2)
    size_t odiagxt = 3 *                      // x-theta part consists of 3x1  blocks.
        (2 * 2                                //    The two endpoints always contribute 2 blocks each (for the two nearest edges).
        + ((nv >= 4) ? (3 * 2 + 4 * (nv - 4)) //    If there are at least 3 edges (nv >= 4), then the two endpoint-adjacent vertices contribute 3 blocks each, and the rest contribute 4.
                     : 2));                   //    Otherwise, the middle vertex contributes 2 blocks (for the two edges)
    size_t odiagtt = ne - 1;                  // theta-theta off diagonal consists of a single band.
    size_t result  = diag + odiagxx + odiagxt + odiagtt;

    if (variableRestLen) {
        result += odiagxt;    // x-restlen part is identical in size to x-theta part
        result += 3 * ne - 2; // theta-restlen block is tri-diagonal (and we take the whole thing)
        result += 2 * ne - 1; // restlen-restlen part is tridiagonal (and we take the upper tri)
    }
    return result;
}

template<typename Real_>
typename ElasticRod_T<Real_>::CSCMat ElasticRod_T<Real_>::hessianSparsityPattern(bool variableRestLen, Real_ val) const {
    CSCMat result;
    result.symmetry_mode = CSCMat::SymmetryMode::UPPER_TRIANGLE;
    const size_t nnz = hessianNNZ(variableRestLen);
    const size_t ndof = variableRestLen ? numExtendedDoF() : numDoF();
    const size_t nv = numVertices(),
                 ne = numEdges();

    result.m = result.n = ndof;
    result.nz = nnz;
    result.Ap.reserve(ndof + 1);
    result.Ai.reserve(nnz);
    result.Ax.assign(nnz, val);

    auto &Ap = result.Ap;
    auto &Ai = result.Ai;

    // Append the indices [start, end) to Ai
    auto addIdxRange = [&](const size_t start, const size_t end) {
        assert((start <= ndof) && (end <= ndof));
        const size_t len = end - start, oldSize = Ai.size();
        Ai.resize(oldSize + len);
        for (size_t i = 0; i < len; ++i)
            Ai[oldSize + i] = start + i;
    };

    // Build sparsity pattern directly in compressed column format
    result.Ap.push_back(0);
    // Loop over the columns of the Hessian by looping over all centerline pos, thetas, rest lengths in order
    // The vertex-associated columns of the Hessian's upper triangle hold only the x-x terms.
    for (size_t vi = 0; vi < nv; ++vi) {
        for (size_t c = 0; c < 3; ++c) {
            const size_t j = 3 * vi + c;
            // Column for x_vi has contributions from x_{vi - 2}, x_{vi - 1}, x_vi
            const size_t vi_m2 = vi - std::min<size_t>(2, vi);
            addIdxRange(3 * vi_m2, j + 1);
            Ap.push_back(Ai.size()); // End index for this column, start of next
        }
    }
    // The columnns associated with thetas contain the x-theta and theta-theta blocks
    for (size_t ei = 0; ei < ne; ++ei) {
        const size_t j = 3 * nv + ei;
        // Column for theta^ei has contributions from x_{ei - 1}, x_ei, x_{ei + 1}, x_{ei + 2}
        const size_t ei_m1 = ei - std::min<size_t>(1, ei),
                     ei_p2 = std::min<size_t>(ei + 2, nv - 1);
        addIdxRange(3 * ei_m1, 3 * (ei_p2 + 1));
        // Column for theta^ei has contributions from th^{ei - 1}, th^ei
        addIdxRange(3 * nv + ei_m1, j + 1);
        Ap.push_back(Ai.size()); // End index for this column, start of next
    }
    if (variableRestLen) {
        // The columnns associated with the rest lengths contain the x-rl and theta-rl and rl-rl blocks
        for (size_t ei = 0; ei < ne; ++ei) {
            const size_t j = 3 * nv + ne + ei;
            // Column for rl^ei has contributions from x_{ei - 1}, x_ei, x_{ei + 1}, x_{ei + 2}
            const size_t ei_m1 = ei - std::min<size_t>(1, ei),
                         ei_p2 = std::min<size_t>(ei + 2, nv - 1);
            addIdxRange(3 * ei_m1, 3 * (ei_p2 + 1));
            // Column for rl^ei has contributions from th^{ei - 1}, th^ei, th^{ei + 1}
            addIdxRange(3 * nv + ei_m1, 3 * nv + std::min<size_t>(ei + 1, ne - 1) + 1);

            // Column for rl^ei has contributions from rl^{ei - 1}, rl^ei
            addIdxRange(3 * nv + ne + ei_m1, j + 1);
            Ap.push_back(Ai.size()); // End index for this column, start of next
        }
    }

    return result;
}

// Hessian of twisting energy with respect to material axis (theta) variables.
// This hessian is a constant tridiagonal matrix that can be interpreted as a
// non-uniform 1D Laplacian (for the dual mesh)
template<typename Real_>
TriDiagonalSystem<Real_> ElasticRod_T<Real_>::hessThetaEnergyTwist() const {
    const size_t ne = numEdges(), nv = numVertices();
    // Construct -1, 0, 1 diagonals in vectors a, d, c respectively
    // Since the matrix is symmetric we only need to construct d and c (upper triangle)
    std::vector<Real_> d(ne), c(ne - 1);

    // Compute Hessian of internal vertices' contributions to the twisting energy
    for (size_t i = 1; i < nv - 1; ++i) {
        Real_ coeff = (2.0 * m_twistingStiffness[i]) / (m_restLen[i - 1] + m_restLen[i]);
        // Recall, the gradient contributions were
        //      gt[i - 1] -= coeff * (theta[i] - theta[i - 1] + const);
        //      gt[i    ] += coeff * (theta[i] - theta[i - 1] + const);
        d[i - 1] +=  coeff; // (i - 1, i - 1)
        d[i    ] +=  coeff; // (    i,     i)
        c[i - 1]  = -coeff; // (i - 1,    gradient i)
    }

    auto a = c;
    return TriDiagonalSystem<Real_>(std::move(a), std::move(d), std::move(c));
}

// Accumulate stretching Hessian into H.
template<typename Real_>
void ElasticRod_T<Real_>::hessEnergyStretch(ElasticRod_T<Real_>::CSCMat &H, bool variableRestLen) const {
    using M3d = Mat3_T<Real_>;
    assert(H.symmetry_mode == CSCMat::SymmetryMode::UPPER_TRIANGLE);
    const size_t ndof = variableRestLen ? numExtendedDoF() : numDoF();
    assert((size_t(H.m) == ndof) && (size_t(H.n) == ndof));
    UNUSED(ndof);

    const auto &dc = deformedConfiguration();
    const size_t ne = numEdges(), nv = numVertices();

    // Accumulate per-edge Hessian contributions
    for (size_t j = 0; j < ne; ++j) {
        const Real_ ks = density(j) * m_stretchingStiffness[j];
        const Real_ ks_epsilon_div_lj = ks * (1.0 / m_restLen[j] - 1.0 / dc.len[j]);
        const Real_ coeff = ks / m_restLen[j] - ks_epsilon_div_lj;
        const auto &t = dc.tangent[j];

        // Per-edge Hessian consists of four 3x3 blocks that are identical up to sign.
        // The sign is negative for the mixed derivatives, and positive for the
        // Hessian with respect to a single vertex.
        M3d hessianBlock = ks_epsilon_div_lj * M3d::Identity()
                         + (coeff * t) * t.transpose();

        // Only two vertices affect edge 'j': vertex 'j' and 'j + 1'
        for (size_t col = 0; col < 3; ++col) {
            // prev-prev, prev-next, next-next
                         H.addNZ(3 * (j    ), 3 * (    j) + col,  hessianBlock.col(col).head(col + 1)); // d2 / (dx_{    j} dx_{    j})
            size_t idx = H.addNZ(3 * (j    ), 3 * (j + 1) + col, -hessianBlock.col(col)              ); // d2 / (dx_{    j} dx_{j + 1})
                         H.addNZ(idx,                             hessianBlock.col(col).head(col + 1)); // d2 / (dx_{j + 1} dx_{j + 1})
        }

        if (variableRestLen) {
            const size_t rl_offset = 3 * nv + ne + j;
            Real_ fracLen = dc.len[j] / m_restLen[j];
            // (x, restlen) term
            // -(grad l^j) * ks * (l^j / (restLen^j)^2) := -block
            Vec3 block = t * ks * fracLen / m_restLen[j];
            size_t next_idx = H.addNZ(3 * j, rl_offset, block);
                              H.addNZ(next_idx,        -block);
            // (restlen, restlen) term
            H.addNZ(rl_offset, rl_offset, ks * fracLen * fracLen / m_restLen[j]);
        }
    }
}

// Hessian ***evaluated assuming the source frame has been updated to the current frame***
template<typename Real_>
void ElasticRod_T<Real_>::hessEnergyBend(ElasticRod_T<Real_>::CSCMat &H, bool variableRestLen) const {
    assert(H.symmetry_mode == CSCMat::SymmetryMode::UPPER_TRIANGLE);
    const size_t ndof = variableRestLen ? numExtendedDoF() : numDoF();
    UNUSED(ndof);
    assert((size_t(H.m) == ndof) && (size_t(H.n) == ndof));

    using M3d = Mat3_T<Real_>;
    using M2d = Mat2_T<Real_>;

    const size_t nv = numVertices(), ne = numEdges();
    const auto &dc = deformedConfiguration();

    for (size_t i = 1; i < nv - 1; ++i) {
        const auto &kb = dc.kb[i];
        const Real_ inv_2libar = 1.0 / (m_restLen[i - 1] + m_restLen[i]);
        const std::array<Real_, 2> B_div_2libar = {{m_bendingStiffness[i].lambda_1 * inv_2libar,
                                                    m_bendingStiffness[i].lambda_2 * inv_2libar}};

        // Accumulate per-interior-vertex Hessian contributions.
        // The DoFs involved are the centerline positions of the vertex and its two neighbors
        // and the twisting angles of the two incident edges.
        Eigen::Matrix<Real_,  9,  9> perVertexHessian_x_x;        // No zero-initialization needed
        Eigen::Matrix<Real_,  9,  2> perVertexHessian_x_theta(    Eigen::Matrix<Real_, 9, 2>::Zero());
        Eigen::Matrix<Real_,  2,  2> perVertexHessian_theta_theta(Eigen::Matrix<Real_, 2, 2>::Zero());

        //////////////////////////////////////////////////////
        // Quantities needed by multiple parts of the Hessian.
        //////////////////////////////////////////////////////
        const auto &ti    = dc.tangent[i],
                   &tim1  = dc.tangent[i - 1];
        std::array<Real_, 2> inv_len{{1.0 / dc.len[i - 1],
                                      1.0 / dc.len[i    ]}};

        const Real_ inv_chi = 1.0 / (1.0 + tim1.dot(ti));
        const Vec3  t_tilde = (tim1  + ti ) * inv_chi;
        // Precompute some matrix terms that will be re-used
        M3d tilde_t_otimes_t_2 = 2 * t_tilde * t_tilde.transpose();
        M3d I_minus_ti_otimes_ti     = M3d::Identity() - ti   * ti  .transpose(),
            I_minus_tim1_otimes_tim1 = M3d::Identity() - tim1 * tim1.transpose(),
            I_plus_tim1_otimes_ti    = M3d::Identity() + tim1 * ti  .transpose();

        M3d d2E_deim1_deim1(M3d::Zero()),
            d2E_deim1_dei  (M3d::Zero()),
            d2E_dei_dei    (M3d::Zero());

        ////////////////////////////////////////////////////////////////////////
        // Gradient outer product terms
        ////////////////////////////////////////////////////////////////////////
        std::array<Vec3, 2> oproduct_grad_kappa_term_ei, oproduct_grad_kappa_term_eim1;
        // d_kappa_k_j_de_im1[k][j] = d/deim1 (kappa_k)_i^j
        std::array<std::array<Vec3, 2>, 2> d_kappa_k_j_de_im1, d_kappa_k_j_de_i;

        // d_kappa_k_j_dtheta_j(k, j) = d/dtheta_j (kappa_k)_i^j
        M2d d_kappa_k_j_dtheta_j;
        M2d oproduct_grad_kappa_k_term_theta;

        for (size_t k = 0; k < 2; ++k) {
            const size_t kother = (k + 1) % 2;
            for (size_t adj_edge = 0; adj_edge < 2; ++adj_edge) {
                const size_t j = adj_edge + (i - 1);
                d_kappa_k_j_dtheta_j(k, adj_edge) = -kb.dot(dc.materialFrame[j].get(k));
                const Real_ sign = (k == 0) ? 1 : -1; // Infinitesimal transport kappa_2^j term is just like kappa_1, except d2 is replaced with -d1.
                d_kappa_k_j_de_im1[k][adj_edge] = inv_len[0] * (( 2 * sign * inv_chi) *   ti.cross(dc.materialFrame[j].get(kother)) - dc.per_corner_kappa[i].col(adj_edge)[k] * t_tilde);
                d_kappa_k_j_de_i  [k][adj_edge] = inv_len[1] * ((-2 * sign * inv_chi) * tim1.cross(dc.materialFrame[j].get(kother)) - dc.per_corner_kappa[i].col(adj_edge)[k] * t_tilde);
            }
        }

        if (m_bendingEnergyType == BendingEnergyType::Bergou2010) {
            for (size_t k = 0; k < 2; ++k) {
                oproduct_grad_kappa_term_ei  [k] = 0.5 * (d_kappa_k_j_de_i  [k][0] + d_kappa_k_j_de_i  [k][1]);
                oproduct_grad_kappa_term_eim1[k] = 0.5 * (d_kappa_k_j_de_im1[k][0] + d_kappa_k_j_de_im1[k][1]);
            }
            oproduct_grad_kappa_k_term_theta = 0.5 * d_kappa_k_j_dtheta_j;
        }
        for (size_t adj_edge = 0; adj_edge < 2; ++adj_edge) {
            const size_t j = adj_edge + (i - 1);
            for (size_t k = 0; k < 2; ++k) {
                if (m_bendingEnergyType == BendingEnergyType::Bergou2008) {
                    oproduct_grad_kappa_term_eim1[k] = (m_restLen[j] * 2 * inv_2libar) * d_kappa_k_j_de_im1[k][adj_edge];
                    oproduct_grad_kappa_term_ei  [k] = (m_restLen[j] * 2 * inv_2libar) * d_kappa_k_j_de_i  [k][adj_edge];

                    oproduct_grad_kappa_k_term_theta.setZero();
                    oproduct_grad_kappa_k_term_theta.col(adj_edge) = m_restLen[j] * 2 * inv_2libar * d_kappa_k_j_dtheta_j.col(adj_edge);
                }

                // e, e outer product term
                d2E_deim1_deim1 += B_div_2libar[k] * (oproduct_grad_kappa_term_eim1[k] * d_kappa_k_j_de_im1[k][adj_edge].transpose());
                d2E_deim1_dei   += B_div_2libar[k] * (oproduct_grad_kappa_term_eim1[k] * d_kappa_k_j_de_i  [k][adj_edge].transpose());
                d2E_dei_dei     += B_div_2libar[k] * (oproduct_grad_kappa_term_ei  [k] * d_kappa_k_j_de_i  [k][adj_edge].transpose());

                // x, theta outer product term: theta_j only affects kappa_k_j
                perVertexHessian_x_theta.template block<3, 1>(0, adj_edge) += (B_div_2libar[k] * d_kappa_k_j_dtheta_j(k, adj_edge)) * (-oproduct_grad_kappa_term_eim1[k]                                 ); // (x i - 1, theta j)
                perVertexHessian_x_theta.template block<3, 1>(3, adj_edge) += (B_div_2libar[k] * d_kappa_k_j_dtheta_j(k, adj_edge)) * ( oproduct_grad_kappa_term_eim1[k] - oproduct_grad_kappa_term_ei[k]); // (x i    , theta j)
                perVertexHessian_x_theta.template block<3, 1>(6, adj_edge) += (B_div_2libar[k] * d_kappa_k_j_dtheta_j(k, adj_edge)) * ( oproduct_grad_kappa_term_ei  [k]                                 ); // (x i + 1, theta j)

                // theta, theta outer product term
                perVertexHessian_theta_theta.col(adj_edge) += B_div_2libar[k] * d_kappa_k_j_dtheta_j(k, adj_edge) * oproduct_grad_kappa_k_term_theta.row(k).transpose();
            }
        }

        ////////////////////////////////////////////////////////////////////////
        // Kappa Hessian terms
        ////////////////////////////////////////////////////////////////////////
        for (size_t adj_edge = 0; adj_edge < 2; ++adj_edge) { // 0 ==> i - 1, 1 ==> i
            size_t j = (i + adj_edge) - 1;
            const auto &t_j = dc.tangent[j];

            // dE_dkappa_k_j[k] = dE/(kappa_k)_i^j
            std::array<Real_, 2> dE_dkappa_k_j;
            if (m_bendingEnergyType == BendingEnergyType::Bergou2010) {
                const Vec2 kappaDiff = dc.kappa[i] - m_restKappa[i];
                dE_dkappa_k_j = {{B_div_2libar[0] * kappaDiff[0],
                                  B_div_2libar[1] * kappaDiff[1]}};
            }
            else if (m_bendingEnergyType == BendingEnergyType::Bergou2008) {
                const Vec2 kappaDiff = dc.per_corner_kappa[i].col(adj_edge) - m_restKappa[i];
                dE_dkappa_k_j = {{2.0 * m_restLen[j] * inv_2libar * B_div_2libar[0] * kappaDiff[0],
                                  2.0 * m_restLen[j] * inv_2libar * B_div_2libar[1] * kappaDiff[1]}};
            }
            else { assert(false); }

            for (size_t k = 0; k < 2; ++k) {
                const size_t kother = (k + 1) % 2;
                const double sign = (k == 0) ? 1.0 : -1.0; // Infinitesimal transport kappa_2^j term is just like kappa_1, except d2 is replaced with -d1.
                // Contribution from H (kappa_k)_i^j
                // (x, x) part (upper left block)
                const Real_      kappa = dc.per_corner_kappa[i].col(adj_edge)[k];
                const Real_ half_kappa = kappa * 0.5;
                const Real_ sign_inv_chi_2 = sign * 2 * inv_chi;
                const auto &d_kother_j = dc.materialFrame[j].get(kother);
                const auto &d_k_j      = dc.materialFrame[j].get(k);

                Vec3 cross_prod_term_eim1 = sign_inv_chi_2 * ti  .cross(d_kother_j),
                     cross_prod_term_ei   = sign_inv_chi_2 * tim1.cross(d_kother_j);
                // Use symmetry of d2_kappa_k_j_deim1_deim1 and d2_kappa_k_j_dei_dei to speed up calculation (full result will be d2_kappa_k_j_deim1_deim1 + d2_kappa_k_j_deim1_deim1.transpose())
                M3d d2_kappa_k_j_deim1_deim1 = half_kappa * tilde_t_otimes_t_2 - (cross_prod_term_eim1 * t_tilde.transpose() /* + t_tilde otimes cross_prod_term_eim1 */  ) - (half_kappa * inv_chi) * I_minus_tim1_otimes_tim1;
                M3d d2_kappa_k_j_dei_dei     = half_kappa * tilde_t_otimes_t_2 + (cross_prod_term_ei   * t_tilde.transpose() /* + t_tilde otimes cross_prod_term_e    */  ) - (half_kappa * inv_chi) * I_minus_ti_otimes_ti;
                M3d d2_kappa_k_j_deim1_dei   =      kappa * tilde_t_otimes_t_2 - (cross_prod_term_eim1 * t_tilde.transpose() - t_tilde * cross_prod_term_ei  .transpose() ) - (     kappa * inv_chi) * I_plus_tim1_otimes_ti;

                // Add in [d_kother_j]_x term
                d2_kappa_k_j_deim1_dei(0, 1) += sign_inv_chi_2 * d_kother_j[2];
                d2_kappa_k_j_deim1_dei(0, 2) -= sign_inv_chi_2 * d_kother_j[1];
                d2_kappa_k_j_deim1_dei(1, 2) += sign_inv_chi_2 * d_kother_j[0];
                d2_kappa_k_j_deim1_dei(1, 0) -= sign_inv_chi_2 * d_kother_j[2];
                d2_kappa_k_j_deim1_dei(2, 0) += sign_inv_chi_2 * d_kother_j[1];
                d2_kappa_k_j_deim1_dei(2, 1) -= sign_inv_chi_2 * d_kother_j[0];

                // We have a finite transport term only for the derivative with respect to edge "j"; leverage symmetry to drop half of the terms...
                M3d finiteTransportTerm = sign * (kb * d_kother_j.transpose() /* + d_kother_j * kb.transpose() */) + 0.5 * (kb.cross(t_j) * d_k_j.transpose() /* + d_k_j * kb.cross(t_j).transpose() */) /* - half_kappa * (M3d::Identity() - t_j * t_j.transpose()) <---- moved below */;

                if (adj_edge == 0) d2_kappa_k_j_deim1_deim1 += finiteTransportTerm - half_kappa * I_minus_tim1_otimes_tim1;
                if (adj_edge == 1) d2_kappa_k_j_dei_dei     += finiteTransportTerm - half_kappa * I_minus_ti_otimes_ti;

                d2E_deim1_deim1 += (dE_dkappa_k_j[k] * inv_len[0] * inv_len[0]) * (d2_kappa_k_j_deim1_deim1 + d2_kappa_k_j_deim1_deim1.transpose());
                d2E_dei_dei     += (dE_dkappa_k_j[k] * inv_len[1] * inv_len[1]) * (d2_kappa_k_j_dei_dei     + d2_kappa_k_j_dei_dei    .transpose());
                d2E_deim1_dei   += (dE_dkappa_k_j[k] * inv_len[0] * inv_len[1]) * (d2_kappa_k_j_deim1_dei                                         );

                // Theta, theta Hessian (Hessian of kappas with respect to thetas have no cross terms)
                perVertexHessian_theta_theta(adj_edge, adj_edge) += dE_dkappa_k_j[k] * sign * d_kappa_k_j_dtheta_j(kother, adj_edge);

                // x, theta Hessian
                Vec3 d2_kappa_k_j_deim1_dtheta_term = (dE_dkappa_k_j[k] * inv_len[0]) * (-(2 * inv_chi) * ti  .cross(d_k_j) + kb.dot(d_k_j) * t_tilde);
                Vec3 d2_kappa_k_j_dei_dtheta_term   = (dE_dkappa_k_j[k] * inv_len[1]) * ( (2 * inv_chi) * tim1.cross(d_k_j) + kb.dot(d_k_j) * t_tilde);

                perVertexHessian_x_theta.template block<3, 1>(0, adj_edge) += -d2_kappa_k_j_deim1_dtheta_term                               ; // (x i - 1, theta j)
                perVertexHessian_x_theta.template block<3, 1>(3, adj_edge) +=  d2_kappa_k_j_deim1_dtheta_term - d2_kappa_k_j_dei_dtheta_term; // (x i    , theta i - 1)
                perVertexHessian_x_theta.template block<3, 1>(6, adj_edge) +=  d2_kappa_k_j_dei_dtheta_term                                 ; // (x i + 1, theta i - 1)
            }
        }

#if 0 // Incorrect versions in Bergou2010
        d2k1_deim1_deim1 = inv_len_im1 * inv_len_im1 * ( (2 * k1 * t_tilde) * t_tilde.transpose() - ti  .cross(d2_tilde) * t_tilde.transpose() - t_tilde * ti  .cross(d2_tilde).transpose() - inv_chi * k1 * (Eigen::Matrix3d::Identity() - tim1 * tim1.transpose()) + 0.25 * (kb * d2im1.transpose() + d2im1 * kb.transpose()));
        d2k1_dei_dei     = inv_len_i   * inv_len_i   * ( (2 * k1 * t_tilde) * t_tilde.transpose() + tim1.cross(d2_tilde) * t_tilde.transpose() + t_tilde * tim1.cross(d2_tilde).transpose() - inv_chi * k1 * (Eigen::Matrix3d::Identity() - ti   * ti  .transpose()) + 0.25 * (kb * d2i  .transpose() + d2i   * kb.transpose()));
#endif

        perVertexHessian_x_x.template block<3, 3>(0, 0) =  d2E_deim1_deim1                                                          ; // (x i - 1, x i - 1) ==> (-ei-1, -ei-1)
        perVertexHessian_x_x.template block<3, 3>(0, 3) =  d2E_deim1_dei   - d2E_deim1_deim1                                        ; // (x i - 1, x i    ) ==> (-ei-1, +ei-1), (-ei-1, -ei)
        perVertexHessian_x_x.template block<3, 3>(0, 6) = -d2E_deim1_dei                                                            ; // (x i - 1, x i + 1) ==> (-ei-1, +ei)
     // perVertexHessian_x_x.template block<3, 3>(3, 0) =  d2E_deim1_dei.transpose() - d2E_deim1_deim1                              ; // (x i    , x i - 1) ==> (-ei, -ei-1), (+ei-1, -ei-1)
        perVertexHessian_x_x.template block<3, 3>(3, 3) =  d2E_deim1_deim1 - d2E_deim1_dei - d2E_deim1_dei.transpose() + d2E_dei_dei; // (x i    , x i    ) ==> (+ei-1, +ei-1), (+ei-1, -ei), (-ei, +ei-1), (-ei, -ei)
        perVertexHessian_x_x.template block<3, 3>(3, 6) =  d2E_deim1_dei   - d2E_dei_dei                                            ; // (x i    , x i + 1) ==> (+ei-1, +ei), (-ei, +ei)
     // perVertexHessian_x_x.template block<3, 3>(6, 0) = -d2E_deim1_dei.transpose()                                                ; // (x i + 1, x i - 1) ==> (+ei, -ei-1)
     // perVertexHessian_x_x.template block<3, 3>(6, 3) =  d2E_deim1_dei.transpose() - d2E_dei_dei                                  ; // (x i + 1, x i    ) ==> (+ei, +ei-1), (+ei, -ei)
        perVertexHessian_x_x.template block<3, 3>(6, 6) =  d2E_dei_dei                                                              ; // (x i + 1, x i + 1) ==> (+ei, +ei)

        // Offset into full sparse Hessian where we should accumulate contributions.
        const size_t x_offset = 3 * (i - 1),      // Index of the first position variable for the stencil
                 theta_offset = 3 * nv + (i - 1); // Index of the first theta variable

        /////////////////////////////////////////////
        // Assemble Hessian blocks into sparse matrix
        /////////////////////////////////////////////
        for (size_t c1 = 0; c1 < 9; ++c1) H.addNZ(    x_offset,     x_offset + c1, perVertexHessian_x_x.col(c1).head(c1 + 1));
        for (size_t c1 = 0; c1 < 2; ++c1) H.addNZ(    x_offset, theta_offset + c1, perVertexHessian_x_theta.col(c1));
        for (size_t c1 = 0; c1 < 2; ++c1) H.addNZ(theta_offset, theta_offset + c1, perVertexHessian_theta_theta.col(c1).head(c1 + 1));

        /////////////////////////////////////////////
        // Rest length derivatives
        /////////////////////////////////////////////
        if (!variableRestLen) continue;
        Vec3 tilde_t = inv_chi * (ti + tim1);
        const size_t rl_offset = theta_offset + ne; // index of the first rest length variable for the stencil

        for (size_t adj_edge = 0; adj_edge < 2; ++adj_edge) { // 0 ==> i - 1, 1 ==> i
            const size_t j = (i - 1) + adj_edge;
            const size_t jother = i - adj_edge; // 0 ==> i, 1 ==> i - 1

            // Derivative of the bending energy with respect to (kappa_k)_i^j
            std::array<Real_, 2> d2E_dkappa_k_j_dljbar, d2E_dkappa_k_j_dljotherbar;
            if (m_bendingEnergyType == BendingEnergyType::Bergou2010) {
                const Vec2 kappaDiff = dc.kappa[i] - m_restKappa[i];
                d2E_dkappa_k_j_dljbar = {{-inv_2libar * inv_2libar * m_bendingStiffness[i].lambda_1 * kappaDiff[0],
                                          -inv_2libar * inv_2libar * m_bendingStiffness[i].lambda_2 * kappaDiff[1]}};
                d2E_dkappa_k_j_dljotherbar = d2E_dkappa_k_j_dljbar;

                Real_ contrib = (m_bendingStiffness[i].lambda_1 * kappaDiff[0] * kappaDiff[0]
                               + m_bendingStiffness[i].lambda_2 * kappaDiff[1] * kappaDiff[1]) * 2.0 * inv_2libar * inv_2libar * inv_2libar;
                if (adj_edge == 0) H.addNZ(rl_offset + adj_edge, rl_offset    , contrib);
                H.addNZ                   (rl_offset + adj_edge, rl_offset + 1, contrib);
            }
            else if (m_bendingEnergyType == BendingEnergyType::Bergou2008) {
                const Vec2 kappaDiff = dc.per_corner_kappa[i].col(adj_edge) - m_restKappa[i];
                d2E_dkappa_k_j_dljbar      = {{ 2.0 * (inv_2libar * inv_2libar * inv_2libar) * (m_restLen[jother] - m_restLen[j]) * m_bendingStiffness[i].lambda_1 * kappaDiff[0],
                                                2.0 * (inv_2libar * inv_2libar * inv_2libar) * (m_restLen[jother] - m_restLen[j]) * m_bendingStiffness[i].lambda_2 * kappaDiff[1]}};
                d2E_dkappa_k_j_dljotherbar = {{-2.0 * (inv_2libar * inv_2libar * inv_2libar) * (                2 * m_restLen[j]) * m_bendingStiffness[i].lambda_1 * kappaDiff[0],
                                               -2.0 * (inv_2libar * inv_2libar * inv_2libar) * (                2 * m_restLen[j]) * m_bendingStiffness[i].lambda_2 * kappaDiff[1]}};
                Real_ contrib = (m_bendingStiffness[i].lambda_1 * kappaDiff[0] * kappaDiff[0]
                               + m_bendingStiffness[i].lambda_2 * kappaDiff[1] * kappaDiff[1]) * inv_2libar * inv_2libar * inv_2libar * inv_2libar;

                size_t var      = rl_offset +     adj_edge,
                       varOther = rl_offset + 1 - adj_edge;
                H.addNZ(var     , var     , contrib * (2 * m_restLen[j] - 4 * m_restLen[jother]));
                H.addNZ(varOther, varOther, contrib * (6 * m_restLen[j]                        ));
                if (var > varOther) std::swap(var, varOther); // Upper triangle entry only
                H.addNZ(var, varOther, contrib * (4 * m_restLen[j] - 2 * m_restLen[jother]));
            }
            else { assert(false); }

            // Accumulate energy dependence through (kappa_k)_i^j
            for (size_t k = 0; k < 2; ++k) {
                // Compute material frame angle (theta) dependence
                H.addNZ(theta_offset + adj_edge, rl_offset +     adj_edge, -d2E_dkappa_k_j_dljbar     [k] * kb.dot(dc.materialFrame[j].get(k)));
                H.addNZ(theta_offset + adj_edge, rl_offset + 1 - adj_edge, -d2E_dkappa_k_j_dljotherbar[k] * kb.dot(dc.materialFrame[j].get(k)));

                // Compute centerline position dependence
                // First, compute variation of kappa_k^j with respect to a perturbation of the edge tangents.
                const size_t kother = (k + 1) % 2;
                double sign = (k == 0) ? 1 : -1; // Infinitesimal transport kappa_2^j term is just like kappa_1, except d2 is replaced with -d1.
                Vec3 d_kappa_k_j_de_im1 = inv_len[0] * (( 2 * sign * inv_chi) *   ti.cross(dc.materialFrame[j].get(kother)) - dc.per_corner_kappa[i].col(adj_edge)[k] * tilde_t);
                Vec3 d_kappa_k_j_de_i   = inv_len[1] * ((-2 * sign * inv_chi) * tim1.cross(dc.materialFrame[j].get(kother)) - dc.per_corner_kappa[i].col(adj_edge)[k] * tilde_t);

                // Second, compute variation of bending energy with respect to the centerline positions
                for (size_t c = 0; c < 3; ++c) {
                    H.addNZ(x_offset + 0 + c, rl_offset +     adj_edge, -d2E_dkappa_k_j_dljbar     [k] *  d_kappa_k_j_de_im1[c]);                        // (x i - 1, rl j)
                    H.addNZ(x_offset + 0 + c, rl_offset + 1 - adj_edge, -d2E_dkappa_k_j_dljotherbar[k] *  d_kappa_k_j_de_im1[c]);                        // (x i - 1, rl jother)

                    H.addNZ(x_offset + 3 + c, rl_offset +     adj_edge,  d2E_dkappa_k_j_dljbar     [k] * (d_kappa_k_j_de_im1[c] - d_kappa_k_j_de_i[c])); // (x i    , rl j)
                    H.addNZ(x_offset + 3 + c, rl_offset + 1 - adj_edge,  d2E_dkappa_k_j_dljotherbar[k] * (d_kappa_k_j_de_im1[c] - d_kappa_k_j_de_i[c])); // (x i    , rl jother)

                    H.addNZ(x_offset + 6 + c, rl_offset +     adj_edge,  d2E_dkappa_k_j_dljbar     [k] *  d_kappa_k_j_de_i[c]);                          // (x i + 1, rl j)
                    H.addNZ(x_offset + 6 + c, rl_offset + 1 - adj_edge,  d2E_dkappa_k_j_dljotherbar[k] *  d_kappa_k_j_de_i[c]);                          // (x i + 1, rl jother)
                }
            }
        }
    }
}

// Hessian ***evaluated assuming the source frame has been updated to the current frame***
template<typename Real_>
void ElasticRod_T<Real_>::hessEnergyTwist(ElasticRod_T<Real_>::CSCMat &H, bool variableRestLen) const {
    assert(H.symmetry_mode == CSCMat::SymmetryMode::UPPER_TRIANGLE);
    const size_t ndof = variableRestLen ? numExtendedDoF() : numDoF();
    assert((size_t(H.m) == ndof) && (size_t(H.n) == ndof));
    UNUSED(ndof);

    using M3d = Eigen::Matrix<Real_, 3, 3>;

    const size_t nv = numVertices(), ne = numEdges();
    const auto &dc = deformedConfiguration();

    // Accumulate per-interior-vertex Hessian contributions.
    // The DoFs involved are the centerline positions of the vertex and its two neighbors
    // and the twisting angles of the two incident edges.
    Eigen::Matrix<Real_,  9,  9> perVertexHessian_x_x;
    Eigen::Matrix<Real_,  9,  2> perVertexHessian_x_theta;

    for (size_t i = 1; i < nv - 1; ++i) {
        const auto &kb = dc.kb[i];

        /////////////////////////////////
        // (x, x) part (upper left block)
        /////////////////////////////////
        const Real_ inv_libar2 = 1.0 / (m_restLen[i - 1] + m_restLen[i]);
        const M3d block = (0.5 * m_twistingStiffness[i] * inv_libar2) * (kb * kb.transpose());
        const Real_ inv_len_i   = 1.0 / dc.len[i],
                    inv_len_im1 = 1.0 / dc.len[i - 1];

        // Local and global indices of the involved vertex and edge quantities:
        //       9      10
        //      i-1      i
        //  +--------+-------+
        // i-1       i      i+1
        //  0        3       6

        // Outer product term
        perVertexHessian_x_x.template block<3, 3>(0, 0) = (             -inv_len_im1 * -inv_len_im1             ) * block; // (i - 1, i - 1)
        perVertexHessian_x_x.template block<3, 3>(0, 3) = (             -inv_len_im1 * (inv_len_im1 - inv_len_i)) * block; // (i - 1,     i)
        perVertexHessian_x_x.template block<3, 3>(0, 6) = (             -inv_len_im1 * inv_len_i                ) * block; // (i - 1, i + 1)
     // perVertexHessian_x_x.template block<3, 3>(3, 0) = ((inv_len_im1 - inv_len_i) * -inv_len_im1             ) * block; // (    i, i - 1)
        perVertexHessian_x_x.template block<3, 3>(3, 3) = ((inv_len_im1 - inv_len_i) * (inv_len_im1 - inv_len_i)) * block; // (    i,     i)
        perVertexHessian_x_x.template block<3, 3>(3, 6) = ((inv_len_im1 - inv_len_i) * inv_len_i                ) * block; // (    i, i + 1)
     // perVertexHessian_x_x.template block<3, 3>(6, 0) = (                inv_len_i * -inv_len_im1             ) * block; // (i + 1, i - 1)
     // perVertexHessian_x_x.template block<3, 3>(6, 3) = (                inv_len_i * (inv_len_im1 - inv_len_i)) * block; // (i + 1,     i)
        perVertexHessian_x_x.template block<3, 3>(6, 6) = (                inv_len_i *  inv_len_i               ) * block; // (i + 1, i + 1)

        // Twist Hessian term
        Real_ dE_dm = 2.0 * inv_libar2 * m_twistingStiffness[i] * (dc.theta(i) - dc.theta(i - 1) + dc.referenceTwist[i] - m_restTwist[i]);

        const Vec3    &ti   = dc.tangent[i];
        const Vec3    &tim1 = dc.tangent[i - 1];
        const Real_ inv_chi = 1.0 / (1.0 + tim1.dot(ti));
        const Vec3  t_tilde = (tim1 + ti) * inv_chi;

        // surprisingly correct symmetrized formulas from Bergou2010's appendix
        M3d tmp = kb * (t_tilde + ti  ).transpose();
        const M3d d2E_dei_dei     = -(dE_dm * 0.25 * inv_len_i   * inv_len_i  ) * (tmp + tmp.transpose()); tmp = kb * (t_tilde + tim1).transpose();
        const M3d d2E_deim1_deim1 = -(dE_dm * 0.25 * inv_len_im1 * inv_len_im1) * (tmp + tmp.transpose());

        M3d d2E_deim1_dei  = /* cross product matrix term added, and coefficient applied below */ -kb * t_tilde.transpose();
        d2E_deim1_dei(0, 1) -= (2.0 * inv_chi) * tim1[2];
        d2E_deim1_dei(0, 2) += (2.0 * inv_chi) * tim1[1];
        d2E_deim1_dei(1, 2) -= (2.0 * inv_chi) * tim1[0];
        d2E_deim1_dei(1, 0) += (2.0 * inv_chi) * tim1[2];
        d2E_deim1_dei(2, 0) -= (2.0 * inv_chi) * tim1[1];
        d2E_deim1_dei(2, 1) += (2.0 * inv_chi) * tim1[0];
        d2E_deim1_dei *= dE_dm * 0.5 * inv_len_im1 * inv_len_i;

        perVertexHessian_x_x.template block<3, 3>(0, 0) += d2E_deim1_deim1;                                                           // (-ei-1, -ei-1)
        perVertexHessian_x_x.template block<3, 3>(0, 3) += d2E_deim1_dei - d2E_deim1_deim1;                                           // (-ei-1, -ei), (-ei-1, +ei-1)
        perVertexHessian_x_x.template block<3, 3>(0, 6) -= d2E_deim1_dei;                                                             // (-ei-1, +ei)
     // perVertexHessian_x_x.template block<3, 3>(3, 0) += d2E_dei_deim1 - d2E_deim1_deim1;                                           // (-ei, -ei-1), (+ei-1, -ei-1)
        perVertexHessian_x_x.template block<3, 3>(3, 3) += d2E_dei_dei - d2E_deim1_dei.transpose() - d2E_deim1_dei + d2E_deim1_deim1; // (-ei, -ei), (-ei, +ei-1), (+ei-1, -ei), (+ei-1, +ei-1)
        perVertexHessian_x_x.template block<3, 3>(3, 6) += d2E_deim1_dei - d2E_dei_dei;                                               // (-ei, +ei), (+ei-1, +ei)
     // perVertexHessian_x_x.template block<3, 3>(6, 0) -= d2E_dei_deim1;                                                             // (+ei, -ei-1)
     // perVertexHessian_x_x.template block<3, 3>(6, 3) += d2E_dei_deim1 - d2E_dei_dei;                                               // (+ei, -ei), (+ei, +ei-1)
        perVertexHessian_x_x.template block<3, 3>(6, 6) += d2E_dei_dei;                                                               // (+ei, +ei)

        //////////////////////////////////////
        // (x, theta) part (upper right block)
        //////////////////////////////////////
        // Outer product term
        Vec3 scaled_kb = m_twistingStiffness[i] * inv_libar2 * kb;
        perVertexHessian_x_theta.template block<3, 1>(0, 0) =  inv_len_im1 * scaled_kb;               // (x i-1, theta i - 1)
        perVertexHessian_x_theta.template block<3, 1>(0, 1) = -inv_len_im1 * scaled_kb;               // (x i-1, theta i)
        perVertexHessian_x_theta.template block<3, 1>(3, 0) = -(inv_len_im1 - inv_len_i) * scaled_kb; // (x i  , theta i - 1)
        perVertexHessian_x_theta.template block<3, 1>(3, 1) =  (inv_len_im1 - inv_len_i) * scaled_kb; // (x i  , theta i)
        perVertexHessian_x_theta.template block<3, 1>(6, 0) = -inv_len_i * scaled_kb;                 // (x i+1, theta i - 1)
        perVertexHessian_x_theta.template block<3, 1>(6, 1) =  inv_len_i * scaled_kb;                 // (x i+1, theta i)

        // Assemble contributions to the full sparse Hessian
        const size_t x_offset = 3 * (i - 1),      // Index of the first position variable for the stencil
                 theta_offset = 3 * nv + (i - 1); // Index of the first theta variable

        // Assemble x-x, x-theta partials
        for (size_t c = 0; c < 9; ++c) H.addNZ(x_offset,     x_offset + c, perVertexHessian_x_x.col(c).head(c + 1));
        for (size_t c = 0; c < 2; ++c) H.addNZ(x_offset, theta_offset + c, perVertexHessian_x_theta.col(c));

        // Assemble upper triangle of theta-theta Hessian
        const Real_ coeff = 2.0 * inv_libar2 * m_twistingStiffness[i];
                   H.addNZ(theta_offset, theta_offset + 0,  coeff); // (theta i - 1, theta i - 1)
        auto idx = H.addNZ(theta_offset, theta_offset + 1, -coeff); // (theta i - 1, theta i    )
                   H.addNZ(idx                           ,  coeff); // (theta i    , theta i    )

        if (variableRestLen) {
            const Real_ deltaTwist = (dc.theta(i) - dc.theta(i - 1) + dc.referenceTwist[i] - m_restTwist[i]);
            const Real_ d2E_dljbar_dm = -2 * m_twistingStiffness[i] * inv_libar2 * inv_libar2 * deltaTwist;
            const Real_ d2E_dljbar_dljbar = 2 * m_twistingStiffness[i] * inv_libar2 * inv_libar2 * inv_libar2 * deltaTwist * deltaTwist;

            Vec3 d2E_dxim1_dlj = -(0.5 * d2E_dljbar_dm * (inv_len_im1            )) * dc.kb[i],
                 d2E_dxi_dlj   =  (0.5 * d2E_dljbar_dm * (inv_len_im1 - inv_len_i)) * dc.kb[i],
                 d2E_dxip1_dlj =  (0.5 * d2E_dljbar_dm * (inv_len_i              )) * dc.kb[i];

            const size_t rl_offset = theta_offset + ne; // index of the first rest length variable for the stencil

            // (x, restLen) terms
            for (size_t c = 0; c < 3; ++c) {
                H.addNZ(x_offset + 0 + c, rl_offset + 0, d2E_dxim1_dlj[c]); // (x i - 1, rl i - 1)
                H.addNZ(x_offset + 0 + c, rl_offset + 1, d2E_dxim1_dlj[c]); // (x i - 1, rl i    )

                H.addNZ(x_offset + 3 + c, rl_offset + 0, d2E_dxi_dlj  [c]); // (x i    , rl i - 1)
                H.addNZ(x_offset + 3 + c, rl_offset + 1, d2E_dxi_dlj  [c]); // (x i    , rl i    )

                H.addNZ(x_offset + 6 + c, rl_offset + 0, d2E_dxip1_dlj[c]); // (x i + 1, rl i - 1)
                H.addNZ(x_offset + 6 + c, rl_offset + 1, d2E_dxip1_dlj[c]); // (x i + 1, rl i    )
            }

            // (theta, restLen) terms
            H.addNZ(theta_offset + 0, rl_offset + 0, -d2E_dljbar_dm); // (theta i - 1, rl i - 1)
            H.addNZ(theta_offset + 0, rl_offset + 1, -d2E_dljbar_dm); // (theta i - 1, rl i    )
            H.addNZ(theta_offset + 1, rl_offset + 0,  d2E_dljbar_dm); // (theta i    , rl i - 1)
            H.addNZ(theta_offset + 1, rl_offset + 1,  d2E_dljbar_dm); // (theta i    , rl i    )

            // (restLen, restLen) terms (upper triangle)
            H.addNZ(rl_offset + 0, rl_offset + 0, d2E_dljbar_dljbar); // (rul i - 1, rl i - 1)
            H.addNZ(rl_offset + 0, rl_offset + 1, d2E_dljbar_dljbar); // (rul i - 1, rl i    )
            H.addNZ(rl_offset + 1, rl_offset + 1, d2E_dljbar_dljbar); // (rul i    , rl i    )
        }
    }
}

template<typename Real_>
void ElasticRod_T<Real_>::hessEnergy(ElasticRod_T<Real_>::CSCMat &H, bool variableRestLen) const {
    assert(H.symmetry_mode == CSCMat::SymmetryMode::UPPER_TRIANGLE);
    // BENCHMARK_START_TIMER_SECTION("ElasticRod hessEnergy()");
    hessEnergyStretch(H, variableRestLen);
    hessEnergyBend   (H, variableRestLen);
    hessEnergyTwist  (H, variableRestLen);
    // BENCHMARK_STOP_TIMER_SECTION("ElasticRod hessEnergy()");
}

template<typename Real_>
void ElasticRod_T<Real_>::massMatrix(ElasticRod_T<Real_>::CSCMat &M) const {
    assert(M.symmetry_mode == CSCMat::SymmetryMode::UPPER_TRIANGLE);

    using M3d = Eigen::Matrix<Real_, 3, 3>;
    // Per-edge contribution to the mass matrix:
    //        / [I/3  I/6  0]       [t^2 I    st I]    \
    //   int |  [I/6  I/3  0] + D^T [ st I   s^2 I] D   | dst * restLen * rho
    //    C   \ [  0    0  0]                          /
    // Where I is the 3x3 identity matrix, 0 is the 3x3 zero matrix, C is the cross-section geometry, (s, t) are
    // the integration coordinates over C, D is the Jacobian of material frame vectors d2 and d1 with respect to x_i, x_{i + 1}, and theta^i,
    // and rho is the density.
    //
    // D = ( -d d2/de, d d2/de, -d1)
    //     ( -d d1/de, d d1/de,  d2)
    //
    // This expression assumes that the rod is made up of a collection of rigid generalized cylinders, one per edge.
    // The edge is assumed to pass through through the cross-section's center of mass, and no mass is assigned to the
    // gaps formed as adjacent edges bend. Finally, we assume that the moment of inertia tensor,
    //      int_C [t^2   st] dst,
    //            [st   s^2]
    // has been diagonalized. Note that this moment of inertia tensor differs from the one determining the
    // bending stiffness, but actually they both have the same eigenvalues/diagonalized form.
    // Finally, to slightly simplify the formulas, we assume the source frame has been updated (i.e.
    // we use the infinitesimal transport formulas).
    const auto &dc = deformedConfiguration();
    const size_t nv = numVertices(), ne = numEdges();
    for (size_t j = 0; j < ne; ++j) {
        const auto &mat = material(j);

        Real_ restLenRho = m_restLen[j] * density(j);

        Real_ rest_len_div_len_sq = restLenRho / (dc.len[j] * dc.len[j]);
        Eigen::Matrix<Real_, 3, 3> M_xjxj   = M3d::Identity() * (mat.area * restLenRho / 3.0);
        Eigen::Matrix<Real_, 3, 3> M_xjxjp1 = M3d::Identity() * (mat.area * restLenRho / 6.0);

        const auto &d1 = dc.materialFrame[j].d1,
                   &d2 = dc.materialFrame[j].d2;

        // Contribution from material frame derivatives using (infinitesimal transport assumption).
        // Note: lambda_1 = int_omega t^2 dA is paired with d2, not d1, since 't' is the cross-section point coordinate along axis d2.
        M3d contrib = (mat.momentOfInertia.lambda_1 * rest_len_div_len_sq * d2) * d2.transpose()
                    + (mat.momentOfInertia.lambda_2 * rest_len_div_len_sq * d1) * d1.transpose();
        M_xjxj   += contrib;
        M_xjxjp1 -= contrib;

        // Assemble into global matrix.
        // x-x components:
        for (size_t c = 0; c < 3; ++c) {
                         M.addNZ(3 * (j    ), 3 * (j    ) + c, M_xjxj.col(c).head(c + 1));
            size_t idx = M.addNZ(3 * (j    ), 3 * (j + 1) + c, M_xjxjp1.col(c));
                         M.addNZ(idx,                          M_xjxj.col(c).head(c + 1));
        }

        // x-theta components are zero: perturbing a vertex rotates the director in the tangent direction, while
        // perturbing theta rotates the director perpendicularly to the tangent.

        // theta-theta component: sum_k lambda_k * d_dk_dtheta.squaredNorm() = sum_k lambda_k * 1
        M.addNZ(3 * nv + j, 3 * nv + j, restLenRho * mat.momentOfInertia.trace());
    }
}

template<typename Real_>
VecX_T<Real_> ElasticRod_T<Real_>::lumpedMassMatrix() const {
    VecX result(numDoF());
    result.setZero();
    // When summing the rows of the mass matrix corresponding to vertices, the
    // off-diagonal terms due to the material frame rotation cancel and we are
    // just left with the contribution from the scaled identity matrix blocks.
    // Summing these, we find that elements contribute half of their mass
    // restLen * area * rho to each vertex.
    const size_t ne = numEdges(), nv = numVertices();
    for (size_t j = 0; j < ne; ++j) {
        const auto &mat = material(j);
        result.template segment<3>(3 *  j     ) += 0.5 * m_restLen[j] * density(j) * mat.area * Vec3::Ones(3);
        result.template segment<3>(3 * (j + 1)) += 0.5 * m_restLen[j] * density(j) * mat.area * Vec3::Ones(3);
        result[3 * nv + j] +=                            m_restLen[j] * density(j) * mat.momentOfInertia.trace();
    }
    return result;
}

// Approximate the greatest velocity of any point in the rod induced by
// changing the parameters at rate paramVelocity.
// For a given edge j, point (alpha, s, t) on the generalized cylinder
// geometry moves at speed:
//      pdot(alpha, s, t) = (1 - alpha) xdot_j + alpha xdot_{j + 1} + s d1dot^j + t d2dot^j
// We wish to approximate
//      max_(alpha, s, t) ||pdot(alpha, s, t)||
// The function ||pdot|| is convex, so its maximum must be found at the boundary
// of the domain.
// We also assume that the source frame has been updated to simplify the formulas.
template<typename Real_>
Real_ ElasticRod_T<Real_>::approxLinfVelocity(const VecX_T<Real_> &paramVelocity) const {
    if (size_t(paramVelocity.size()) != numDoF()) throw std::runtime_error("Invalid parameter velocity size");
    const size_t nv = numVertices(), ne = numEdges();
    const auto &dc = deformedConfiguration();
    Real_ maxvel = 0;
    for (size_t j = 0; j < ne; ++j) {
        const auto &mat = material(j);
        const auto &t  = dc.tangent[j],
                   &d1 = dc.materialFrame[j].d1,
                   &d2 = dc.materialFrame[j].d2;
        std::array<Vec3, 2> xVel{{ paramVelocity.template segment<3>(3 * j), paramVelocity.template segment<3>(3 * (j + 1)) }},
                            d1Vel, d2Vel;
        Vec3 d1Dot = -t * (d1.dot(xVel[1] - xVel[0]) / dc.len[j]) + d2 * paramVelocity[3 * nv + j];
        Vec3 d2Dot = -t * (d2.dot(xVel[1] - xVel[0]) / dc.len[j]) - d1 * paramVelocity[3 * nv + j];
#if 0
        // Less accurate but cheaper approximation: check the 8 points
        //      alpha = +/-1, s = {smax, smin}, t = {tmax, tmin},
        // where smin/smax and tmin/tmax are from the cross-section geometry's bounding box.
        const auto &bb = mat.crossSectionBBox;
        d1Vel = {{ bb.minCorner[0] * d1Dot, bb.maxCorner[0] * d1Dot }};
        d2Vel = {{ bb.minCorner[1] * d2Dot, bb.maxCorner[1] * d2Dot }};
        for (size_t k = 0; k < 1 << 3; ++k)
            maxvel = std::max(maxvel, (xVel[k & 1] + d1Vel[(k >> 1) & 1] + d2Vel[k >> 2]).norm());
#else
        // Check each of the points of the visualization polygon
        for (size_t k = 0; k < 2; ++k) {
            for (const auto &pt : mat.crossSectionBoundaryPts) {
                maxvel = std::max(maxvel, (xVel[k] + pt[0] * d1Dot + pt[1] * d2Dot).norm());
            }
        }
#endif
    }

    return maxvel;
}

////////////////////////////////////////////////////////////////////////////////
// 1D uniform Laplacian regularization energy for the rest length optimization:
//      0.5 * sum_i (lbar^{i} - lbar^{i - 1})^2
////////////////////////////////////////////////////////////////////////////////
template<typename Real_>
Real_ ElasticRod_T<Real_>::restLengthLaplacianEnergy() const {
    const size_t nv = numVertices();
    Real_ result = 0;
    for (size_t i = 1; i < nv - 1; ++i) {
        Real_ diff = m_restLen[i] - m_restLen[i - 1];
        result += diff * diff;
    }
    return 0.5 * result;
}

template<typename Real_>
VecX_T<Real_> ElasticRod_T<Real_>::restLengthLaplacianGradEnergy() const {
    const size_t nv = numVertices();
    VecX result(VecX::Zero(numEdges()));
    for (size_t i = 1; i < nv - 1; ++i) {
        result[i    ] += m_restLen[i] - m_restLen[i - 1];
        result[i - 1] -= m_restLen[i] - m_restLen[i - 1];
    }
    return result;
}

template<typename Real_>
typename ElasticRod_T<Real_>::TMatrix ElasticRod_T<Real_>::restLengthLaplacianHessEnergy() const {
    const size_t ne = numEdges();
    std::vector<Real_> diag(ne, 2.0);
    diag.front() = diag.back() = 1.0;
    return triplet_matrix_from_tridiagonal(
            TriDiagonalSystem<Real_>(std::vector<Real_>(ne - 1, -1.0), std::move(diag),
                                     std::vector<Real_>(ne - 1, -1.0)));
}

////////////////////////////////////////////////////////////////////////////////
// Stress analysis
// See doc/structural_analysis.tex
////////////////////////////////////////////////////////////////////////////////
template<typename Real_>
Eigen::VectorXd ElasticRod_T<Real_>::stretchingStresses() const {
    const size_t ne = numEdges();
    Eigen::VectorXd result(ne);
    for (size_t j = 0; j < ne; ++j)
        result[j] = stripAutoDiff(material(j).youngModulus * (deformedConfiguration().len[j] / m_restLen[j] - 1.0));
    return result;
}

template<typename Real_>
Eigen::MatrixX2d ElasticRod_T<Real_>::bendingStresses() const {
    const size_t nv = numVertices();
    const auto &dc = deformedConfiguration();
    Eigen::MatrixX2d result(nv, 2);
    result.row(     0).setZero();
    result.row(nv - 1).setZero();
    for (size_t i = 1; i < nv - 1; ++i) {
        Real_ libar2 = m_restLen[i - 1] + m_restLen[i];
        Vec2 sigma = Vec2::Zero();
        for (size_t adj_edge = 0; adj_edge < 2; ++adj_edge) {
            Vec2 kappa = dc.per_corner_kappa[i].col(adj_edge);
            sigma += m_restLen[i - 1 + adj_edge] * material(i - 1 + adj_edge).bendingStresses(stripAutoDiff(kappa));
        }
        sigma /= (0.5 * libar2) * libar2;
        result.row(i) = stripAutoDiff(sigma).transpose();
    }
    return result;
}

template<typename Real_>
Eigen::VectorXd ElasticRod_T<Real_>::twistingStresses() const {
    const size_t nv = numVertices();
    const auto &dc = deformedConfiguration();
    Eigen::VectorXd result(nv);
    result[0] = result[nv - 1] = 0.0;
    for (size_t i = 1; i < nv - 1; ++i) {
        Real_ libar2 = m_restLen[i - 1] + m_restLen[i];
        double tau = std::fabs(stripAutoDiff((dc.theta(i) - dc.theta(i - 1) + dc.referenceTwist[i] - m_restTwist[i]) / (0.5 * libar2)));
        Real_ integratedStressCoeff = m_restLen[i - 1] * material(i - 1).torsionStressCoefficient
                                    + m_restLen[i    ] * material(i    ).torsionStressCoefficient;
        result[i] = tau * stripAutoDiff(integratedStressCoeff / libar2);
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////////
// Hessian matvec implementation
////////////////////////////////////////////////////////////////////////////////
#include "ElasticRodHessVec.inl"

////////////////////////////////////////////////////////////////////////////////
// Converting constructor for AD types
////////////////////////////////////////////////////////////////////////////////
template<typename TIn, typename AllocIn, typename TOut, typename AllocOut>
void castStdADVector(const std::vector<TIn, AllocIn> &in, std::vector<TOut, AllocOut> &out) {
    out.clear();
    out.reserve(in.size());
    for (const auto &val : in) out.push_back(autodiffCast<TOut>(val));
}

template<typename TIn, typename TOut>
void castStdADVectorDirectors(const std::vector<TIn> &in, std::vector<TOut> &out) {
    out.clear();
    out.reserve(in.size());
    for (const auto &val : in) out.emplace_back(val.d1, val.d2);
}

template<typename Real_>
template<typename Real2>
ElasticRod_T<Real_>::ElasticRod_T(const ElasticRod_T<Real2> &r) {
    std::vector<Pt3> rpts;
    castStdADVector(r.restPoints(), rpts);
    setRestConfiguration(rpts);

    castStdADVectorDirectors(r.restDirectors(), m_restDirectors);

    castStdADVector(r.restKappas(), m_restKappa);
    castStdADVector(r.restTwists(), m_restTwist);
    castStdADVector(r.restLengths(), m_restLen);

    setMaterial(r.edgeMaterials());

    setBendingStiffnesses(r.bendingStiffnesses());
    castStdADVector(r.twistingStiffnesses(), m_twistingStiffness);
    castStdADVector(r.stretchingStiffnesses(), m_stretchingStiffness);

    setBendingEnergyType(static_cast<BendingEnergyType>(r.bendingEnergyType()));

    castStdADVector(r.densities(), m_density);
    setInitialMinRestLen(r.initialMinRestLength());

    auto &dc_in = r.deformedConfiguration();
    DeformedState &dc_out = deformedConfiguration();

    castStdADVector         (dc_in.sourceTangent, dc_out.sourceTangent);
    castStdADVectorDirectors(dc_in.sourceReferenceDirectors, dc_out.sourceReferenceDirectors);
    castStdADVector         (dc_in.sourceTheta, dc_out.sourceTheta);
    castStdADVector         (dc_in.sourceReferenceTwist, dc_out.sourceReferenceTwist);

    std::vector<Pt3> pts;
    std::vector<Real_> thetas;
    castStdADVector(dc_in.points(), pts);
    castStdADVector(dc_in.thetas(), thetas);
    dc_out.update(pts, thetas);
}

////////////////////////////////////////////////////////////////////////////////
// Explicit instantiation for ordinary double type and autodiff type.
////////////////////////////////////////////////////////////////////////////////
template struct ElasticRod_T<  Real>;
template struct ElasticRod_T<ADReal>;
template ElasticRod_T<ADReal>::ElasticRod_T(const ElasticRod_T<Real> &);

// Instantiations for the non-default gradient stencil types.
template ElasticRod_T<  Real>::Gradient ElasticRod_T<  Real>::gradient<GradientStencilMaskCustom       >(bool, ElasticRod_T<  Real>::EnergyType, bool, bool, const GradientStencilMaskCustom        &) const;
template ElasticRod_T<ADReal>::Gradient ElasticRod_T<ADReal>::gradient<GradientStencilMaskCustom       >(bool, ElasticRod_T<ADReal>::EnergyType, bool, bool, const GradientStencilMaskCustom        &) const;
template ElasticRod_T<  Real>::Gradient ElasticRod_T<  Real>::gradient<GradientStencilMaskTerminalsOnly>(bool, ElasticRod_T<  Real>::EnergyType, bool, bool, const GradientStencilMaskTerminalsOnly &) const;
template ElasticRod_T<ADReal>::Gradient ElasticRod_T<ADReal>::gradient<GradientStencilMaskTerminalsOnly>(bool, ElasticRod_T<ADReal>::EnergyType, bool, bool, const GradientStencilMaskTerminalsOnly &) const;

// The following annoying explicit instantiations really shouldn't be needed,
// but due to an apparent bug, GCC 8.3 (the default on Ubuntu 19.04) fails to
// instantiate them automatically...
template ElasticRod_T<  Real>::Gradient ElasticRod_T<  Real>::gradient<GradientStencilMaskIncludeAll   >(bool, ElasticRod_T<  Real>::EnergyType, bool, bool, const GradientStencilMaskIncludeAll    &) const;
template ElasticRod_T<ADReal>::Gradient ElasticRod_T<ADReal>::gradient<GradientStencilMaskIncludeAll   >(bool, ElasticRod_T<ADReal>::EnergyType, bool, bool, const GradientStencilMaskIncludeAll    &) const;

template ElasticRod_T<  Real>::Gradient ElasticRod_T<  Real>::gradEnergyStretch<GradientStencilMaskIncludeAll>(      bool, bool, const GradientStencilMaskIncludeAll &) const;
template ElasticRod_T<  Real>::Gradient ElasticRod_T<  Real>::gradEnergyBend   <GradientStencilMaskIncludeAll>(bool, bool, bool, const GradientStencilMaskIncludeAll &) const;
template ElasticRod_T<  Real>::Gradient ElasticRod_T<  Real>::gradEnergyTwist  <GradientStencilMaskIncludeAll>(bool, bool, bool, const GradientStencilMaskIncludeAll &) const;
template ElasticRod_T<  Real>::Gradient ElasticRod_T<  Real>::gradEnergy       <GradientStencilMaskIncludeAll>(bool, bool, bool, const GradientStencilMaskIncludeAll &) const;
template ElasticRod_T<ADReal>::Gradient ElasticRod_T<ADReal>::gradEnergyStretch<GradientStencilMaskIncludeAll>(      bool, bool, const GradientStencilMaskIncludeAll &) const;
template ElasticRod_T<ADReal>::Gradient ElasticRod_T<ADReal>::gradEnergyBend   <GradientStencilMaskIncludeAll>(bool, bool, bool, const GradientStencilMaskIncludeAll &) const;
template ElasticRod_T<ADReal>::Gradient ElasticRod_T<ADReal>::gradEnergyTwist  <GradientStencilMaskIncludeAll>(bool, bool, bool, const GradientStencilMaskIncludeAll &) const;
template ElasticRod_T<ADReal>::Gradient ElasticRod_T<ADReal>::gradEnergy       <GradientStencilMaskIncludeAll>(bool, bool, bool, const GradientStencilMaskIncludeAll &) const;

template ElasticRod_T<  Real>::Gradient ElasticRod_T<  Real>::gradEnergyStretch<GradientStencilMaskTerminalsOnly>(      bool, bool, const GradientStencilMaskTerminalsOnly &) const;
template ElasticRod_T<  Real>::Gradient ElasticRod_T<  Real>::gradEnergyBend   <GradientStencilMaskTerminalsOnly>(bool, bool, bool, const GradientStencilMaskTerminalsOnly &) const;
template ElasticRod_T<  Real>::Gradient ElasticRod_T<  Real>::gradEnergyTwist  <GradientStencilMaskTerminalsOnly>(bool, bool, bool, const GradientStencilMaskTerminalsOnly &) const;
template ElasticRod_T<  Real>::Gradient ElasticRod_T<  Real>::gradEnergy       <GradientStencilMaskTerminalsOnly>(bool, bool, bool, const GradientStencilMaskTerminalsOnly &) const;
template ElasticRod_T<ADReal>::Gradient ElasticRod_T<ADReal>::gradEnergyStretch<GradientStencilMaskTerminalsOnly>(      bool, bool, const GradientStencilMaskTerminalsOnly &) const;
template ElasticRod_T<ADReal>::Gradient ElasticRod_T<ADReal>::gradEnergyBend   <GradientStencilMaskTerminalsOnly>(bool, bool, bool, const GradientStencilMaskTerminalsOnly &) const;
template ElasticRod_T<ADReal>::Gradient ElasticRod_T<ADReal>::gradEnergyTwist  <GradientStencilMaskTerminalsOnly>(bool, bool, bool, const GradientStencilMaskTerminalsOnly &) const;
template ElasticRod_T<ADReal>::Gradient ElasticRod_T<ADReal>::gradEnergy       <GradientStencilMaskTerminalsOnly>(bool, bool, bool, const GradientStencilMaskTerminalsOnly &) const;

template ElasticRod_T<  Real>::Gradient ElasticRod_T<  Real>::gradEnergyStretch<GradientStencilMaskCustom>(      bool, bool, const GradientStencilMaskCustom &) const;
template ElasticRod_T<  Real>::Gradient ElasticRod_T<  Real>::gradEnergyBend   <GradientStencilMaskCustom>(bool, bool, bool, const GradientStencilMaskCustom &) const;
template ElasticRod_T<  Real>::Gradient ElasticRod_T<  Real>::gradEnergyTwist  <GradientStencilMaskCustom>(bool, bool, bool, const GradientStencilMaskCustom &) const;
template ElasticRod_T<  Real>::Gradient ElasticRod_T<  Real>::gradEnergy       <GradientStencilMaskCustom>(bool, bool, bool, const GradientStencilMaskCustom &) const;
template ElasticRod_T<ADReal>::Gradient ElasticRod_T<ADReal>::gradEnergyStretch<GradientStencilMaskCustom>(      bool, bool, const GradientStencilMaskCustom &) const;
template ElasticRod_T<ADReal>::Gradient ElasticRod_T<ADReal>::gradEnergyBend   <GradientStencilMaskCustom>(bool, bool, bool, const GradientStencilMaskCustom &) const;
template ElasticRod_T<ADReal>::Gradient ElasticRod_T<ADReal>::gradEnergyTwist  <GradientStencilMaskCustom>(bool, bool, bool, const GradientStencilMaskCustom &) const;
template ElasticRod_T<ADReal>::Gradient ElasticRod_T<ADReal>::gradEnergy       <GradientStencilMaskCustom>(bool, bool, bool, const GradientStencilMaskCustom &) const;
