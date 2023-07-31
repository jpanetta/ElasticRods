#include "RodLinkage.hh"
#include "LinkageTerminalEdgeSensitivity.hh"
#include <queue>
#include <map>
#include <algorithm>
#include <iterator>
#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/MSHFieldWriter.hh>
#include <MeshFEM/utils.hh>
#include <MeshFEM/filters/merge_duplicate_vertices.hh>
#include <MeshFEM/Geometry.hh>
#if MESHFEM_WITH_TBB
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>
#endif

#include <MeshFEM/unused.hh>

template<typename Real_>
void RodLinkage_T<Real_>::read(const std::string &path, size_t subdivision, bool initConsistentAngle) {
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    MeshIO::load(path, vertices, elements);
    set(vertices, elements, subdivision, initConsistentAngle);
}

template<typename Real_>
void RodLinkage_T<Real_>::set(std::vector<MeshIO::IOVertex > vertices, // copy edited inside
                              std::vector<MeshIO::IOElement> edges,    // copy edited inside
                              size_t subdivision,
                              bool initConsistentAngle) {
    {
        const size_t old_nv = vertices.size();
        merge_duplicate_vertices(vertices, edges, vertices, edges, 1e-10);
        const size_t nv = vertices.size();
        if (nv != old_nv) std::cerr << "WARNING: merged " << old_nv - nv << " duplicated vertices" << std::endl;
    }

    const size_t nv = vertices.size(),
                 ne = edges.size();
    m_segments.clear();
    m_joints.clear();
    m_segments.reserve(edges.size());
    m_joints.reserve(vertices.size());
    std::vector<size_t> valence(nv);
    // Note: only the first valence[vi] entries of incidentEdges are used; the
    // rest are left uninitialized
    std::vector<std::array<size_t, 4>> incidentEdges(nv);

    // Determine vertex-edge connectivity;
    for (size_t i = 0; i < ne; ++i) {
        const auto &e = edges[i];
        if (e.size() != 2) throw std::runtime_error("Invalid element; all elements must be lines");

        incidentEdges[e[0]].at(valence.at(e[0])++) = i;
        incidentEdges[e[1]].at(valence.at(e[1])++) = i;
    }

    // Generate a rod segment for each edge.
    for (const auto &e : edges) {
        m_segments.emplace_back(vertices[e[0]].point,
                                vertices[e[1]].point,
                                subdivision);
    }

    // Generate joints at the valence 2, 3, and 4 vertices.
    size_t firstJointVtx = NONE; // Index of a vertex corresponding to a joint (used to initiate BFS below)
    std::vector<size_t> jointForVertex(nv, size_t(NONE));
    for (size_t vi = 0; vi < nv; ++vi) {
        const size_t jointValence = valence[vi];
        if (jointValence == 1) continue; // free end; no joint
        if (jointValence  > 4) throw std::runtime_error("Invalid vertex valence " + std::to_string(valence[vi]) + "; must be 1, 2, 3, or 4");

        // Valence 2, 3, or 4:
        if (firstJointVtx == NONE) firstJointVtx = vi;
        // Group the incident edges into pairs that connect to form
        // mostly-straight rods
        // Do this by considering the *outward-pointing* edge vectors:
        std::array<Vec3,  4> edgeVecs;
        std::array<Real_, 4> edgeVecLens;
        std::array<bool,  4> isStartPt;
        for (size_t k = 0; k < jointValence; ++k) {
            const auto &e = edges.at(incidentEdges[vi][k]);
            edgeVecs[k] = vertices[e[1]].point - vertices[e[0]].point;
            edgeVecLens[k] = edgeVecs[k].norm();
            edgeVecs[k] /= edgeVecLens[k];
            isStartPt[k] = (e[0] == vi);
            if (isStartPt[k]) continue;
            assert(e[1] == vi);
            edgeVecs[k] *= -1.0;
        }

        // Partition the segments into those forming "Rod A" and those forming "Rod B"
        std::array<size_t, 2> segmentsA{{NONE, NONE}}, segmentsB{{NONE, NONE}};
        size_t numA = 0, numB = 0;
        std::array<bool, 2> isStartA{{false, false}}, isStartB{{false, false}};

        if (jointValence == 2) {
            // There are no continuation edges if the valence is 2; one segment belongs to "Rod A" and the other to "Rod B"
            // No terminal edge averaging needs to be done.
            numA = numB = 1;
            segmentsA[0] = 0, segmentsB[0] = 1;
        }
        if (jointValence == 3) {
            // Determine which two of the 3 incident edges best connect to form Rod A
            // (Try to pick the two that connect the straightest, but verify that this
            //  preserves a quad topology; if not, the straigtest valid connection must be made.)
            Real_ minCosTheta = safe_numeric_limits<Real_>::max();
            for (size_t j = 0; j < jointValence; ++j) {
                for (size_t k = j + 1; k < jointValence; ++k) {
                    Real_ cosTheta = edgeVecs[j].dot(edgeVecs[k]);
                    if (cosTheta < minCosTheta) {
                        // Check if connecting edges (j, k) creates a triangle
                        // instead of a quad. This happens if the joints
                        // connected by edges j and k have neighborhoods that
                        // share more than "vi" in commmon.
                        auto get_neighbor = [&edges,&incidentEdges](size_t v, size_t local_eidx) {
                            const auto &e = edges[incidentEdges[v][local_eidx]];
                            if (e[0] == v) return e[1];
                            if (e[1] == v) return e[0];
                            throw std::runtime_error("Edge is not incident v!");
                        };

                        const size_t vj = get_neighbor(vi, j),
                                     vk = get_neighbor(vi, k);
                        std::vector<size_t> nj, nk;
                        for (size_t i = 0; i < valence[vj]; ++i) { nj.push_back(get_neighbor(vj, i)); }
                        for (size_t i = 0; i < valence[vk]; ++i) { nk.push_back(get_neighbor(vk, i)); }
                        std::sort(nj.begin(), nj.end());
                        std::sort(nk.begin(), nk.end());
                        std::vector<size_t> nboth;
                        std::set_intersection(nj.begin(), nj.end(), nk.begin(), nk.end(), std::back_inserter(nboth));
                        if (nboth.size() > 1) continue; // connecting (j, k) forms a triangle; forbid it.

                        minCosTheta = cosTheta;
                        segmentsA[0] = j; segmentsA[1] = k;
                    }
                }
            }
            if (segmentsA[0] == NONE) throw std::runtime_error("Failed to link up valence 3 vertex (without creating triangles)");
            numA = 2; numB = 1;
            segmentsB[0] = 3 - (segmentsA[0] + segmentsA[1]); // all indices add up to 3; complement by subtraction
        }
        if (jointValence == 4) {
            // Order the edges clockwise around the joint normal and assign them alternating rod labels A, B, A, B.
            // First choose a joint normal by crossing the two vectors that are closest to perpendicular
            Vec3 n = Vec3::Zero();
            for (size_t k = 1; k < jointValence; ++k) {
                Vec3 a2 = edgeVecs[k].cross(edgeVecs[0]);
                if (a2.squaredNorm() > n.squaredNorm()) n = a2;
            }
            n.normalize();

            // Next, compute the angles between edge 0 and every other edge.
            std::array<Real_, 3> angles;
            Vec3 v0 = edgeVecs[0] - n * n.dot(edgeVecs[0]);
            for (size_t k = 1; k < jointValence; ++k) {
                Vec3 vk = edgeVecs[k] - n * n.dot(edgeVecs[k]);
                Real_ theta = angle(n, v0, vk); // angle in [-pi, pi]
                if (theta < 0) theta += 2.0 * M_PI; // compute angle in [0, 2 pi]
                angles[k - 1] = theta;
            }
            // Sort vectors 1, 2, 3  clockwise (ascending angle wrt vector 0), assign alternating labels
            std::vector<size_t> p = sortPermutation(angles); // sorted list: [angles[p[0]], angles[p[1]], angles[p[2]]]
            segmentsA = {{   0,  1 + p[1] }};
            segmentsB = {{ 1 + p[0], 1 + p[2] }};

            numA = numB = 2;
        }

        for (size_t k = 0; k < numA; ++k) isStartA[k] = isStartPt[segmentsA[k]];
        for (size_t k = 0; k < numB; ++k) isStartB[k] = isStartPt[segmentsB[k]];

        // Determine this joint's edge vectors for rod A and rod B. If there's only one
        // segment for a rod A or B at this joint, then the terminal edge of this segment gives this
        // vector (up to sign). If there are two connecting segments, we must construct an averaged vector.
        // In all cases, we construct the edge that points out of segment 0 and into segment 1.
        // This vector is scaled to be the smallest of the two participating edges (to prevent inversions of the adjacent rod edges).
        // Note: this averaging/scaling operation will change the rest
        // length of the neighboring edges, so rod segments' rest lengths
        // will need to be recomputed.
        Vec3 edgeA = -edgeVecs[segmentsA[0]], edgeB = -edgeVecs[segmentsB[0]]; // get vector pointing out of segment 0
        Real_ segmentFracLen = 1.0 / (subdivision - 1); // only (subdivision - 1) segment lengths fit between the endpoints; rod extends half a segment past each endpoint.
        Real_ lenA  = edgeVecLens[segmentsA[0]] * segmentFracLen,
              lenB  = edgeVecLens[segmentsB[0]] * segmentFracLen;
        if (numA == 2) {
            lenA  = std::min<Real_>(lenA, edgeVecLens[segmentsA[1]] * segmentFracLen);
            edgeA += edgeVecs[segmentsA[1]];
        }
        if (numB == 2) {
            lenB  = std::min<Real_>(lenB, edgeVecLens[segmentsB[1]] * segmentFracLen);
            edgeB += edgeVecs[segmentsB[1]];
        }
        edgeA *= lenA / edgeA.norm();
        edgeB *= lenB / edgeB.norm();

        // Convert to global segment indices
        for (size_t k = 0; k < numA; ++k) segmentsA[k] = incidentEdges[vi][segmentsA[k]];
        for (size_t k = 0; k < numB; ++k) segmentsB[k] = incidentEdges[vi][segmentsB[k]];

        const size_t ji = m_joints.size();
        jointForVertex[vi] = ji;
        m_joints.emplace_back(this, vertices[vi].point, edgeA, edgeB, segmentsA, segmentsB, isStartA, isStartB);

        // Link the incident segments to this joint.
        for (size_t k = 0; k < jointValence; ++k) {
            auto &s = m_segments.at(incidentEdges[vi][k]);
            if (isStartPt[k]) s.startJoint = ji;
            else              s.endJoint   = ji;
        }
    }

    if (firstJointVtx == NONE)
        throw std::runtime_error("There must be at least one joint in the network");

    // Propagate consistent joint normals throughout the graph using BFS
    // (to prevent 180 degree twists); warn if the graph is disconnected.
    {
        std::queue<size_t> bfsQueue;
        std::vector<bool> visited(nv, false);
        visited[firstJointVtx] = true;
        bfsQueue.push(firstJointVtx);
        size_t numVisited = 1;
        while (!bfsQueue.empty()) {
            size_t u = bfsQueue.front();
            size_t ju = jointForVertex[u];
            assert(ju != NONE);
            bfsQueue.pop();
            for (size_t k = 0; k < valence[u]; ++k) {
                const auto &e = edges.at(incidentEdges[u][k]);
                assert((e[0] == u) != (e[1] == u));
                const size_t v = (e[0] == u) ? e[1] : e[0];

                if (visited.at(v)) continue;
                visited[v] = true;
                ++numVisited;
                size_t jv = jointForVertex[v];
                if (jv == NONE) continue; // terminate search at valence 1 vertices
                m_joints.at(jv).makeNormalConsistent(m_joints.at(ju));
                bfsQueue.push(v);
            }
        }
        if (numVisited != nv) throw std::runtime_error("Disconnected edge graph");
    }

    // Propagate consistent joint opening angle definitions throughout the graph:
    //      +----------------------+
    //      |  B\.-./A     A\   /  |
    //      |    \ /         \ /   |
    //      |     X     vs  ( X    |
    //      |    / \         / \   |
    //      |   /   \      B/   \  |
    //      +----------------------+
    // This is equivalent to ensuring the "A" rods of one joint connect with the
    // "A" rods of its neighbors. This angle consistency is needed so that all
    // angles change in the same direction during deployment/closing (permitting
    // actuation by an average target angle constraint).
    // We pick the definition that makes the majority of angles acute.
    {
        // Try to make all joints consistent with the first.
        std::queue<size_t> bfsQueue;
        std::vector<size_t> prev(numJoints(), size_t(NONE)); // visited array, also allowing path recovery for debugging
        prev[0] = 0;
        bfsQueue.push(0);
        while (!bfsQueue.empty()) {
            size_t u = bfsQueue.front();
            bfsQueue.pop();
            const auto &ju = m_joints.at(u);
            for (size_t AB_u = 0; AB_u < 2; ++AB_u) { // Separately visit neighbors connected along "A" (AB_u = 0) and "B" (AB_u = 1)
                ju.visitNeighbors(
                    [&](size_t v, size_t /* si */, size_t AB_v) {
                        bool consistent = AB_u == AB_v;
                        if (prev[v] != NONE) {
                            // joint "v" has already been visited/set. If it
                            // is not already consistent, our strategy has failed.
                            if (!consistent && initConsistentAngle) {
                                // Output debugging information about the two inconsistent BFS paths.
                                auto reportPath = [&](size_t v) {
                                    while (v != 0) {
                                        std::cout << v;
                                        size_t p = prev.at(v);
                                        const auto &jv = m_joints.at(v);
                                        jv.visitNeighbors([p,&jv](size_t ji, size_t si, size_t AB) { if (ji == p) std::cout << "--" << si << "(" << char(jv.segmentABOffset(si) + 'A') << ", " << char(AB + 'A') << ")--"; });
                                        v = p;
                                    }
                                    std::cout << 0 << std::endl;
                                };
                                std::cout << "Trying to set " << v << " from path:" << std::endl;
                                reportPath(u);
                                std::cout << "Inconsistent with earlier path:" << std::endl;
                                reportPath(v);
                                throw std::runtime_error("Propagating consistent angle definitions failed");
                            }
                            return;
                        }

                        prev[v] = u;
                        bfsQueue.push(v);
                        if (!consistent) m_joints.at(v).swapAngleDefinition();
                }, AB_u);
            }
        }

        // Choose joint definitions so that the majority are acute.
        size_t numAcute = 0;
        for (const auto &ju : m_joints) {
            if (ju.alpha() < 0) throw std::runtime_error("Negative joint angle");
            if (ju.alpha() < M_PI / 2) ++numAcute;
        }

        if (numAcute < (numJoints() - numAcute))
            for (auto &ju : m_joints) ju.swapAngleDefinition();
    }

    ////////////////////////////////////////////////////////////////////////////
    // Rest length/point spacing initialization
    ////////////////////////////////////////////////////////////////////////////
    // Initial guess for the length of each segment: straight line distance
    VecX segmentRestLenGuess(ne);
    for (size_t i = 0; i < ne; ++i) {
        const auto &e = edges[i];
        segmentRestLenGuess[i] = (vertices[e[1]].point - vertices[e[0]].point).norm();
    }

    // Build the segment len->edge len map and use it to construct
    // the rest length guess for every edge in the network
    m_constructSegmentRestLenToEdgeRestLenMapTranspose(segmentRestLenGuess);
    m_perSegmentRestLen = segmentRestLenGuess;
    m_setRestLengthsFromPSRL();

    // Initialize the DoF offset table (no constraints yet).
    m_buildDoFOffsets();

    // Also set a reasonable initialization for the deformed configuration
    auto params = getDoFs();
    setDoFs(params, true /* set spatially coherent thetas */);

    // The terminal edges of each segment have been twisted to conform to
    // the joints, but the internal edges are in their default orientation.
    // We update the edges' material axes (thetas) by minimizing the twist
    // energy with respect to the thetas.
    for (auto &s : m_segments)
        s.setMinimalTwistThetas();

    // Update the "source thetas" used to maintain temporal coherence
    updateSourceFrame();

    // Use the linkage's cached material to initialize each rod's material.
    setMaterial(m_material);

    m_initMinRestLen = minRestLength();

    m_clearCache();
}

template<typename Real_>
void RodLinkage_T<Real_>::m_setRestLengthsFromPSRL() {
    if (m_segmentRestLenToEdgeRestLenMapTranspose.m == 0) throw std::runtime_error("Must run m_constructSegmentRestLenToEdgeRestLenMapTranspose first");
    VecX restLens = m_segmentRestLenToEdgeRestLenMapTranspose.apply(m_perSegmentRestLen, /* transpose */ true);
    // {
    //     std::ofstream rlens_file("rlens.txt");
    //     rlens_file.precision(16);
    //     rlens_file << restLens << std::endl;
    // }

    // Apply these rest lengths to the linkage.
    size_t offset = 0;
    for (auto &s : m_segments) {
        const size_t ne = s.rod.numEdges();
        // Visit each internal/free edge:
        for (size_t ei = s.hasStartJoint(); ei < (s.hasEndJoint() ? ne - 1 : ne); ++ei)
            s.rod.restLengthForEdge(ei) = restLens[offset++];
    }
    // {
    //     std::ofstream rlens_file("pre_joint_edge_rlens.txt");
    //     rlens_file.precision(16);
    //     for (auto &s : m_segments) {
    //         for (auto rl : s.rod.restLengths())
    //             rlens_file << rl << std::endl;
    //     }
    // }
    // std::ofstream jdebug_file("joint_len_debug.txt");
    for (auto &j : m_joints) {
        // jdebug_file << "Setting rlens " << restLens.template segment<2>(offset).transpose() << std::endl;
        // jdebug_file << "segmentsA: " << j.segmentsA()[0] << " " << j.segmentsA()[1] << ",  segmentsB:" << j.segmentsB()[0] << " " << j.segmentsB()[1] << "\t";
        // jdebug_file << "isStartA:  " << j.isStartA() [0] << " " << j.isStartA() [1] << ",  isStartB: " << j.isStartB() [0] << " " << j.isStartB() [1] << std::endl;
        j.setRestLengths(restLens.template segment<2>(offset));
        offset += 2;
    }

    // {
    //     std::ofstream rlens_file("edge_rlens.txt");
    //     rlens_file.precision(16);
    //     for (auto &s : m_segments) {
    //         for (auto rl : s.rod.restLengths())
    //             rlens_file << rl << std::endl;
    //     }
    // }

    assert(offset = size_t(restLens.size()));
}

template<typename Real_>
void RodLinkage_T<Real_>::m_buildDoFOffsets() {
    m_dofOffsetForSegment.resize(m_segments.size());
    m_dofOffsetForJoint.resize(m_joints.size());
    size_t offset = 0;
    for (size_t i = 0; i < numSegments(); ++i) {
        m_dofOffsetForSegment[i] = offset;
        offset += m_segments[i].numDoF();
    }
    for (size_t i = 0; i < numJoints(); ++i) {
        m_dofOffsetForJoint[i] = offset;
        offset += m_joints[i].numDoF();
    }

    m_restLenDoFOffsetForSegment.resize(m_segments.size());
    for (size_t i = 0; i < numSegments(); ++i) {
        m_restLenDoFOffsetForSegment[i] = offset;
        offset += m_segments[i].numFreeEdges();
    }
    m_restLenDoFOffsetForJoint.resize(m_joints.size());
    for (size_t i = 0; i < numJoints(); ++i) {
        m_restLenDoFOffsetForJoint[i] = offset;
        offset += 2;
    }
}

// Construct the *transpose* of the map from a vector holding the (rest) lengths
// of each segment to a vector holding a (rest) length for every rod length in the
// entire network. The vector output by this map is ordered as follows: all
// lengths for segments' interior and free edges, followed by two lengths for each joint.
// (We use build the transpose instead of the map itself to efficiently support
// the iteration needed to assemble the Hessian chain rule term)
// This is a fixed linear map for the lifetime of the linkage, though
// it depends on the initial distribution of segment lengths:
// to prevent edge "flips" when a long edge meets a short edge at a joint, we
// use the minimum of the two lengths to define the joint edge length. To
// prevent the map from being non-differentiable, we decide at linkage
// construction time which the "short" edge is. (Another solution would be to
// use a soft minimum, but this would require computing an additional Hessian
// term). We finally space the remaining length evenly across the unconstrained
// edges.
template<typename Real_>
void RodLinkage_T<Real_>::m_constructSegmentRestLenToEdgeRestLenMapTranspose(const VecX_T<Real_> &segmentRestLenGuess) {
    assert(size_t(segmentRestLenGuess.size()) == numSegments());
    // Get the initial ideal rest length for the edges of each segment; this is
    // used to decide which segments control which terminal edges.
    VecX idealEdgeLenForSegment(numSegments());
    std::vector<size_t> numEdgesForSegment(numSegments());
    for (size_t si = 0; si < numSegments(); ++si) {
        numEdgesForSegment[si] = segment(si).rod.numEdges();
        idealEdgeLenForSegment[si] = segmentRestLenGuess[si] / (numEdgesForSegment[si] - 1.0);
    }

    // Decide who controls each joint edge: the shorter ideal rest length
    // wins. Ties are broken arbitrarily.
    std::vector<std::array<size_t, 2>> controllersForJoint(numJoints());
    for (size_t ji = 0; ji < numJoints(); ++ji) {
        const auto &j = joint(ji);
        auto &c = controllersForJoint[ji];
        const auto &sA = j.segmentsA(); const auto &sB = j.segmentsB();
        c[0] = sA[0];
        c[1] = sB[0];
        if ((sA[1] != NONE) && (idealEdgeLenForSegment[sA[1]] < idealEdgeLenForSegment[c[0]])) c[0] = sA[1];
        if ((sB[1] != NONE) && (idealEdgeLenForSegment[sB[1]] < idealEdgeLenForSegment[c[1]])) c[1] = sB[1];
    }

    // Determine the number of nonzeros in the map.
    // Each free edge in a segment segment is potentially influenced by segment
    // lengths in the stencil:
    //      +-----+-----+-----+
    //               ^
    // (The segment always influences its own edges, but neighbors controlling the incident
    // joints influence the edges too).
    size_t totalFreeEdges = 0;
    size_t nz = 0;
    // Count the entries in the columns corresponding to segments' internal/free ends
    for (size_t si = 0; si < numSegments(); ++si) {
        const auto &s = segment(si);
        const size_t numFreeEdges = numEdgesForSegment[si] - s.hasStartJoint() - s.hasEndJoint();
        totalFreeEdges += numFreeEdges;

        nz += numFreeEdges; // The segment influences all of its own free edges.
        // A controlling neighbor also influences all of the free edges:
        auto processJoint = [&](size_t ji) {
            if (ji == NONE) return;
            const auto &j = joint(ji);
            size_t controller = controllersForJoint[ji][j.segmentABOffset(si)];
            assert(controller != NONE);
            if (controller != si) nz += numFreeEdges;
        };
        processJoint(s.startJoint);
        processJoint(s.endJoint);
    }
    // The two columns for each joint have only a single entry (one controlling segment)
    nz += numJoints() * 2;

    const SuiteSparse_long m = numSegments(), n = totalFreeEdges + 2 * numJoints();
    CSCMat result(m, n);
    result.nz = nz;

    // Now we fill out the transpose of the map one column (edge) at a time:
    //    #     [               ]
    // segments [               ]
    //              # edges
    auto &Ai = result.Ai;
    auto &Ax = result.Ax;
    auto &Ap = result.Ap;

    Ai.reserve(nz);
    Ax.reserve(nz);
    Ap.reserve(n + 1);

    Ap.push_back(0); // col 0 begin

    // Segments are split into (ne - 1) intervals spanning between
    // the incident joint positions (graph nodes); half an interval
    // extends past the joints at each end.
    // Joints control the lengths of the intervals surrounding them,
    // specifying the length of half a subdivision interval on the incident
    // segments. The remaining length of each segment is then
    // distributed evenly across the "free" intervals.
    // First, build the columns for the free edges of each segment:
    for (size_t si = 0; si < numSegments(); ++si) {
        const auto &s = segment(si);
        // Determine the influencers for each internal/free edge length on this segment.
        struct Influence {
            size_t idx = NONE;
            Real_ val = 0;
            bool operator<(const Influence &b) const { return idx < b.idx; }
        };
        std::array<Influence, 3> infl;
        const size_t ne = numEdgesForSegment[si];
        const Real_ numFreeIntervals = (ne - 1) - 0.5 * (s.hasEndJoint() + s.hasStartJoint());
        infl[0].idx = si;
        infl[0].val = 1.0 / numFreeIntervals; // length is distributed evenly across the free intervals

        // The incident joint edges subtract half their length from the amount
        // distributed to the free intervals.
        auto processJoint = [&](size_t lji) {
            size_t ji = s.joint(lji);
            if (ji == NONE) return;
            const auto &j = joint(ji);
            size_t c = controllersForJoint[ji][j.segmentABOffset(si)];
            assert(c != NONE);
            if (c == si) { infl[0].val -= (0.5 * (1.0 / (ne - 1))) / numFreeIntervals; return; }
            infl[lji + 1].idx = c;
            infl[lji + 1].val = -(0.5 * (1.0 / (numEdgesForSegment[c] - 1))) / numFreeIntervals;
        };
        processJoint(0);
        processJoint(1);

        std::sort(infl.begin(), infl.end());

        // Visit each internal/free edge:
        for (size_t ei = s.hasStartJoint(); ei < (s.hasEndJoint() ? ne - 1 : ne); ++ei) {
            // Add entries for each present influencer.
            for (size_t i = 0; i < 3; ++i) {
                if (infl[i].idx == NONE) continue;
                Ai.push_back(infl[i].idx);
                Ax.push_back(infl[i].val);
            }
            Ap.push_back(Ai.size()); // col end
        }
    }

    // Build the columns for the joint edges
    for (size_t ji = 0; ji < numJoints(); ++ji) {
        for (size_t ab = 0; ab < 2; ++ab) {
            const size_t c = controllersForJoint[ji][ab];
            Ai.push_back(c);
            Ax.push_back(1.0 / (numEdgesForSegment[c] - 1));
            Ap.push_back(Ai.size()); // col end
        }
    }

    assert(Ax.size() == size_t(nz   ));
    assert(Ai.size() == size_t(nz   ));
    assert(Ap.size() == size_t(n + 1));

    m_segmentRestLenToEdgeRestLenMapTranspose = std::move(result);
}

template<typename Real_>
void RodLinkage_T<Real_>::setMaterial(const RodMaterial &material) {
    m_material = material;

    for (auto &s : m_segments) {
        auto &rod = s.rod;
        rod.setMaterial(material);

        // Avoid double-counting stiffness/mass for edges shared at the joints.
        if (s.startJoint != NONE) rod.density(0) = 0.5;
        if (s.  endJoint != NONE) rod.density(rod.numEdges() - 1) = 0.5;
    }
}

template<typename Real_>
void RodLinkage_T<Real_>::setStretchingStiffness(Real_ val) {
    for (auto &s : m_segments) {
        auto &rod = s.rod;
        const size_t ne = rod.numEdges();
        for (size_t j = 0; j < ne; ++j)
            rod.stretchingStiffness(j) = val;
    }
}

template<typename Real_>
size_t RodLinkage_T<Real_>::numDoF() const {
    size_t result = 0;
    for (const auto &s : m_segments) result += s.numDoF();
    for (const auto &j :   m_joints) result += j.numDoF();
    return result;
}

// Full parameters consist of all segment parameters followed by all joint parameters.
template<typename Real_>
VecX_T<Real_> RodLinkage_T<Real_>::getDoFs() const {
    const size_t n = numDoF();
    VecX params(n);

    for (size_t i = 0; i < numSegments(); ++i) { auto slice = params.segment(m_dofOffsetForSegment[i], m_segments[i].numDoF()); m_segments[i].getParameters(slice); }
    for (size_t i = 0; i < numJoints()  ; ++i) { auto slice = params.segment(m_dofOffsetForJoint  [i], m_joints  [i].numDoF()); m_joints  [i].getParameters(slice); }

    return params;
}

// Full parameters consist of all segment parameters followed by all joint parameters.
// "spatialCoherence" affects how terminal edge thetas are determined from the
// joint parameters; see joint.applyConfiguration.
template<typename Real_>
void RodLinkage_T<Real_>::setDoFs(const Eigen::Ref<const VecX> &params, bool spatialCoherence) {
    BENCHMARK_START_TIMER_SECTION("RodLinkage setDoFs");
    const size_t n = numDoF();
    if (size_t(params.size()) != n) throw std::runtime_error("Invalid number of parameters");

    const size_t ns = m_segments.size();
    m_networkPoints.resize(ns);
    m_networkThetas.resize(ns);

    // First, unpack the segment parameters into the points/thetas arrays
    auto processSegment = [&](size_t si) {
        auto slice = params.segment(m_dofOffsetForSegment[si], m_segments[si].numDoF());
        m_segments[si].unpackParameters(slice, m_networkPoints[si], m_networkThetas[si]);
    };
#if MESHFEM_WITH_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, ns), [&](const tbb::blocked_range<size_t> &b) { for (size_t si = b.begin(); si < b.end(); ++si) processSegment(si); });
#else
    for (size_t si = 0; si < ns; ++si) processSegment(si);
#endif

    // Second, set all joint parameters and then
    // use them to configure the segments' terminal edges.
    const size_t nj = m_joints.size();
    auto processJoint = [&](size_t ji) {
        m_joints[ji].setParameters(params.segment(m_dofOffsetForJoint[ji], m_joints[ji].numDoF()));
        m_joints[ji].applyConfiguration(m_segments, m_networkPoints, m_networkThetas, spatialCoherence);
    };
#if MESHFEM_WITH_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, nj), [&](const tbb::blocked_range<size_t> &b) { for (size_t ji = b.begin(); ji < b.end(); ++ji) processJoint(ji); });
#else
    for (size_t ji = 0; ji < nj; ++ji) processJoint(ji);
#endif

    // Finally, set the deformed state of each rod in the network
#if MESHFEM_WITH_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, ns), [&](const tbb::blocked_range<size_t> &b) { for (size_t i = b.begin(); i < b.end(); ++i) m_segments[i].rod.setDeformedConfiguration(m_networkPoints[i], m_networkThetas[i]); });
#else
    for (size_t i = 0; i < ns; ++i) m_segments[i].rod.setDeformedConfiguration(m_networkPoints[i], m_networkThetas[i]);
#endif

    m_sensitivityCache.clear();

    BENCHMARK_STOP_TIMER_SECTION("RodLinkage setDoFs");
}

////////////////////////////////////////////////////////////////////////////////
// Extended degrees of freedom: deformed configuration + rest lengths.
////////////////////////////////////////////////////////////////////////////////
template<typename Real_>
size_t RodLinkage_T<Real_>::numRestLengths() const {
    size_t result = 0;

    // A rest length for every free (non-joint) edge of each segment.
    for (const auto &s : m_segments) result += s.numFreeEdges();
    // Two rest lengths for each joint (one for segment A, one for B)
    result += 2 * numJoints();

    return result;
}

template<typename Real_>
VecX_T<Real_> RodLinkage_T<Real_>::getRestLengths() const {
    VecX result(numRestLengths());

    // A rest length for every free (non-joint) edge of each segment.
    size_t offset = 0;
    for (const auto &s : m_segments) {
        auto rlens = s.rod.restLengths();
        const size_t nfe = s.numFreeEdges();
        result.segment(offset, nfe) = Eigen::Map<VecX>(rlens.data(), rlens.size()).segment(s.hasStartJoint(), nfe);
        offset += nfe;
    }

    for (const auto &j : m_joints) {
        result.segment(offset, 2) = j.getRestLengths();
        offset += 2;
    }

    return result;
}

template<typename Real_>
VecX_T<Real_> RodLinkage_T<Real_>::getExtendedDoFs() const {
    VecX result(numExtendedDoF());
    result.segment(0, numDoF()) = getDoFs();
    result.segment(numDoF(), numRestLengths()) = getRestLengths();

    return result;
}

template<typename Real_>
void RodLinkage_T<Real_>::setExtendedDoFs(const VecX_T<Real_> &params, bool spatialCoherence) {
    if (size_t(params.size()) != numExtendedDoF()) throw std::runtime_error("Extended DoF size mismatch");
    size_t offset = numDoF();
    setDoFs(params.segment(0, offset), spatialCoherence);

    // A rest length for every free (non-joint) edge of each segment.
    for (auto &s : m_segments) {
        auto rlens = s.rod.restLengths();
        const size_t nfe = s.numFreeEdges();
        Eigen::Map<VecX>(rlens.data(), rlens.size()).segment(s.hasStartJoint(), nfe) = params.segment(offset, nfe);
        s.rod.setRestLengths(rlens);
        offset += nfe;
    }

    for (auto &j : m_joints) {
        j.setRestLengths(params.segment(offset, 2));
        offset += 2;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Joint operations
////////////////////////////////////////////////////////////////////////////////
template<typename Real_>
RodLinkage_T<Real_>::Joint::Joint(RodLinkage_T *l, const Pt3 &p, const Vec3 &eA, const Vec3 &eB,
                         const std::array<size_t, 2> segmentsA, const std::array<size_t, 2> segmentsB,
                         const std::array<bool  , 2>  isStartA, const std::array<bool  , 2>  isStartB)
    : m_linkage(l), m_pos(p),
      m_segmentsA(segmentsA), m_segmentsB(segmentsB),
      m_isStartA(isStartA), m_isStartB(isStartB)
{
    if ((segmentsA[0] == RodLinkage::NONE) || (segmentsB[0] == RodLinkage::NONE))
        throw std::runtime_error("First segment must exist for each rod incident this joint");

    m_len_A = eA.norm();
    m_len_B = eB.norm();

    Vec3 tA = eA / m_len_A,
         tB = eB / m_len_B;
    // Pick the sign of edge vector B so that angle "alpha" between edge vectors A and B is
    // acute, not obtuse.
    m_sign_B = std::copysign(1.0, stripAutoDiff(tA.dot(tB)));
    tB *= m_sign_B;
    m_source_normal = tA.cross(tB);
    Real_ sin_alpha = m_source_normal.norm();
    m_source_normal /= sin_alpha;

    // Compute angle bisector "t"; the joint's orientation will be described by frame (t | n x t | n)
    m_source_t = tA + tB;
    assert(m_source_t.norm() > 1e-8); // B's sign was chosen to make alpha acute...
    m_source_t.normalize();

    m_alpha = asin(sin_alpha); // always in [0, pi/2]
    m_omega.setZero();

    m_update();
}

template<typename Real_>
template<class Derived>
void RodLinkage_T<Real_>::Joint::setParameters(const Eigen::DenseBase<Derived> &vars) {
    if (size_t(vars.size()) < numDoF()) throw std::runtime_error("DoF indices out of bounds");
    // 9 parameters: position, omega, alpha, len a, len b
    m_pos   = vars.template segment<3>(0);
    m_omega = vars.template segment<3>(3);
    m_alpha = vars[6];
    m_len_A = vars[7];
    m_len_B = vars[8];

    m_update();
}

template<typename Real_>
template<class Derived>
void RodLinkage_T<Real_>::Joint::getParameters(Eigen::DenseBase<Derived> &vars) const {
    if (size_t(vars.size()) < numDoF()) throw std::runtime_error("DoF indices out of bounds");
    // 9 parameters: position, omega, alpha, len a, len b
    vars.template segment<3>(0) = m_pos;
    vars.template segment<3>(3) = m_omega;
    vars[6]                     = m_alpha;
    vars[7]                     = m_len_A;
    vars[8]                     = m_len_B;
}

// Update the network's full collection of rod points and twist angles with the
// values determined by this joint's configuration (editing only the values
// related to the incident terminal rod edges).
// The "rodSegments" are needed to compute material frame angles from the
// material axis vector.
// "spatialCoherence" determines whether the 2Pi offset ambiguity in theta is
// resolved by minimizing twisting energy (true) or minimizing the change made (temporal coherence; false)
template<typename Real_>
void RodLinkage_T<Real_>::Joint::applyConfiguration(const std::vector<RodSegment>   &rodSegments,
                                                    std::vector<std::vector<Pt3>>   &networkPoints,
                                                    std::vector<std::vector<Real_>> &networkThetas,
                                                    bool spatialCoherence) const {
    // Vector "e" always points outward from the joint into/along the rod.
    auto configureEdge = [&](Vec3 e, bool isStart, const ElasticRod_T<Real_> &rod, std::vector<Pt3> &pts, std::vector<Real_> &thetas) {
        const size_t nv = pts.size();
        const size_t ne = thetas.size();
        assert(nv == rod.numVertices());
        assert(ne == rod.numEdges());

        // Orient "e" so that it agrees with the rod orientation (points into start/out of end)
        // (e is now the scaled tangent vector of the new edge).
        if (!isStart) e *= -1.0;

        pts[isStart ? 0 : nv - 2] = m_pos - 0.5 * e;
        pts[isStart ? 1 : nv - 1] = m_pos + 0.5 * e;

        // Material axis d2 is given by the normal vector.
        const size_t edgeIdx = isStart ? 0 : ne - 1;
        thetas[edgeIdx] = rod.thetaForMaterialFrameD2(m_normal, e, edgeIdx, spatialCoherence);
    };

    // Configure segment 0 of each rod: edge vectors m_e_[AB] point OUT of
    // segment 0, while configureEdge expects an inward-pointing vector
    configureEdge(-m_e_A, m_isStartA[0], rodSegments[m_segmentsA[0]].rod, networkPoints[m_segmentsA[0]], networkThetas[m_segmentsA[0]]);
    configureEdge(-m_e_B, m_isStartB[0], rodSegments[m_segmentsB[0]].rod, networkPoints[m_segmentsB[0]], networkThetas[m_segmentsB[0]]);

    // Configure segment 1, if it exists.
    if (m_segmentsA[1] != NONE) configureEdge(m_e_A, m_isStartA[1], rodSegments[m_segmentsA[1]].rod, networkPoints[m_segmentsA[1]], networkThetas[m_segmentsA[1]]);
    if (m_segmentsB[1] != NONE) configureEdge(m_e_B, m_isStartB[1], rodSegments[m_segmentsB[1]].rod, networkPoints[m_segmentsB[1]], networkThetas[m_segmentsB[1]]);
}

// Update cache; to be called whenever the edge vectors change.
template<typename Real_>
void RodLinkage_T<Real_>::Joint::m_update() {
    Mat3 source_config; // non-orthogonal frame formed by the two edge tangent and the normal
    source_config.col(0) = source_t_A();
    source_config.col(1) = source_t_B();
    source_config.col(2) = m_source_normal;
    Mat3 curr_config = ropt::rotated_matrix(m_omega, source_config);

    m_e_A = curr_config.col(0) * m_len_A;
    m_e_B = curr_config.col(1) * m_len_B;
    m_normal = curr_config.col(2);
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
template<typename Real_>
void RodLinkage_T<Real_>::Joint::swapAngleDefinition() {
    std::swap(m_segmentsA, m_segmentsB);
    std::swap(m_isStartA, m_isStartB);
    std::swap(m_len_A, m_len_B);

    m_source_t = m_sign_B * m_source_normal.cross(m_source_t);
    m_sign_B *= -1;
    m_alpha = M_PI - m_alpha;

    m_update(); // swap cached m_e_A and m_e_B
}

////////////////////////////////////////////////////////////////////////////////
// Rod segment operations
////////////////////////////////////////////////////////////////////////////////
// Construct the initial rest points for a rod; note that the endpoints will be
// repositioned if the rod connects at a joint.
template<typename Real_>
std::vector<Pt3_T<Real_>> constructInitialRestPoints(const Pt3_T<Real_> &startPt, const Pt3_T<Real_> &endPt, size_t nsubdiv) {
    if (nsubdiv < 5)
        throw std::runtime_error("Rods in a linkage must have at least 5 edges (to prevent conflicting start/end joint constraints and fully separate joint influences in Hessian)");
    // Half an edge will extend past each endpoint, so only (nsubdiv - 1) edges
    // fit between the endpoints.
    std::vector<Pt3_T<Real_>> rodPts;
    for (size_t i = 0; i <= nsubdiv; ++i) {
        Real_ alpha = (i - 0.5) / (nsubdiv - 1);
        rodPts.push_back((1 - alpha) * startPt + alpha * endPt);
    }
    return rodPts;
}

template<typename Real_>
RodLinkage_T<Real_>::RodSegment::RodSegment(const Pt3 &startPt, const Pt3 &endPt, size_t nsubdiv)
    : rod(constructInitialRestPoints(startPt, endPt, nsubdiv)) { }

template<typename Real_>
template<class Derived>
void RodLinkage_T<Real_>::RodSegment::unpackParameters(const Eigen::DenseBase<Derived> &vars,
                                                       std::vector<Pt3  > &points,
                                                       std::vector<Real_> &thetas) const {
    if (numDoF() > size_t(vars.size())) throw std::runtime_error("DoF indices out of bounds");
    const size_t nv = rod.numVertices(), ne = rod.numEdges();
    points.resize(nv);
    thetas.resize(ne);

    size_t offset = 0;

    // Set the centerline position degrees of freedom
    for (size_t i = 0; i < nv; ++i) {
        // The first/last edge don't contribute degrees of freedom if they're part of a joint.
        if ((i <       2) && (startJoint != NONE)) continue;
        if ((i >= nv - 2) && (endJoint   != NONE)) continue;
        points[i] = vars.template segment<3>(offset);
        offset += 3;
    }

    // Unpack the material axis degrees of freedom
    for (size_t j = 0; j < ne; ++j) {
        if ((j ==      0) && (startJoint != NONE)) continue;
        if ((j == ne - 1) && (endJoint   != NONE)) continue;
        thetas[j] = vars[offset++];
    }
}

template<typename Real_>
template<class Derived>
void RodLinkage_T<Real_>::RodSegment::setParameters(const Eigen::DenseBase<Derived> &vars) {
    auto points = rod.deformedPoints();
    auto thetas = rod.thetas();
    unpackParameters(vars, points, thetas);
    rod.setDeformedConfiguration(points, thetas);
}

template<typename Real_>
template<class Derived>
void RodLinkage_T<Real_>::RodSegment::getParameters(Eigen::DenseBase<Derived> &vars) const {
    if (numDoF() > size_t(vars.size())) throw std::runtime_error("DoF indices out of bounds");
    const auto &pts    = rod.deformedPoints();
    const auto &thetas = rod.thetas();
    const size_t nv = rod.numVertices(), ne = rod.numEdges();
    size_t offset = 0;

    // get the centerline position degrees of freedom
    for (size_t i = 0; i < nv; ++i) {
        // The first/last edge don't contribute degrees of freedom if they're part of a joint.
        if ((i <       2) && (startJoint != NONE)) continue;
        if ((i >= nv - 2) && (endJoint   != NONE)) continue;
        vars.template segment<3>(offset) = pts[i];
        offset += 3;
    }

    // Unpack the material axis degrees of freedom
    for (size_t j = 0; j < ne; ++j) {
        if ((j ==      0) && (startJoint != NONE)) continue;
        if ((j == ne - 1) && (endJoint   != NONE)) continue;
        vars[offset++] = thetas[j];
    }
}

template<typename Real_>
void RodLinkage_T<Real_>::RodSegment::setMinimalTwistThetas(bool verbose) {
    // Minimize twisting energy wrt theta.
    // The twisting energy is quadratic wrt theta, so we simply solve for the
    // step bringing the gradient to zero using the equation:
    //      H dtheta = -g,
    // where g and H are the gradient and Hessian of the twisting energy with
    // respect to material axis angles.
    if ((startJoint == NONE) && (endJoint == NONE))
        throw std::runtime_error("Rod with two free ends--system will be rank deficient");

    const size_t ne = rod.numEdges();

    auto pts = rod.deformedPoints();
    auto ths = rod.thetas();

    // First, remove any unnecessary twist stored in the rod by rotating the second endpoint
    // by an integer multiple of 2PI (leaving d2 unchanged).
    Real_ rodRefTwist = 0;
    const auto &dc = rod.deformedConfiguration();
    for (size_t j = 1; j < ne; ++j)
        rodRefTwist += dc.referenceTwist[j];
    const size_t lastEdge = ne - 1;
    Real_ desiredTheta = ths[0] - rodRefTwist;
    // Probably could be implemented with an fmod...
    while (ths[lastEdge] - desiredTheta >  M_PI) ths[lastEdge] -= 2 * M_PI;
    while (ths[lastEdge] - desiredTheta < -M_PI) ths[lastEdge] += 2 * M_PI;

    if (verbose) {
        std::cout << "rodRefTwist: "         << rodRefTwist        << std::endl;
        std::cout << "desiredTheta: "        << desiredTheta       << std::endl;
        std::cout << "old last edge theta: " << dc.theta(lastEdge) << std::endl;
        std::cout << "new last edge theta: " << ths[lastEdge]      << std::endl;
    }

    rod.setDeformedConfiguration(pts, ths);

    auto H = rod.hessThetaEnergyTwist();
    auto g = rod.gradEnergyTwist();
    std::vector<Real_> rhs(ne);
    for (size_t j = 0; j < ne; ++j)
        rhs[j] = -g.gradTheta(j);

    if (startJoint != NONE) H.fixVariable(       0, 0);
    if (  endJoint != NONE) H.fixVariable(lastEdge, 0);

    auto thetaStep = H.solve(rhs);

    for (size_t j = 0; j < ths.size(); ++j)
        ths[j] += thetaStep[j];

    rod.setDeformedConfiguration(pts, ths);
}

template<typename Real_>
Real_ RodLinkage_T<Real_>::energy() const {
#if MESHFEM_WITH_TBB
    Real_ result = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, numSegments()), Real_(0.0),
        [&](const tbb::blocked_range<size_t> &b, Real_ localSum) {
            for (size_t si = b.begin(); si < b.end(); ++si)
                localSum += m_segments[si].rod.energy();
            return localSum;
        }, std::plus<Real_>());
#else
    Real_ result = 0;
    for (const auto &s : m_segments) result += s.rod.energy();
#endif
    return result;
}

template<typename Real_>
Real_ RodLinkage_T<Real_>::energyStretch() const {
    Real_ result = 0;
    for (const auto &s : m_segments) result += s.rod.energyStretch();
    return result;
}

template<typename Real_>
Real_ RodLinkage_T<Real_>::energyBend() const {
    Real_ result = 0;
    for (const auto &s : m_segments) result += s.rod.energyBend();
    return result;
}

template<typename Real_>
Real_ RodLinkage_T<Real_>::energyTwist() const {
    Real_ result = 0;
    for (const auto &s : m_segments) result += s.rod.energyTwist();
    return result;
}

template<typename Real_>
Real_ RodLinkage_T<Real_>::maxStrain() const {
    Real_ max_mag = 0, max_val = 0;
    for (const auto &s : m_segments) {
        const auto &r = s.rod;
        const size_t ne = r.numEdges();
        const auto &dc = r.deformedConfiguration();
        const auto &len = dc.len;
        const auto &restLen = r.restLengths();
        for (size_t j = 0; j < ne; ++j) {
            Real_ val = len[j] / restLen[j] - 1.0;
            if (std::abs(stripAutoDiff(val)) > max_mag) {
                max_mag = std::abs(stripAutoDiff(val));
                max_val = val;
            }
        }
    }
    return max_val;
}

template<typename Real_>
typename RodLinkage_T<Real_>::VecX RodLinkage_T<Real_>::rivetForces(EnergyType eType) const {
    for (const auto &j : m_joints)
        if (j.omega().norm() != 0.0) throw std::runtime_error("Please update the rotation parametrization (updateRotationParametrizations()) for physically meaningful torques.");
    return -gradient(false, eType, false, false, /* skip B rods */ true);
}

template<typename Real_>
Eigen::MatrixXd RodLinkage_T<Real_>::rivetNetForceAndTorques(EnergyType eType) const {
    auto rf = rivetForces(eType);
    const size_t nj = numJoints();
    Eigen::MatrixXd result(nj, 6);

    VectorField<double, 3> jointForce, jointTorque;
    for (size_t j = 0; j < nj; ++j) {
        // Joint variable ordering: pos, omega, alpha
        // This means the segment of the gradient gives us [force, torque] as desired.
        result.block<1, 6>(j, 0) = stripAutoDiff(rf.template segment<6>(m_dofOffsetForJoint[j]).eval());
    }

    return result;
}

////////////////////////////////////////////////////////////////////////////////
// Gradient/Hessian computation
////////////////////////////////////////////////////////////////////////////////
#if MESHFEM_WITH_TBB
template<typename Real_>
struct GradientAssemblerData {
    VecX_T<Real_> g;
    bool constructed = false;
};

template<typename Real_>
using GALocalData = tbb::enumerable_thread_specific<GradientAssemblerData<Real_>>;

template<typename F, typename Real_>
struct GradientAssembler {
    GradientAssembler(F &f, const size_t nvars, GALocalData<Real_> &locals) : m_f(f), m_nvars(nvars), m_locals(locals) { }

    void operator()(const tbb::blocked_range<size_t> &r) const {
        GradientAssemblerData<Real_> &data = m_locals.local();
        if (!data.constructed) { data.g.setZero(m_nvars); data.constructed = true; }
        for (size_t si = r.begin(); si < r.end(); ++si) { m_f(si, data.g); }
    }
private:
    F &m_f;
    size_t m_nvars;
    GALocalData<Real_> &m_locals;
};

template<typename F, typename Real_>
GradientAssembler<F, Real_> make_gradient_assembler(F &f, size_t nvars, GALocalData<Real_> &locals) {
    return GradientAssembler<F, Real_>(f, nvars, locals);
}
#endif

template<typename Real_>
VecX_T<Real_> RodLinkage_T<Real_>::gradient(bool updatedSource, EnergyType eType, bool variableRestLen, bool restlenOnly, const bool skipBRods) const {
    BENCHMARK_START_TIMER_SECTION("RodLinkage gradient");
    VecX g(variableRestLen ? numExtendedDoF() : numDoF());
    g.setZero();

    {
        // Generally when evaluating the gradient at a new iterate (when the source frame is updated),
        // the user will also want to compute the Hessian shortly thereafter.
        // Since the joint parametrization Hessian formula can re-use several
        // values computed for the parametrization Jacobian, for now we always
        // pre-compute and cache the parametrization Hessian. If we find a
        // usage pattern where the gradient is evaluated at many iterates where
        // the Hessian is *not* requested, we might change this strategy.
        const bool evalHessian = updatedSource;
        m_sensitivityCache.update(*this, updatedSource, evalHessian);
    }

    // Accumulate contribution of each segment's elastic energy gradient to the full gradient
    auto accumulateSegment = [&](const size_t si, VecX &gout) {
        const auto &s = m_segments[si];
        auto &r = s.rod;
        // Partial derivatives with respect to the segment's unconstrained DoFs
        const auto  &sg = r.gradient(updatedSource, eType, variableRestLen, restlenOnly);
        const size_t nv = r.numVertices(), ne = r.numEdges();

        // Rest length derivatives
        if (variableRestLen) {
            // Copy over the gradient components for the degrees of freedom
            // that directly control interior/free-end edge rest lengths.
            gout.segment(m_restLenDoFOffsetForSegment[si], s.numFreeEdges()) =
                sg.segment(sg.restLenOffset + s.hasStartJoint(), s.numFreeEdges());

            // Accumulate contributions to the rest lengths controlled by each joint
            for (size_t i = 0; i < 2; ++i) {
                size_t jindex = s.joint(i);
                if (jindex == NONE) continue;
                const auto &joint = m_joints.at(jindex);
                size_t edgeIdx = NONE, dofIdx = NONE;
                {
                    double sA, sB;
                    bool isStart;
                    std::tie(sA, sB, isStart) = joint.terminalEdgeIdentification(si);
                    // Decode which of the global variables controls this segment.
                    if (sA != 0) { dofIdx = m_restLenDoFOffsetForJoint[jindex]; }
                    if (sB != 0) { assert(dofIdx == NONE); dofIdx = m_restLenDoFOffsetForJoint[jindex] + 1; }
                    assert(dofIdx != NONE);
                    edgeIdx = isStart ? 0 : r.numEdges() - 1;
                }
                gout[dofIdx] += sg.gradRestLen(edgeIdx);
            }
            if (restlenOnly) return;
        }

        size_t offset = m_dofOffsetForSegment[si];

        // Copy over the gradient components for the degrees of freedom that
        // directly control the interior/free-end centerline positions and
        // material frame angles.
        for (size_t i = 0; i < nv; ++i) {
            // The first/last edge don't contribute degrees of freedom if they're part of a joint.
            if ((i <       2) && s.hasStartJoint()) continue;
            if ((i >= nv - 2) && s.  hasEndJoint()) continue;
            gout.template segment<3>(offset) = sg.gradPos(i);
            offset += 3;
        }
        for (size_t j = 0; j < ne; ++j) {
            if ((j ==      0) && s.hasStartJoint()) continue;
            if ((j == ne - 1) && s.  hasEndJoint()) continue;
            gout[offset++] = sg.gradTheta(j);
        }

        // Accumulate contributions to the start/end joints (if they exist)
        for (size_t i = 0; i < 2; ++i) {
            size_t jindex = s.joint(i);
            if (jindex == NONE) continue;
            if (skipBRods) {
                if (joint(jindex).segmentABOffset(si) == 1) continue;
            }

            offset = m_dofOffsetForJoint.at(jindex);
            const auto &sensitivity = m_sensitivityCache.lookup(si, static_cast<TerminalEdge>(i));
            const size_t j = sensitivity.j;
            //           pos        e_X     theta^j
            // x_j     [  I    -s_jX 0.5 I     0   ] [ I 0 ... 0]
            // x_{j+1} [  I     s_jX 0.5 I     0   ] [ jacobian ]
            // theta^j [  0          0         I   ]
            gout.template segment<3>(offset + 0) += sg.gradPos(j) + sg.gradPos(j + 1);

            Eigen::Matrix<Real_, 4, 1> dE_djointvar;
            dE_djointvar.template segment<3>(0) = (0.5 * sensitivity.s_jX) * (sg.gradPos(j + 1) - sg.gradPos(j));
            dE_djointvar[3] = sg.gradTheta(j);
            gout.template segment<6>(offset + 3) += sensitivity.jacobian.transpose() * dE_djointvar;
        }

    };
#if MESHFEM_WITH_TBB
    GALocalData<Real_> gaLocalData;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, numSegments()), make_gradient_assembler(accumulateSegment, g.size(), gaLocalData));
    for (auto &data : gaLocalData)
        g += data.g;
#else
    for (size_t si = 0; si < numSegments(); ++si) { accumulateSegment(si, g); }
#endif

    BENCHMARK_STOP_TIMER_SECTION("RodLinkage gradient");
    return g;
}

// Construct sparse (compressed row) representation of dvk_dri; dv_dr[k][i] gives the derivative of
// unconstrained segment variable k with respect to the global reduced
// linkage variables i.
// If segmentJointDofOffset != NONE, the derivatives of unconstrained rest lengths with respect
// to global reduced linkage variables are also computed.
template<typename Real_>
struct dv_dr_entry {
    typename CSCMatrix<SuiteSparse_long, Real_>::index_type first;
    typename CSCMatrix<SuiteSparse_long, Real_>::value_type second;
};
template<typename Real_>
using dv_dr_type = std::vector<std::vector<dv_dr_entry<Real_>>>;

// For correct autodiff code, we must still keep zero entries if they have nonzero derivatives!
bool entryIdenticallyZero(double val) { return val == 0; }
bool entryIdenticallyZero(ADReal val) { return (val == 0) && (val.derivatives().squaredNorm() == 0); }

template<typename Real_, typename LTESPtr>
void
dv_dr_for_segment(const typename RodLinkage_T<Real_>::RodSegment &s,
                  const std::array<LTESPtr, 2> &jointSensitivity,
                  const std::array<size_t, 2> &segmentJointDofOffset,
                  size_t segmentDofOffset,
                  dv_dr_type<Real_> &dv_dr, /* output */
                  // Arguments needed only for variable rest length case
                  const std::array<size_t, 2> &segmentJointRestLenDofOffset = std::array<size_t, 2>(),
                  size_t segmentRestLenDofOffset = RodLinkage::NONE,
                  bool skip = false
                  )
{
    const bool variableRestLen = (segmentRestLenDofOffset != RodLinkage::NONE);
    const auto &r = s.rod;
    const size_t nv = r.numVertices(), ne = r.numEdges();
    const size_t numFullDoF = variableRestLen ? r.numExtendedDoF() : r.numDoF();
    using index_type = SuiteSparse_long;

    dv_dr.resize(numFullDoF);
    if (skip) return;
    // Joint variables: pos, omega, alpha, len_A, len_B
    // pos, omega, alpha, len_A affect segment A terminal vertices (1 pos component + 3 omega components + 1 alpha + 1 len = 6 vars)
    // pos, omega, alpha, len_B affect segment B terminal vertices (1 pos component + 3 omega components + 1 alpha + 1 len = 6 vars)
    // omega, alpha             affect segment A's terminal theta  (4 vars) (but alpha dependence is 0 if source frame has been updated...)
    // omega, alpha             affect segment B's terminal theta  (4 vars) (but alpha dependence is 0 if source frame has been updated...)
    for (auto &row : dv_dr) { row.clear(); row.reserve(6); } // At most 6 reduced variables affect each variable.

    for (size_t k = 0; k < numFullDoF; ++k) {
        auto &dvk_dr = dv_dr[k];

        auto jointVertexSensitivity = [&](bool isTail, index_type comp, size_t localJointIdx) {
            index_type o = index_type(segmentJointDofOffset[localJointIdx]);
            const auto &js = *jointSensitivity[localJointIdx];
            Real_ dx_de = isTail ? -1 : 1;
            dvk_dr.push_back({o + comp, 1.0});
            for (index_type l = 0; l < 6; ++l) {
                Real_ de_comp_dvar_l = js.jacobian(comp, l);
                if (entryIdenticallyZero(de_comp_dvar_l)) continue;
                dvk_dr.push_back({o + 3 + l, 0.5 * dx_de * js.s_jX * de_comp_dvar_l});
            }
        };

        auto jointEdgeSensitivity = [&](size_t localJointIdx) {
            index_type o = index_type(segmentJointDofOffset[localJointIdx]);
            const auto &js = *jointSensitivity[localJointIdx];
            for (index_type l = 0; l < 6; ++l) {
                Real_ dtheta_dvar_l = js.jacobian(3, l);
                if (entryIdenticallyZero(dtheta_dvar_l)) continue;
                dvk_dr.push_back({o + 3 + l, dtheta_dvar_l});
            }
        };

        if (k < r.thetaOffset()) {
            size_t vtx = k / 3;
            index_type component = k % 3;
            if      ((vtx <       2) && jointSensitivity[0]) jointVertexSensitivity(vtx == 0,      component, 0);
            else if ((vtx >= nv - 2) && jointSensitivity[1]) jointVertexSensitivity(vtx == nv - 2, component, 1);
            else                                             dvk_dr.push_back({index_type(segmentDofOffset + k - 3 * 2 * s.hasStartJoint()), 1.0}); // Permuted Kronecker delta
        }
        else if (k < r.restLenOffset()) {
            size_t eidx = k - r.thetaOffset();
            if      ((eidx ==      0) && jointSensitivity[0]) jointEdgeSensitivity(0);
            else if ((eidx == ne - 1) && jointSensitivity[1]) jointEdgeSensitivity(1);
            else                                              dvk_dr.push_back({index_type(segmentDofOffset + s.numPosDoF() + eidx - s.hasStartJoint()), 1.0}); // Permuted Kronecker delta
        }
        else {
            // Rest length variable...
            assert(variableRestLen);
            size_t eidx = k - r.restLenOffset();
            if      ((eidx ==      0) && jointSensitivity[0]) dvk_dr.push_back({index_type(segmentJointRestLenDofOffset[0]), 1.0});
            else if ((eidx == ne - 1) && jointSensitivity[1]) dvk_dr.push_back({index_type(segmentJointRestLenDofOffset[1]), 1.0});
            else                                              dvk_dr.push_back({index_type(segmentRestLenDofOffset + eidx - s.hasStartJoint()), 1.0}); // Permuted Kronecker delta
        }
    }
}

// Rod vertex stencil:
//  +---+---+---+---+
//          ^
// Rod edge stencil:
//    +---+---+---+
//          ^
template<typename Real_>
size_t RodLinkage_T<Real_>::hessianNNZ(bool variableRestLen) const {
    size_t result = 0;

    for (size_t si = 0; si < numSegments(); ++si) {
        const auto &s = m_segments[si];
        const auto &r = s.rod;
        if (r.numEdges() < 5) throw std::runtime_error("Assumption of at least 5 subdivisions violated."); // verify assumption used to simplify sparsity pattern analysis (fully separated joints)
        // Number of "free" vertices and joints in the rod (independent degrees of freedom that are not controlled by the joints)
        int nfv = int(s.numFreeVertices()), // integers must be signed for formulas below
            nfe = int(s.numFreeEdges());

        result += 6 * nfv + nfe;               // diagonal blocks of x-x and theta-theta terms
        result += 9 * ((nfv - 1) + (nfv - 2)); // contributions from each free vertex to the previous free vertices in the stencil (up to 2 neighbors)

        size_t odiagxt;
        // Add contribution from free thetas to the free vertices in their stencils; depends on number of joints.
        if      (s.numJoints() == 2) { odiagxt = 3 * (2 * 2 + std::min(2, nfe - 2) * std::min(3, nfv) + std::max(nfv - 3, 0) * 4); }
        else if (s.numJoints() == 1) { odiagxt = 3 * (1 * 2 + std::min(2, nfe - 1) * std::min(3, nfv) + std::max(nfv - 3, 0) * 4); }
        else { throw std::runtime_error("Each segment should have exactly two joints"); }
        result += odiagxt;
        result += nfe - 1; // Contributions of thetas to previous thetas in the edge stencil

        if (variableRestLen) {
            // Entries for this segment's rest lengths
            // x-free-restlen interactions are the same as x-theta
            result += odiagxt;
            result += nfe + 2 * (nfe - 1); // theta-restlen block is tri-diagonal (and we take the whole thing)
            result += nfe +      nfe - 1;  // free restlen-restlen part is tridiagonal (and we take only the upper half)
            // restlen-joint blocks: two closest edges on segment A|B interact with pos, omega, alpha, len_{A|B}
            for (size_t j = 0; j < 2; ++j) {
                size_t ji = s.joint(j);
                if (ji == NONE) continue;
                result += 2 * (6 + 2); // pos, omega, alpha, len_X
            }
        }
    }

    for (const auto &j : m_joints) {
        // All joint variables interact with each other except for (len_B, len_A)
        result += (45 - 1); // upper tri of dense 9x9 block for joint self-interaction, minus the (len_B, len_A) interaction

        // The two closest vertices and thetas of all incident segments depend
        // on the joint's position, omega, alpha vars as well as the joint len
        // vars that control only the rod containing that segment
        // (len_A for rod A, len_B for rod B).
        //                           x   theta     #joint vars
        result += j.valence() * 2 * (3 +   1  ) * (7    +    1);

        if (variableRestLen) {
            // The two joint rest lengths interact with the adjacent free vertices/thetas/rest lengths of their corresponding incident segments.
            // We have one of these sets of interactions for each incident segment.
            result += j.valence() * (3 + 1 + 1);
            // They also interact with the joint variables that affect the corresponding edge vectors
            // ((pos, omega, alpha, len_A) for rod A's rest length, (pos, omega, alpha, len_B) for rod B's rest length)
            result += 2 * 8;

            // They also interact with themselves (but not each other).
            result += 2;
        }
    }

    return result;
}

template<typename Real_>
auto RodLinkage_T<Real_>::hessianSparsityPattern(bool variableRestLen, Real_ val) const -> CSCMat {
    if (variableRestLen && m_cachedHessianVarRLSparsity) {
        if (m_cachedHessianVarRLSparsity->Ax[0] != val) m_cachedHessianVarRLSparsity->fill(val);
        return *m_cachedHessianVarRLSparsity;
    }
    if (!variableRestLen && m_cachedHessianSparsity) {
        if (m_cachedHessianSparsity->Ax[0] != val) m_cachedHessianSparsity->fill(val);
        return *m_cachedHessianSparsity;
    }

    const size_t nnz = hessianNNZ(variableRestLen);
    const size_t ndof = variableRestLen ? numExtendedDoF() : numDoF();

    CSCMat result(ndof, ndof);
    result.symmetry_mode = CSCMat::SymmetryMode::UPPER_TRIANGLE;
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
    auto addIdx = [&](const size_t idx) { Ai.push_back(idx); };

    auto finalizeCol = [&](bool needsSort = false) {
        const size_t colStart = Ap.back();
        const size_t colEnd = Ai.size();
        Ap.push_back(colEnd);
        if (needsSort)
            std::sort(Ai.begin() + colStart, Ai.begin() + colEnd);
    };

    // Build the sparsity pattern in compressed form one column (variable) at a time.
    result.Ap.push_back(0);
    for (size_t si = 0; si < numSegments(); ++si) {
        const auto &s = m_segments[si];
        const auto &r = s.rod;
        const size_t so = m_dofOffsetForSegment[si];
        if (r.numEdges() < 5) throw std::runtime_error("Assumption of at least 5 subdivisions violated."); // verify assumption used to simplify sparsity pattern analysis (fully separated joints)

        const size_t nfv = s.numFreeVertices(), nfe = s.numFreeEdges();

        // Contribution from free vertices to the earlier free vertices before them in their stencils.
        for (size_t vi = 0; vi < nfv; ++vi) {
            const size_t vstart = vi - std::min<size_t>(2, vi);
            for (size_t c = 0; c < 3; ++c) {
                addIdxRange(so + 3 * vstart, so + 3 * vi + c + 1);
                finalizeCol();
            }
        }

        // Contribution from free thetas to the free vertices, thetas in their stencil
        // We have two indexing cases depending on whether or not there is a
        // start joint (whether the first edge shares the index of the vertex
        // before or after).
        //    +---+)--+---+-...---+--(+---+
        //          0 0 1 1     nfv-1
        //    +---+---+---+-...---+--(+---+
        //    0 0 1 1 2 2 3     nfv-1
        // Edge stencil according to free vertex indexing (when has start joint)
        //  ei-2  ei-1   ei   ei+1
        //    +-----+-----+-----+
        //      ei-1   ^    ei+1
        //            ei
        // Edge stencil according to free vertex indexing (no start joint)
        //  ei-1   ei   ei+1  ei+2
        //    +-----+-----+-----+
        //      ei-1   ^    ei+1
        //            ei
        for (size_t ei = 0; ei < nfe; ++ei) {
            const size_t vstart = ei - std::min<size_t>( s.hasStartJoint() ? 2 : 1, ei);
            const size_t vend   = std::min<size_t>(ei + (s.hasStartJoint() ? 1 : 2) + 1, nfv);
            addIdxRange(so + 3 * vstart, so + 3 * vend);
            const size_t estart = ei - std::min<size_t>(1, ei);
            addIdxRange(so + 3 * nfv + estart, so + 3 * nfv + ei + 1);
            finalizeCol();
        }
    }

    for (size_t ji = 0; ji < numJoints(); ++ji) {
        const auto &j = m_joints[ji];
        const size_t jo = m_dofOffsetForJoint[ji];
        for (size_t d = 0; d < j.numDoF(); ++d) {
            // Contribution from the joint variables to the free vertices of the incident segments.
            j.visitInfluencedSegmentVars(d, addIdx);

            // Joint-joint blocks:
            if      (d <= 6) { addIdxRange(jo, jo + d + 1);             } // (pos, omega, alpha) all interact with each other
            else if (d == 7) { addIdxRange(jo, jo + 8);                 } // len_A interacts with (pos, omega, alpha, self)
            else if (d == 8) { addIdxRange(jo, jo + 7); addIdx(jo + 8); } // len_B interacts with (pos, omega, alpha, self) but not len_A
            else assert(false);

            finalizeCol(true); // variables weren't added in order; need sorting
        }
    }

    if (variableRestLen) {
        // Interaction of the segments' rest length variables with the free vertices, thetas, and joint variables.
        for (size_t si = 0; si < numSegments(); ++si) {
            const auto &s = m_segments[si];
            const size_t srlo = m_restLenDoFOffsetForSegment[si];
            const size_t so = m_dofOffsetForSegment[si];
            const size_t nfv = s.numFreeVertices(), nfe = s.numFreeEdges();

            for (size_t ei = 0; ei < nfe; ++ei) {
                // Same restlen-vertex interactions as the theta-vertex interactions
                const size_t vstart = ei - std::min<size_t>( s.hasStartJoint() ? 2 : 1, ei);
                const size_t vend   = std::min<size_t>(ei + (s.hasStartJoint() ? 1 : 2) + 1, nfv);
                addIdxRange(so + 3 * vstart, so + 3 * vend);

                // Rest lengths interact with thetas in the full edge stencil (not just the upper triangle)
                const size_t estart = ei - std::min<size_t>(1, ei);
                addIdxRange(so + 3 * nfv + estart, so + 3 * nfv + std::min<size_t>((ei + 1) + 1, nfe));

                // Rest lengths interact with rest lengths in the upper triangle of the edge stencil
                addIdxRange(srlo + estart, srlo + ei + 1);

                // restlen-joint blocks: closest two edges of both segments interact with the position/omega variables.
                //                       closest two edges of segment A interact with alpha, len_A
                //                       closest two edges of segment B interact with alpha, len_B
                auto jointInteraction = [&](const size_t ji) {
                    assert(ji != NONE);
                    const size_t jo = m_dofOffsetForJoint[ji];
                    const auto &  j = m_joints[ji];
                    const size_t abo = j.segmentABOffset(si);
                    addIdxRange(jo, jo + 7); // end edge and second-to-end edge interact with position, omega, alpha
                    addIdx(jo + 7 + abo); // edges on segment_{A|B} interact with len_{A|B},
                };

                // Note: these two conditions can overlap when nsubdiv = 5!
                if ((ei <        2) && s.hasStartJoint()) jointInteraction(s.startJoint);
                if ((ei >= nfe - 2) && s.  hasEndJoint()) jointInteraction(s.  endJoint);

                // Joint variable row indices weren't added in order; needs sorting
                finalizeCol(true);
            }
        }

        // Interaction of the joints' rest length variables with the free vertices, thetas, and joint variables, free edge rest lengths, and rest lengths
        for (size_t ji = 0; ji < numJoints(); ++ji) {
            const auto &j = m_joints[ji];

            for (size_t i = 0; i < 2; ++i) {
                // Interactions with all free vertices/thetas/rest len variables
                j.visitInfluencedSegmentVars(i, addIdx, true);
                const size_t jo = m_dofOffsetForJoint[ji];

                // Add interactions with the joint variables
                if (i == 0) { addIdxRange(jo, jo + 8);                 } // (pos, omega, alpha, len_A)
                else        { addIdxRange(jo, jo + 7); addIdx(jo + 8); } // (pos, omega, alpha, len_B)

                // Add the self-interactions of the joint rest lengths
                addIdx(m_restLenDoFOffsetForJoint[ji] + i);

                // Variables weren't added in order; needs sorting
                finalizeCol(true);
            }
        }

#if 0
        // Verify that all columns are now sorted.
        for (size_t i = 0; i < ndof; ++i) {
            const auto j0 = Ap.at(i);
            auto prev = Ai.at(j0);
            for (auto j = j0; j < Ap.at(i + 1); ++j) {
                if (Ai.at(j) < prev) throw std::runtime_error("Row index out of order");
                prev = Ai.at(j);
            }
        }
#endif
    }

    if (size_t(result.nz) != result.Ai.size()) throw std::runtime_error("Incorrect NNZ prediction: " + std::to_string(result.nz) + " vs " + std::to_string(result.Ai.size()));

    if (variableRestLen) { m_cachedHessianVarRLSparsity = std::make_unique<CSCMat>(result); }
    else                 { m_cachedHessianSparsity      = std::make_unique<CSCMat>(result); }

    return result;
}

template<typename Real_>
VecX_T<Real_> RodLinkage_T<Real_>::getExtendedDoFsPSRL() const {
    VecX result(numExtendedDoFPSRL());
    const size_t ns = numSegments();
    result.head(numDoF()) = getDoFs();
    result.tail(ns) = m_perSegmentRestLen;
    return result;
}

template<typename Real_>
void RodLinkage_T<Real_>::setExtendedDoFsPSRL(const VecX &params, bool spatialCoherence) {
    if (size_t(params.size()) != numExtendedDoFPSRL()) throw std::runtime_error("Extended DoF size mismatch");
    setDoFs(params.head(numDoF()), spatialCoherence);
    m_perSegmentRestLen = params.tail(numSegments());
    m_setRestLengthsFromPSRL();
}

template<typename Real_>
VecX_T<Real_> RodLinkage_T<Real_>::gradientPerSegmentRestlen(bool updatedSource, EnergyType eType) const {
    auto gPerEdgeRestLen = gradient(updatedSource, eType, true);
    VecX result(numExtendedDoFPSRL());
    result.head(numDoF()) = gPerEdgeRestLen.head(numDoF());
    m_segmentRestLenToEdgeRestLenMapTranspose.applyRaw(gPerEdgeRestLen.tail(numRestLengths()).data(), result.tail(numSegments()).data(), /* no transpose */ false);
    return result;
}

template<typename Real_>
auto RodLinkage_T<Real_>::hessianPerSegmentRestlenSparsityPattern(Real_ val) const -> CSCMat {
    if (m_cachedHessianPSRLSparsity) return *m_cachedHessianPSRLSparsity;

    auto hspPerEdge = hessianSparsityPattern(true);

    const size_t nd = numDoF();
    const size_t nedpsrl = nd + numSegments();
    TMatrix result(nedpsrl, nedpsrl);
    result.symmetry_mode = TMatrix::SymmetryMode::UPPER_TRIANGLE;

    auto isRestLen = [&](const size_t i) { return i >= nd; };
    const CSCMat &S = m_segmentRestLenToEdgeRestLenMapTranspose;
    for (const auto &t : hspPerEdge) {
        // i <= j, so "i" can only be a rest length if "j" is as well
        if (isRestLen(t.j)) {
            // Loop over the segments affecting rest length "j"
            const size_t sj = t.j - nd;
            const size_t ibegin = S.Ap.at(sj), iend = S.Ap.at(sj + 1);
            for (size_t idx = ibegin; idx < iend; ++idx) {
                size_t j = S.Ai[idx] + nd;

                if (isRestLen(t.i)) {
                    // Loop over the segments affecting rest length "i"
                    const size_t si = t.i - nd;
                    const size_t ibegin2 = S.Ap.at(si), iend2 = S.Ap.at(si + 1);
                    for (size_t idx2 = ibegin2; idx2 < iend2; ++idx2) {
                        size_t i = S.Ai[idx2] + nd;
                        if (i <= j) result.addNZ(i, j, 1.0);
                        if ((t.i != t.j) && (j <= i)) result.addNZ(j, i, 1.0);
                    }
                }
                else {
                    // Guaranteed t.i < j
                    result.addNZ(t.i, j, 1.0);
                }

            }
        }
        else {
            // Identity block
            result.addNZ(t.i, t.j, 1.0);
        }
    }
    m_cachedHessianPSRLSparsity = std::make_unique<CSCMat>(result);
    m_cachedHessianPSRLSparsity->fill(val);
    return *m_cachedHessianPSRLSparsity;
}

template<typename Real_>
void RodLinkage_T<Real_>::hessianPerSegmentRestlen(CSCMat &H, EnergyType eType) const {
    assert(H.symmetry_mode == CSCMat::SymmetryMode::UPPER_TRIANGLE);
    BENCHMARK_SCOPED_TIMER_SECTION timer("hessianPerSegmentRestlen");

    const size_t ndof = numDoF() + numSegments();
    assert((size_t(H.m) == ndof) && (size_t(H.n) == ndof));
    UNUSED(ndof);

    // Compute Hessian using per-edge rest lengths
    auto HPerEdge = hessianSparsityPattern(true);
    hessian(HPerEdge, eType, true);

    const size_t nd = numDoF();

    // Leverage the fact that HPerEdge is upper triangular: i <= j.
    // Copy the deformed configuration entries over (identity block)
    std::copy(HPerEdge.Ap.begin(), HPerEdge.Ap.begin() + nd, H.Ap.begin());
    std::copy(HPerEdge.Ax.begin(), HPerEdge.Ax.begin() + HPerEdge.Ap[nd], H.Ax.begin());
    std::copy(HPerEdge.Ai.begin(), HPerEdge.Ai.begin() + HPerEdge.Ap[nd], H.Ai.begin());

    // Use m_segmentRestLenToEdgeRestLenMapTranspose to transform the rest
    // length part of the Hessian.
    const CSCMat &S = m_segmentRestLenToEdgeRestLenMapTranspose;
    const size_t nrl = numRestLengths();
    size_t hint = 0;
    for (size_t rlj = 0; rlj < nrl; ++rlj) {
        const size_t j = rlj + nd;
        // Loop over each output column "l" generated by per-edge rest length "j"
        const size_t lend = S.Ap[rlj + 1];
        for (size_t idx = S.Ap[rlj]; idx < lend; ++idx) {
            const size_t l = S.Ai[idx] + nd;
            const Real_ colMultiplier = S.Ax[idx];

            // Create entries for each input Hessian entry
            const size_t input_end = HPerEdge.Ap[j + 1];
            for (size_t idx_in = HPerEdge.Ap[j]; idx_in < input_end; ++idx_in) {
                const Real_ colVal = colMultiplier * HPerEdge.Ax[idx_in];
                const size_t i = HPerEdge.Ai[idx_in];
                if (i < nd) { // left transformation is in the identity block
                    hint = H.addNZ(i, l, colVal, hint);
                }
                else {
                    // Loop over each output entry
                    const size_t rli = i - nd;
                    size_t kprev = 0;
                    size_t kprev_idx = 0;
                    const size_t outrow_end = S.Ap[rli + 1];
                    for (size_t outrow_idx = S.Ap[rli]; outrow_idx < outrow_end; ++outrow_idx) {
                        const size_t k = S.Ai[outrow_idx] + nd;
                        const Real_ val = S.Ax[outrow_idx] * colVal;
                        if (k <= l) {
                            // Accumulate entries from input's upper triangle
                            if (k == kprev) { H.addNZ(kprev_idx, val); }
                            else     { hint = H.addNZ(k, l, val, hint);
                                       kprev = k, kprev_idx = hint - 1; }
                        }
                        if ((i != j) && (l <= k)) H.addNZ(l, k, val); // accumulate entries from input's (strict) lower triangle
                    }
                }
            }
        }
    }
}

// Partition the segment indices into list of segments making up each rod (i.e., polylines)
// Within each rod, segment indices are listed in order. We attempt to pick a
// consistent direction for all "A" rods and all "B" rods (with A and B oriented oppositely).
template<typename Real_>
std::vector<std::tuple<bool, std::vector<size_t>>> RodLinkage_T<Real_>::traceRods() const {
    // Could be optimized...
    std::vector<std::tuple<bool, std::vector<size_t>>> result;

    size_t numVisited = 0;
    const size_t ns = numSegments();
    std::vector<bool> segment_visited(ns, false);

    while (numVisited < ns) {
        // Find an unvisited terminal segment (one with no continuation segment at at least one joint)
        // so that the BFS traverses the segment indices in a single direction.
        size_t si;
        for (si = 0; si < ns; ++si) {
            if (segment_visited[si]) continue;
            if (segment(si).startJoint == NONE) break;
            if (segment(si).endJoint   == NONE) break;
            if (joint(segment(si).startJoint).continuationSegment(si) == NONE) break;
            if (joint(segment(si).endJoint  ).continuationSegment(si) == NONE) break;
        }
        if (si == ns) throw std::runtime_error("Only non-terminal unvisited segments exist.");

        result.emplace_back();
        auto &rodSegments = std::get<1>(result.back());
        std::queue<size_t> bfsQueue;

        auto visit = [&](size_t i) {
            if (i == NONE) return;
            if (segment_visited[i]) return;
            ++numVisited;
            segment_visited[i] = true;
            bfsQueue.push(i);
            rodSegments.push_back(i);
        };

        visit(si);
        size_t abo = 2;
        while (!bfsQueue.empty()) {
            size_t u = bfsQueue.front();
            bfsQueue.pop();

            auto visitJoint = [&](size_t ji) {
                if (ji == NONE) return;
                size_t v = joint(ji).continuationSegment(u);
                if (abo == 2) abo = joint(ji).segmentABOffset(u);
                if (abo != joint(ji).segmentABOffset(u)) throw std::runtime_error("Inconsistent rod label");
                visit(v);
            };
            visitJoint(segment(u).startJoint);
            visitJoint(segment(u).endJoint);
        }

        if (abo > 1) throw std::runtime_error("Rod label not detected");
        std::get<0>(result.back()) = (abo == 0);
    }

    // Choose the direction of each rod: orient all A rods consistently with
    // the other A rods (and same for B), but ensure B rods are the reverse of
    // A rods.
    // Note: this approach is only a heuristic, assuming all rods follow a
    // single general direction (instead of smoothly varying direction); a more
    // correct solution would use a BFS to propagate the local choice.
    // Note: we use the rest positions for this ordering so that we get the same
    // results before/after deployment.
    Vec3 Adir, Bdir;
    Adir.setZero(), Bdir.setZero();
    for (size_t ri = 0; ri < result.size(); ++ri) {
        auto &segs      = std::get<1>(result[ri]);
        const bool is_A = std::get<0>(result[ri]);
        assert(segs.size() != 0);
        if (segs.size() <= 1) continue; // nothing to sort...
        const auto &r1 = segment(segs[1]).rod;
        const auto &r0 = segment(segs[0]).rod;
        Vec3 dir = (r1.restPoints()[r1.numEdges() / 2] - r0.restPoints()[r0.numEdges() / 2]).normalized();
        if (is_A) {
            if (Adir.squaredNorm() == 0) {
                Adir = dir;
                Bdir = -dir;
            }
            if (Adir.dot(dir) < 0) std::reverse(segs.begin(), segs.end());
        }
        else {
            if (Bdir.squaredNorm() == 0) {
                Bdir = dir;
                Adir = -dir;
            }
            if (Bdir.dot(dir) < 0) std::reverse(segs.begin(), segs.end());
        }
    }

    return result;
}

template<typename Real_>
template<class F>
void RodLinkage_T<Real_>::Joint::visitInfluencedSegmentVars(const size_t var, F &visitor, bool restLenVar) const {
    assert(m_linkage);
    const auto &l = *m_linkage;
    // Visit the affected variables of segment "si".
    // closestOnly: whether to visit only the closest free vertex/theta instead of the closest two.
    auto visitSegment = [&](const size_t si, bool isStart, bool closestOnly) {
        if (si == NONE) return;
        const size_t so = l.dofOffsetForSegment(si);
        const auto &s = l.segment(si);
        const size_t nfv = s.numFreeVertices(), nfe = s.numFreeEdges();

        // Intervals [vstart, vend), [estart, eend) of affected entities.
        size_t vstart, vend, estart, eend;
        if (isStart) {
            vstart = 0, vend = closestOnly ? 1 : 2;
            estart = 0, eend = closestOnly ? 1 : 2;
        }
        else {
            vstart = closestOnly ? nfv - 1 : nfv - 2, vend = nfv;
            estart = closestOnly ? nfe - 1 : nfe - 2, eend = nfe;
        }
        for (size_t i = so + 3 * vstart      ; i < so + 3 * vend      ; ++i) visitor(i);
        for (size_t i = so + 3 * nfv + estart; i < so + 3 * nfv + eend; ++i) visitor(i);

        if (restLenVar) {
            assert(closestOnly);
            // Influenced segment rest length
            visitor(l.restLenDofOffsetForSegment(si) + estart);
        }
    };
    if (!restLenVar) {
        if (var < 7) {
            // Joint position, omega, alpha affect all incident segment variables
            for (size_t i = 0; i < 2; ++i) {
                visitSegment(m_segmentsA[i], m_isStartA[i], false);
                visitSegment(m_segmentsB[i], m_isStartB[i], false);
            }
        }
        if (var == 7) { // Edge length A affects the closest two vertices/thetas of rod A only
            for (size_t i = 0; i < 2; ++i)
                visitSegment(m_segmentsA[i], m_isStartA[i], false);
        }
        if (var == 8) { // Edge length B affects the closest two vertices/thetas of rod B only
            for (size_t i = 0; i < 2; ++i)
                visitSegment(m_segmentsB[i], m_isStartB[i], false);
        }
    }
    else {
        // The joint's two rest lengths affect the closest vertex/theta/restlen of the corresponding incident segments
        assert(var < 2);
        if (var == 0) { visitSegment(m_segmentsA[0], m_isStartA[0], true); visitSegment(m_segmentsA[1], m_isStartA[1], true); }
        if (var == 1) { visitSegment(m_segmentsB[0], m_isStartB[0], true); visitSegment(m_segmentsB[1], m_isStartB[1], true); }
    }
}
template<typename Real_>
std::tuple<std::vector<std::vector<size_t>>,
           std::vector<std::vector<double>>,
           std::vector<std::vector<double>>>
RodLinkage_T<Real_>::rodStresses() const {
    std::vector<std::tuple<bool, std::vector<size_t>>> rods = traceRods();
    const size_t numRods = rods.size();

    // The stress values at distinct points along a rod.
    std::vector<std::vector<double>> stresses(numRods);

    // The parametric coordinates of each stress sample (i.e. each vertex) along the rod.
    // Joints are assigned sequential integer coordinates along the rod starting from 0 at the beginning.
    // The coordinates are interpolated linearly along the arc length of the rod segments.
    std::vector<std::vector<double>> paramValues(numRods);

    // Indices of the joints in order along a rod.
    std::vector<std::vector<size_t>> jointIndices(numRods);

    for (size_t ri = 0; ri < numRods; ++ri) {
        const auto &segment_indices = std::get<1>(rods[ri]);
        const size_t s_begin_idx = segment_indices.front();
        const auto &s_begin = m_segments[s_begin_idx];
        if (segment_indices.size() == 1) {
            jointIndices[ri].push_back(s_begin.startJoint);
            jointIndices[ri].push_back(s_begin.  endJoint);

            continue;
        }

        if (s_begin.hasStartJoint() + s_begin.hasEndJoint() == 0) throw std::runtime_error("Free ends unsupported");

        // Pick the correct orientation for the first segment.
        size_t  inContinuation = m_joints[s_begin.startJoint].continuationSegment(s_begin_idx);
        size_t outContinuation = m_joints[s_begin.endJoint  ].continuationSegment(s_begin_idx);
        if ((inContinuation == NONE) == (outContinuation == NONE)) throw std::runtime_error("Segment should be at one end, and single segment rod case should already have been handled... ");
        if (inContinuation == NONE) { // This segment is oriented in agreement with the rod
            jointIndices[ri].push_back(s_begin.startJoint);
        }
        else { // This segment is oriented opposite the rod
            jointIndices[ri].push_back(s_begin.  endJoint);
        }

        // Process each segment of the rod.
        // Invariant: one of the two joint end indices of the segment should already
        // be in jointIndices.
        double parametricOffset = 0; // offset of the current segment's start joint in the parametric domain
        for (const size_t si : segment_indices) {
            const auto &s = m_segments[si];
            bool reverse = false;
            if (s.hasStartJoint() + s.hasEndJoint() == 0) throw std::runtime_error("Free ends unsupported");
            if (jointIndices[ri].back() == s.startJoint) {
                jointIndices[ri].push_back(s.endJoint);
            }
            else {
                if (jointIndices[ri].back() != s.endJoint) throw std::runtime_error("Error traversing rod");
                jointIndices[ri].push_back(s.startJoint);
                reverse = true;
            }

            const auto &r = s.rod;
            const size_t nv = r.numVertices();
            const size_t ne = r.numVertices();
            auto perVertexEnergy = r.energyBendPerVertex();

            // Compute the total (deformed) length of the span between joints
            // This is half the length of the first and last edges plus the full length of the internal edges
            const auto &dc = r.deformedConfiguration();
            const auto &lens = dc.len;
            double internalLength = 0.5 * stripAutoDiff(lens.front()) + 0.5 * stripAutoDiff(lens.back());
            for (size_t i = 1; i < ne - 1; ++i)
                internalLength += stripAutoDiff(lens[i]);

            // Add sample points for each internal vertex of the rod segment
            double arclen = 0.5 * stripAutoDiff(reverse ? lens.back() : lens.front());
            int start = reverse ? nv - 2 : 1;
            int end   = reverse ? 0 : nv - 1; // non-inclusive
            int inc   = reverse ? -1 : 1;
            for (int vi = start; vi != end; vi += inc) {
                paramValues[ri].push_back(parametricOffset + arclen / internalLength);
                stresses   [ri].push_back(std::sqrt(stripAutoDiff(perVertexEnergy[vi])));
                arclen += stripAutoDiff(lens[reverse ? (vi - 1): vi]);
            }

           ++parametricOffset;
        }
    }
    return std::make_tuple(jointIndices, stresses, paramValues);
}


template<typename Real_>
void RodLinkage_T<Real_>::
florinVisualizationGeometry(std::vector<std::vector<size_t>> &polylinesA,
                            std::vector<std::vector<size_t>> &polylinesB,
                            std::vector<Point3D> &points, std::vector<Vector3D> &normals,
                            std::vector<Real> &stresses) const
{
    std::vector<std::tuple<bool, std::vector<size_t>>> rods = traceRods();
    const size_t numRods = rods.size();

    polylinesA.clear();
    polylinesB.clear();

    // Indices of the joints in order along a rod. This is used to traverse the rod...
    std::vector<std::vector<size_t>> jointIndices(numRods);

    auto add = [&](const Point3D &pt, const Vector3D &n, Real stress) {
        points.push_back(pt);
        normals.push_back(n);
        stresses.push_back(stress);
        return points.size() - 1;
    };

    for (size_t ri = 0; ri < numRods; ++ri) {
        const bool isA = std::get<0>(rods[ri]);
        if (isA) polylinesA.emplace_back();
        else     polylinesB.emplace_back();
        std::vector<size_t> &polyline = isA ? polylinesA.back() : polylinesB.back();

        const auto &segment_indices = std::get<1>(rods[ri]);
        const size_t s_begin_idx = segment_indices.front();
        const auto &s_begin = m_segments[s_begin_idx];

        if (s_begin.hasStartJoint() + s_begin.hasEndJoint() == 0) throw std::runtime_error("Free ends unsupported");

        // Pick the correct orientation for the first segment.
        size_t  inContinuation = m_joints[s_begin.startJoint].continuationSegment(s_begin_idx);
        if (inContinuation == NONE) { // This segment is oriented in agreement with the rod (or it's the only segment in the rod)
            jointIndices[ri].push_back(s_begin.startJoint);
        }
        else { // This segment is oriented opposite the rod
            jointIndices[ri].push_back(s_begin.  endJoint);
        }

        // Process each segment of the rod.
        // Invariant: one of the two joint end indices of the segment should already
        // be in jointIndices.
        for (size_t si_loc = 0; si_loc < segment_indices.size(); ++si_loc) {
            const auto &s = m_segments[segment_indices[si_loc]];

            bool reverse = false;
            if (s.hasStartJoint() + s.hasEndJoint() == 0) throw std::runtime_error("Free ends unsupported");
            if (jointIndices[ri].back() == s.startJoint) {
                jointIndices[ri].push_back(s.endJoint);
            }
            else {
                if (jointIndices[ri].back() != s.endJoint) throw std::runtime_error("Error traversing rod");
                jointIndices[ri].push_back(s.startJoint);
                reverse = true;
            }

            const auto &r = s.rod;
            const size_t nv = r.numVertices();
            const size_t ne = r.numEdges();
            auto perVertexEnergy = r.energyBendPerVertex();

            const auto &dc = r.deformedConfiguration();

            // We need to add the first vertex of the first segment on the polyline
            if (si_loc == 0) {
                int vi = reverse ? nv - 1 : 0;
                polyline.push_back(add(stripAutoDiff(r.deformedPoint(vi)),
                                   stripAutoDiff(dc.materialFrame[reverse ? ne - 1 : 0].d2),
                                   std::sqrt(stripAutoDiff(perVertexEnergy[vi]))));
            }

            // Add sample points for each internal vertex of the rod segment
            int start = reverse ? nv - 2 : 1;
            int end   = reverse ? 0 : nv - 1; // non-inclusive
            int inc   = reverse ? -1 : 1;
            for (int vi = start; vi != end; vi += inc) {
                polyline.push_back(add(stripAutoDiff(r.deformedPoint(vi)),
                                       stripAutoDiff((dc.materialFrame[vi    ].d2 + dc.materialFrame[vi - 1].d2).normalized().eval()),
                                       stripAutoDiff(std::sqrt(stripAutoDiff(perVertexEnergy[vi])))));
            }

            // We need to add the last vertex of the last segment on the polyline
            if (si_loc == segment_indices.size() - 1) {
                int vi = reverse ? 0 : nv - 1;
                polyline.push_back(add(stripAutoDiff(r.deformedPoint(vi)),
                                   stripAutoDiff(dc.materialFrame[reverse ? 0 : ne - 1].d2),
                                   std::sqrt(stripAutoDiff(perVertexEnergy[vi]))));
            }
        }
    }
}

#if MESHFEM_WITH_TBB

// Note: the following parallel_for + thread local data implementation is
// significantly faster than parallel_reduce. This is because parallel_reduce
// spawns way more tasks than worker threads, which incurs high split/join
// overhead due to the sparse matrix copy/accumulate.
template<typename Real_>
struct HessianAssemblerData {
    CSCMatrix<SuiteSparse_long, Real_> H;
    dv_dr_type<Real_> dv_dr;
    bool constructed = false;
};

template<typename Real_>
using HALocalData = tbb::enumerable_thread_specific<HessianAssemblerData<Real_>>;

template<typename F, typename Real_>
struct HessianAssembler {
    using CSCMat = CSCMatrix<SuiteSparse_long, Real_>;
    HessianAssembler(F &f, const CSCMat &H, HALocalData<Real_> &locals) : Hsp(H), m_f(f), m_locals(locals) { }

    void operator()(const tbb::blocked_range<size_t> &r) const {
        HessianAssemblerData<Real_> &data = m_locals.local();
        if (!data.constructed) { data.H.zeros_like(Hsp); data.constructed = true; }
        for (size_t si = r.begin(); si < r.end(); ++si) { m_f(si, data.dv_dr, data.H); }
    }

    const CSCMat &Hsp; // sparsity pattern for H
private:
    F &m_f;
    HALocalData<Real_> &m_locals;
};

template<typename F, typename Real_>
HessianAssembler<F, Real_> make_hessian_assembler(F &f, const CSCMatrix<SuiteSparse_long, Real_> &H, HALocalData<Real_> &locals) {
    return HessianAssembler<F, Real_>(f, H, locals);
}

#endif

template<typename Real_>
void RodLinkage_T<Real_>::hessian(CSCMat &H, EnergyType eType, const bool variableRestLen) const {
    assert(H.symmetry_mode == CSCMat::SymmetryMode::UPPER_TRIANGLE);
    BENCHMARK_SCOPED_TIMER_SECTION timer("RodLinkage Hessian");

    const size_t ndof = variableRestLen ? numExtendedDoF() : numDoF();
    assert((size_t(H.m) == ndof) && (size_t(H.n) == ndof));
    UNUSED(ndof);

    // Our Hessian can only be evaluated after the source configuration has
    // been updated; use the more efficient gradient formulas.
    const bool updatedSource = true;
    m_sensitivityCache.update(*this, updatedSource, true /* make sure the joint Hessian is cached */);

    // Assemble the (transformed) Hessian of each rod segment using the
    // gradients of the parameters with respect to the reduced parameters.
    auto assemblePerSegmentHessian = [&](size_t si, dv_dr_type<Real_> &dv_dr, CSCMat &Hout) {
        // BENCHMARK_START_TIMER_SECTION("Segment hessian preamble");
        const auto &s = m_segments[si];
        const auto &r = s.rod;

        // BENCHMARK_START_TIMER_SECTION("Rod hessian + grad");
        // Gradient and Hessian with respect to the segment's unconstrained DoFs
        auto sH = r.hessianSparsityPattern(variableRestLen);

        r.hessian(sH, eType, variableRestLen);
        const auto sg = r.gradient(updatedSource, eType); // we never need the variable rest length gradient since the mapping from global to local rest lengths is linear
        // BENCHMARK_STOP_TIMER_SECTION("Rod hessian + grad");

        size_t segmentDofOffset = m_dofOffsetForSegment[si];
        // BENCHMARK_STOP_TIMER_SECTION("Segment hessian preamble");

        // Sensitivity of terminal edges to the start/end joints (if they exist)
        // BENCHMARK_START_TIMER_SECTION("LinkageTerminalEdgeSensitivity");
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
        // BENCHMARK_STOP_TIMER_SECTION("LinkageTerminalEdgeSensitivity");

        // BENCHMARK_START_TIMER_SECTION("dv_dr_for_segment");
        dv_dr_for_segment(s, jointSensitivity, segmentJointDofOffset, segmentDofOffset, dv_dr, segmentJointRestLenDofOffset, variableRestLen ? m_restLenDoFOffsetForSegment[si] : NONE);
        // BENCHMARK_STOP_TIMER_SECTION("dv_dr_for_segment");

        // BENCHMARK_START_TIMER_SECTION("rod hessian contrib");
        // Accumulate contribution of each (upper triangle) entry in H to the
        // full Hessian term:
        //      dvk_dri sH_kl dvl_drj
        // This step still takes a majority of the time despite optimization efforts...
        // Entries in dv_dr tend to be contiguous, so we only use a binary search to find
        // the first output entry for a given column.
        using Idx = typename CSCMat::index_type;
        Idx idx = 0, idx2 = 0;
        Idx ncol = sH.n, colbegin = sH.Ap[0];
        for (Idx l = 0; l < ncol; ++l) {
            const Idx colend = sH.Ap[l + 1];
            for (auto entry = colbegin; entry < colend; ++entry) {
                const Idx k = sH.Ai[entry];
                const auto v = sH.Ax[entry];
                assert(k <= l);
                const auto &dvk_dr = dv_dr[k];
                const auto &dvl_dr = dv_dr[l];
                for (const auto &dvl_drj : dvl_dr) {
                    const Idx j = dvl_drj.first;
#if 1
                    {
                        const Idx i = dvk_dr[0].first;
                        if (i > j) continue;
                        if ((idx >= Hout.Ap[j + 1]) || (Hout.Ai[idx] != i) || (idx < Hout.Ap[j]))
                            idx = Hout.findEntry(i, j);
                    }
                    const auto val = dvl_drj.second * v;
                    Hout.Ax[idx++] += val * dvk_dr[0].second;
                    for (size_t ii = 1; ii < dvk_dr.size(); ++ii) {
                        const Idx i = dvk_dr[ii].first;
                        if (i > j) break;
                        while (Hout.Ai[idx] < i) ++idx;
                        Hout.Ax[idx++] += val * dvk_dr[ii].second;
                    }
#else
                    for (const auto &dvk_dri : dvk_dr) {
                        Idx i = dvk_dri.first;
                        if (i > j) break;
                        // Accumulate contributions from sH's upper triangle entry
                        // (k, l) and, if in the strict upper triangle, its
                        // corresponding lower triangle entry (l, k). This
                        // corresponding entry is found by exchanging i, j.
                        // Of course, we only keep contributions in the upper
                        // triangle of H.
                        hint = Hout.addNZ(i, j, dvk_dri.second * val, hint); // contribution from (k, l), if it falls in the upper triangle of H.
                    }
#endif
                }
                if (k != l) {
                    // Contribution from (l, k), if it falls in the upper triangle of H
                    for (const auto &dvl_drj : dvk_dr) {
                        const Idx j = dvl_drj.first;
#if 1
                        {
                            const Idx i = dvl_dr[0].first;
                            if (i > j) continue;
                            if ((idx2 >= Hout.Ap[j + 1]) || (Hout.Ai[idx2] != i) || (idx2 < Hout.Ap[j]))
                                idx2 = Hout.findEntry(i, j);
                        }
                        const auto val = dvl_drj.second * v;
                        Hout.Ax[idx2++] += val * dvl_dr[0].second;
                        for (size_t ii = 1; ii < dvl_dr.size(); ++ii) {
                            const Idx i = dvl_dr[ii].first;
                            if (i > j) break;
                            while (Hout.Ai[idx2] < i) ++idx2;
                            Hout.Ax[idx2++] += val * dvl_dr[ii].second;
                        }
#else
                        auto val = dvl_drj.second * v;
                        for (const auto &dvk_dri : dvl_dr) {
                            Idx i = dvk_dri.first;
                            if (i > j) break;
                            idx2 = Hout.addNZ(i, j, dvk_dri.second * val, idx2);
                        }
#endif
                    }
                }
            }
            colbegin = colend;
        }
        // BENCHMARK_STOP_TIMER_SECTION("rod hessian contrib");

        // BENCHMARK_START_TIMER_SECTION("joint hessian contrib");
        // Accumulate contribution of the Hessian of e^j and theta^j wrt the joint parameters.
        //      dE/var^j (d^2 var^j / djoint_var_k djoint_var_l)
        for (size_t ji = 0; ji < 2; ++ji) {
            if (jointSensitivity[ji] == nullptr) continue;
            const auto &js = *jointSensitivity[ji];
            const size_t o = segmentJointDofOffset[ji] + 3; // DoF index for first component of omega
            Vec3 dE_de_j = 0.5 * (sg.gradPos(js.j + 1) - sg.gradPos(js.j));
            Eigen::Matrix<Real_, 6, 6> contrib;
            contrib = (js.s_jX * dE_de_j[0]) * js.hessian[0]
                    + (js.s_jX * dE_de_j[1]) * js.hessian[1]
                    + (js.s_jX * dE_de_j[2]) * js.hessian[2]
                    +  sg.gradTheta(js.j)    * js.hessian[3];
            {
                size_t hint = Hout.findDiagEntry(o);
                for (size_t l = 0; l < 6; ++l) {
                    for (size_t k = 0; k <= l; ++k) {
                        Real_ val = contrib(k, l);
                        if (entryIdenticallyZero(val)) continue;
                        hint = Hout.addNZ(o + k, o + l, val, hint);
                    }
                }
            }

        }
        // BENCHMARK_STOP_TIMER_SECTION("joint hessian contrib");

        // BENCHMARK_STOP_TIMER("Accumulate Contributions");
    };

#if MESHFEM_WITH_TBB
    HALocalData<Real_> haLocalData;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, numSegments()), make_hessian_assembler(assemblePerSegmentHessian, H, haLocalData));
    // BENCHMARK_START_TIMER_SECTION("finalize");
    for (HessianAssemblerData<Real_> &data : haLocalData)
        H.addWithIdenticalSparsity(data.H);
    // BENCHMARK_STOP_TIMER_SECTION("finalize");
#else
    dv_dr_type<Real_> dv_dr;
    for (size_t si = 0; si < numSegments(); ++si) assemblePerSegmentHessian(si, dv_dr, H);
#endif
}

template<typename Real_>
auto RodLinkage_T<Real_>::hessian(EnergyType eType, bool variableRestLen) const -> TMatrix {
    auto H = hessianSparsityPattern(variableRestLen);
    hessian(H, eType, variableRestLen);
    return H.getTripletMatrix();
}

template<typename Real_>
void RodLinkage_T<Real_>::massMatrix(CSCMat &M, bool updatedSource, bool useLumped) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer("RodLinkage Mass");
    assert(M.symmetry_mode == CSCMat::SymmetryMode::UPPER_TRIANGLE);

    {
        // Also cache the joint parametrization Hessian if it can be computed accurately
        // (if the source frames are up-to-date); we should almost always want the Hessian
        // too when asking for the Mass matrix.
        const bool evalHessian = updatedSource;
        m_sensitivityCache.update(*this, updatedSource, evalHessian);
    }

    // Assemble the (transformed) mass matrix of each rod segment using the
    // gradients of the parameters with respect to the reduced parameters.
    auto assemblePerSegmentMassMatrix = [&](size_t si, dv_dr_type<Real_> &dv_dr, CSCMat &Mout) {
        const auto &s = m_segments[si];
        const auto &r = s.rod;

        size_t segmentDofOffset = m_dofOffsetForSegment[si];

        std::array<const LinkageTerminalEdgeSensitivity<Real_> *, 2> jointSensitivity{{ nullptr, nullptr }};
        std::array<size_t, 2> segmentJointDofOffset;
        for (size_t i = 0; i < 2; ++i) {
            size_t ji = s.joint(i);
            if (ji == NONE) continue;
            jointSensitivity[i] = &m_sensitivityCache.lookup(si, static_cast<TerminalEdge>(i));
            segmentJointDofOffset[i] = m_dofOffsetForJoint[ji];
        }

        dv_dr_for_segment(s, jointSensitivity, segmentJointDofOffset, segmentDofOffset, dv_dr);

        // Mass matrix with respect to the segment's unconstrained DoFs
        CSCMat sM;
        if (useLumped) { sM.setDiag(r.lumpedMassMatrix()); }
        else           { sM = r.hessianSparsityPattern();
                              r.massMatrix(sM); }

        // Accumulate contribution of each (upper triangle) entry in sM to the
        // full mass matrix term:
        //      dvk_dri M_kl dvl_drj
        size_t hint = 0;
        for (const auto &t : sM) {
            const size_t k = t.i, l = t.j;
            assert(k <= l);
            for (const auto &dvl_drj : dv_dr.at(l)) {
                size_t j = dvl_drj.first;
                for (const auto &dvk_dri : dv_dr.at(k)) {
                    size_t i = dvk_dri.first;
                    Real_ val = dvk_dri.second * dvl_drj.second * t.v;
                    // Accumulate contributions from sM's upper triangle entry
                    // (k, l) and, if in the strict upper triangle, its
                    // corresponding lower triangle entry (l, k). This
                    // corresponding entry is found by exchanging i, j.
                    // Of course, we only keep contributions in the upper
                    // triangle of M.
                    if ( i <= j             ) hint = Mout.addNZ(i, j, val, hint); // contribution from (k, l), if it falls in the upper triangle of H.
                    if ((j <= i) && (k != l)) hint = Mout.addNZ(j, i, val, hint); // contribution from (l, k), if it falls in the upper triangle of H and wasn't already added.
                }
            }
        }
    };

#if MESHFEM_WITH_TBB
    HALocalData<Real_> maLocalData;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, numSegments()), make_hessian_assembler(assemblePerSegmentMassMatrix, M, maLocalData));
    for (HessianAssemblerData<Real_> &data : maLocalData)
        M.addWithIdenticalSparsity(data.H);
#else
    dv_dr_type<Real_> dv_dr;
    for (size_t si = 0; si < numSegments(); ++si) assemblePerSegmentMassMatrix(si, dv_dr, M);
#endif
}

// Diagonal lumped mass matrix constructed by summing the mass in each row.
// WARNING: this matrix is usually not positive definite!
template<typename Real_>
auto RodLinkage_T<Real_>::lumpedMassMatrix(bool /* updatedSource */) const -> VecX {
    const size_t ndof = numDoF();
#if 0
    auto M = massMatrix(updatedSource, true);
    Eigen::VectorXd Mdiag = Eigen::VectorXd::Zero(ndof);

    for (const auto &t : M) {
        Mdiag[t.i] += t.v;
        if (t.j != t.i) Mdiag[t.j] += t.v;
    }

    if (Mdiag.minCoeff() <= 0) throw std::runtime_error("Lumped mass matrix is non-positive");

    return Mdiag;
#else
    return VecX::Ones(ndof);
#endif
}

template<typename Real_>
Real_ RodLinkage_T<Real_>::approxLinfVelocity(const VecX &paramVelocity) const {
    const size_t ndof = numDoF();
    TMatrix M(ndof, ndof);
    M.symmetry_mode = TMatrix::SymmetryMode::UPPER_TRIANGLE;

    const bool updatedSource = true; // The elastic rod approxLinfVelocity formulas already assume an updated source frame...
    {
        // We should almost always want the Hessian too when asking for the
        // Mass matrix...
        const bool evalHessian = updatedSource;
        m_sensitivityCache.update(*this, updatedSource, evalHessian);
    }

    VecX rodParamVelocity;

    // Assemble the (transformed) mass matrix of each rod segment using the
    // gradients of the parameters with respect to the reduced parameters.
    Real_ maxvel = 0;
    for (size_t si = 0; si < numSegments(); ++si) {
        const auto &s = m_segments[si];
        const auto &r = s.rod;
        const size_t nv = r.numVertices(), ne = r.numEdges();

        // Apply chain rule to determine the velocity of the rod's unconstrained DoFs
        rodParamVelocity.resize(r.numDoF());
        rodParamVelocity.setZero();

        // Copy over the velocities for the degrees of freedom that
        // directly control the interior/free-end centerline positions and
        // material frame angles.
        size_t offset = m_dofOffsetForSegment[si];
        for (size_t i = 0; i < nv; ++i) {
            // The first/last edge don't contribute degrees of freedom if they're part of a joint.
            if ((i <       2) && s.hasStartJoint()) continue;
            if ((i >= nv - 2) && s.  hasEndJoint()) continue;
            rodParamVelocity.template segment<3>(3 * i) = paramVelocity.template segment<3>(offset);
            offset += 3;
        }
        for (size_t j = 0; j < ne; ++j) {
            if ((j ==      0) && s.hasStartJoint()) continue;
            if ((j == ne - 1) && s.  hasEndJoint()) continue;
            rodParamVelocity[3 * nv + j] = paramVelocity[offset++];
        }

        // Set velocities induced by the start/end joints (if they exist)
        for (size_t i = 0; i < 2; ++i) {
            size_t jindex = s.joint(i);
            if (jindex == NONE) continue;
            const size_t offset = m_dofOffsetForJoint.at(jindex);
            const auto &sensitivity = m_sensitivityCache.lookup(si, static_cast<TerminalEdge>(i));
            const size_t j = sensitivity.j;
            //           pos        e_X     theta^j
            // x_j     [  I    -s_jX 0.5 I     0   ] [ I 0 ... 0]
            // x_{j+1} [  I     s_jX 0.5 I     0   ] [ jacobian ]
            // theta^j [  0          0         I   ]
            Vec3 d_edge = sensitivity.s_jX * sensitivity.jacobian.template block<3, 6>(0, 0) * paramVelocity.template segment<6>(offset + 3);
            rodParamVelocity.template segment<3>(3 * (j    )) = paramVelocity.template segment<3>(offset) - 0.5 * d_edge;
            rodParamVelocity.template segment<3>(3 * (j + 1)) = paramVelocity.template segment<3>(offset) + 0.5 * d_edge;
            rodParamVelocity                    [3 * nv + j]  = sensitivity.jacobian.template block<1, 6>(3, 0) * paramVelocity.template segment<6>(offset + 3);
        }

        maxvel = std::max(maxvel, r.approxLinfVelocity(rodParamVelocity));
    }

    return maxvel;
}

template<typename Real_>
VecX_T<Real_> RodLinkage_T<Real_>::gravityForce(Real_ rho, const Vec3_T<Real_> &g) const {
    VecX result(VecX::Zero(numDoF()));
    VecX ge;

    for (size_t si = 0; si < numSegments(); ++si) {
        const auto &s = segment(si);
        ge = s.rod.gravityForce(rho, g);
        const size_t nfv = s.numFreeVertices();
        // Copy over gravity forces on the "free" vertices.
        result.segment(dofOffsetForSegment(si), 3 * nfv) = ge.segment((s.hasStartJoint() * 2) * 3, 3 * nfv);

        // Accumulate contributions to the start/end joints (if they exist)
        for (size_t lji = 0; lji < 2; ++lji) {
            size_t jindex = s.joint(lji);
            if (jindex == NONE) continue;
            const size_t j = (lji == 0) ? 0 : s.rod.numEdges() - 1;

            result.template segment<3>(dofOffsetForJoint(jindex)) +=
                ge.template segment<3>(3 * (j    )) +
                ge.template segment<3>(3 * (j + 1));
        }
    }

    return result;
}

////////////////////////////////////////////////////////////////////////////////
// 1D uniform Laplacian regularization energy for the rest length optimization:
//      0.5 * sum_i (lbar^{i} - lbar^{i - 1})^2
////////////////////////////////////////////////////////////////////////////////
template<typename Real_>
Real_ RodLinkage_T<Real_>::restLengthLaplacianEnergy() const {
    Real_ result = 0;
    for (const auto &s : m_segments)
        result += s.rod.restLengthLaplacianEnergy();
    return result;
}

template<typename Real_>
auto RodLinkage_T<Real_>::restLengthLaplacianGradEnergy() const -> VecX {
    VecX g = VecX::Zero(numRestLengths());
    // Get offset of the first joint rest length variable (within the full list of rest lengths)
    size_t jointRestLenDoFOffset = 0;
    for (const auto &s : m_segments) { jointRestLenDoFOffset += s.numFreeEdges(); }

    size_t offset = 0;
    for (size_t si = 0; si < numSegments(); ++si) {
        const auto &s = segment(si);
        const size_t nfe = s.numFreeEdges();
        VecX sg = s.rod.restLengthLaplacianGradEnergy();
        // Copy over direct dependence on free rest lengths.
        g.segment(offset, nfe) = sg.segment(s.hasStartJoint(), nfe);
        if (s.hasStartJoint()) {
            size_t abOffset = joint(s.startJoint).segmentABOffset(si);
            assert(abOffset != NONE);
            g[jointRestLenDoFOffset + 2 * s.startJoint + abOffset] += sg[0];
        }
        if (s.hasEndJoint()) {
            size_t abOffset = joint(s.endJoint).segmentABOffset(si);
            assert(abOffset != NONE);
            g[jointRestLenDoFOffset + 2 * s.endJoint + abOffset] += sg[s.rod.numEdges() - 1];
        }
        offset += nfe;
    }

    return g;
}

template<typename Real_>
auto RodLinkage_T<Real_>::restLengthLaplacianHessEnergy() const -> TMatrix {
    const size_t nrl = numRestLengths();
    TMatrix result(nrl, nrl);
    result.symmetry_mode = TMatrix::SymmetryMode::UPPER_TRIANGLE;

    {
        size_t nnz_prediction = 0;
        for (const auto &s : m_segments) {
            size_t n = s.rod.numRestLengths();
            nnz_prediction += n + (n - 1); // individual segments' Laplacian matrices are symmetric tridiagonal.
        }
        result.reserve(nnz_prediction);
    }

    // Get offset of the first joint rest length variable (within the full list of rest lengths)
    size_t jointRestLenDoFOffset = 0;
    for (const auto &s : m_segments) { jointRestLenDoFOffset += s.numFreeEdges(); }

    size_t offset = 0;
    for (size_t si = 0; si < numSegments(); ++si) {
        const auto &s = segment(si);
        const size_t ne  = s.rod.numEdges();
        TMatrix sH = s.rod.restLengthLaplacianHessEnergy();

        auto global_idx = [&](size_t li) {
            if ((li == 0) && s.hasStartJoint()) {
                size_t abOffset = joint(s.startJoint).segmentABOffset(si);
                assert(abOffset != NONE);
                return jointRestLenDoFOffset + 2 * s.startJoint + abOffset;
            }
            if ((li == ne - 1) && s.hasEndJoint()) {
                size_t abOffset = joint(s.endJoint).segmentABOffset(si);
                assert(abOffset != NONE);
                return jointRestLenDoFOffset + 2 * s.endJoint + abOffset;
            }
            return offset + li - s.hasStartJoint();
        };

        for (const auto &t : sH) {
            assert(t.i <= t.j);
            // Convert local rest length index to global
            size_t gi = global_idx(t.i),
                   gj = global_idx(t.j);
            if (gi > gj) std::swap(gi, gj);
            result.addNZ(gi, gj, t.v);
        }
        offset += s.numFreeEdges();
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////////
// Hessian matvec implementation
////////////////////////////////////////////////////////////////////////////////
#include "RodLinkageHessVec.inl"

////////////////////////////////////////////////////////////////////////////////
// Expose TerminalEdgeSensitivity for debugging
////////////////////////////////////////////////////////////////////////////////
template<typename Real_>
struct LinkageTerminalEdgeSensitivity;

template<typename Real_>
const LinkageTerminalEdgeSensitivity<Real_> &RodLinkage_T<Real_>::getTerminalEdgeSensitivity(size_t si, TerminalEdge which, bool updatedSource, bool evalHessian) {
    m_sensitivityCache.update(*this, updatedSource, evalHessian);
    assert(si < numSegments());
    assert(m_segments[si].joint(static_cast<int>(which)) != NONE);
    return m_sensitivityCache.lookup(si, which);
}

// Out-of-line constructor and destructor needed because LinkageTerminalEdgeSensitivity<Real_> is an incomplete type upon declaration of sensitivityForTerminalEdge.
template<typename Real_> RodLinkage_T<Real_>::SensitivityCache:: SensitivityCache() { }
template<typename Real_> RodLinkage_T<Real_>::SensitivityCache::~SensitivityCache() { }

template<typename Real_>
void RodLinkage_T<Real_>::SensitivityCache::clear() { sensitivityForTerminalEdge.clear(); evaluatedHessian = false; evaluatedWithUpdatedSource = true; }

template<typename Real_>
void RodLinkage_T<Real_>::SensitivityCache::update(const RodLinkage_T &l, bool updatedSource, bool evalHessian) {
    if (evalHessian && !updatedSource) throw std::runtime_error("Hessian formulas only accurate if source frames are updated");
    if (!sensitivityForTerminalEdge.empty() && (evaluatedWithUpdatedSource == updatedSource) && (evaluatedHessian || !evalHessian)) return;
    evaluatedWithUpdatedSource = updatedSource;
    evaluatedHessian = evalHessian;
    const size_t ns = l.numSegments();
    sensitivityForTerminalEdge.resize(2 * ns);
    auto processSegment = [this, evalHessian, updatedSource, &l](size_t si) {
        const auto &s = l.segment(si);
        size_t ji = s.joint(0); if (ji != NONE) sensitivityForTerminalEdge[2 * si + 0].update(l.joint(ji), si, s.rod, updatedSource, evalHessian);
               ji = s.joint(1); if (ji != NONE) sensitivityForTerminalEdge[2 * si + 1].update(l.joint(ji), si, s.rod, updatedSource, evalHessian);
    };
#if MESHFEM_WITH_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, ns), [&](const tbb::blocked_range<size_t> &b) { for (size_t si = b.begin(); si < b.end(); ++si) processSegment(si); });
#else
    for (size_t si = 0; si < ns; ++si) processSegment(si);
#endif
}

template<typename Real_>
void RodLinkage_T<Real_>::SensitivityCache::update(const RodLinkage_T &l, bool updatedSource, const VecX &delta_params) {
    if (!updatedSource) throw std::runtime_error("Hessian formulas only accurate if source frames are updated");
    // If the full joint Hessian is cached and up-to-date, use it to compute the directional derivatives
    const bool validHessianCache = !sensitivityForTerminalEdge.empty() && (evaluatedWithUpdatedSource == updatedSource) && evaluatedHessian;
    evaluatedWithUpdatedSource = updatedSource;
    evaluatedHessian           = validHessianCache; // We only keep the cached Hessian if it is still valid. We do not cache a new one.
    const size_t ns = l.numSegments();
    sensitivityForTerminalEdge.resize(2 * ns);
    auto processSegment = [this, updatedSource, validHessianCache, &l, &delta_params](size_t si) {
        const auto &s = l.segment(si);
        for (size_t lji = 0; lji < 2; ++lji) {
            const size_t ji = s.joint(lji);
            if (ji == NONE) continue;
            const size_t offset = l.dofOffsetForJoint(ji) + 3; // DoF index for first omega variable
            auto &tes = sensitivityForTerminalEdge[2 * si + lji];
            const auto &delta_jparams = delta_params.template segment<6>(offset);
            if (validHessianCache) {
                tes.delta_jacobian.row(0) = delta_jparams.transpose() * tes.hessian[0];
                tes.delta_jacobian.row(1) = delta_jparams.transpose() * tes.hessian[1];
                tes.delta_jacobian.row(2) = delta_jparams.transpose() * tes.hessian[2];
                tes.delta_jacobian.row(3) = delta_jparams.transpose() * tes.hessian[3];
            }
            else {
                tes.update(l.joint(ji), si, s.rod, updatedSource, true, delta_jparams);
            }
        }
    };
#if MESHFEM_WITH_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, ns), [&](const tbb::blocked_range<size_t> &b) { for (size_t si = b.begin(); si < b.end(); ++si) processSegment(si); });
#else
    for (size_t si = 0; si < ns; ++si) processSegment(si);
#endif
}

////////////////////////////////////////////////////////////////////////////////
// I/O for Visualization/Debugging
////////////////////////////////////////////////////////////////////////////////
template<typename Real_>
void RodLinkage_T<Real_>::visualizationGeometry(std::vector<MeshIO::IOVertex > &vertices,
                                                std::vector<MeshIO::IOElement> &quads, const bool averagedMaterialFrames) const {
    for (const auto &s : m_segments)
        s.rod.visualizationGeometry(vertices, quads, averagedMaterialFrames);
}

template<typename Real_>
void RodLinkage_T<Real_>::saveVisualizationGeometry(const std::string &path, const bool averagedMaterialFrames) const {
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> quads;
    visualizationGeometry(vertices, quads, averagedMaterialFrames);
    MeshIO::save(path, vertices, quads, MeshIO::FMT_GUESS, MeshIO::MESH_QUAD);
}

template<typename Real_>
void RodLinkage_T<Real_>::saveStressVisualization(const std::string &path) const {
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> quads;
#if 0
    Eigen::VectorXd sqrtBendingEnergy,
                    stretchingStress,
                    maxBendingStress,
                    minBendingStress,
                    twistingStress;
    for (const auto &s : m_segments) {
        s.rod.stressVisualizationGeometry(vertices, quads, sqrtBendingEnergy, stretchingStress, maxBendingStress, minBendingStress, twistingStress);
    }

    MSHFieldWriter writer(path, vertices, quads, MeshIO::MESH_QUAD);
    writer.addField("sqrt bending energy", ScalarField<Real>(sqrtBendingEnergy), DomainType::PER_NODE);
    writer.addField("stretching stress",   ScalarField<Real>(stretchingStress),  DomainType::PER_NODE);
    writer.addField("max bending stress",  ScalarField<Real>(maxBendingStress),  DomainType::PER_NODE);
    writer.addField("min bending stress",  ScalarField<Real>(minBendingStress),  DomainType::PER_NODE);
    writer.addField("twisting stress",     ScalarField<Real>(twistingStress),    DomainType::PER_NODE);
#endif
    visualizationGeometry(vertices, quads);
    MSHFieldWriter writer(path, vertices, quads, MeshIO::MESH_QUAD);
    writer.addField("sqrt bending energy", ScalarField<Real>(visualizationField(sqrtBendingEnergies())), DomainType::PER_NODE);
    writer.addField("stretching stress",   ScalarField<Real>(visualizationField(stretchingStresses ())), DomainType::PER_ELEMENT);
    writer.addField("max bending stress",  ScalarField<Real>(visualizationField(maxBendingStresses ())), DomainType::PER_NODE);
    writer.addField("min bending stress",  ScalarField<Real>(visualizationField(minBendingStresses ())), DomainType::PER_NODE);
    writer.addField("twisting stress",     ScalarField<Real>(visualizationField(twistingStresses   ())), DomainType::PER_NODE);

}

template<typename Real_>
void RodLinkage_T<Real_>::writeRodDebugData(const std::string &path, const size_t singleRod) const {
    // Extract line mesh for each elastic rod in the linkage
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    for (size_t si = 0, offset = 0; si < m_segments.size(); ++si) {
        if ((singleRod != NONE) && (si != singleRod)) continue;
        const auto &s = m_segments[si];
        for (auto &p : s.rod.deformedPoints())        vertices.emplace_back(stripAutoDiff(p));
        for (size_t i = 0; i < s.rod.numEdges(); ++i) elements.emplace_back(offset + i, offset + i + 1);
        offset = vertices.size();
    }

    const size_t ne = elements.size(),
                 nv = vertices.size();

    // Extract internal state of each elastic rod as vector and scalar fields on the line meshes.
    VectorField<double, 3> referenceD1(ne), materialD1(ne), curvatureBinormal(nv);
    ScalarField<double   > referenceTwist(nv), theta(ne), restLen(ne), len(ne);
    ScalarField<double   > bendStiffness1(nv), bendStiffness2(nv), twistStiffness(nv);

    {
        size_t vtxOffset = 0, edgeOffset = 0;
        for (size_t si = 0; si < m_segments.size(); ++si) {
            if ((singleRod != NONE) && (si != singleRod)) continue;
            const auto &s = m_segments[si];
            const auto &r = s.rod;
            const auto &dc = s.rod.deformedConfiguration();

            for (size_t j = 0; j < r.numEdges(); ++j) {
                referenceD1(edgeOffset + j) = stripAutoDiff(dc.referenceDirectors[j].d1);
                materialD1 (edgeOffset + j) = stripAutoDiff(dc.materialFrame[j].d1);
                theta      [edgeOffset + j] = stripAutoDiff(dc.theta(j));
                len        [edgeOffset + j] = stripAutoDiff(dc.len[j]);
                restLen    [edgeOffset + j] = stripAutoDiff(r.restLengths()[j]);
            }

            for (size_t i = 0; i < r.numVertices(); ++i) {
                referenceTwist   [vtxOffset + i] = stripAutoDiff(dc.referenceTwist[i]);
                curvatureBinormal(vtxOffset + i) = stripAutoDiff(dc.kb[i]);
                bendStiffness1   [vtxOffset + i] = stripAutoDiff(r.bendingStiffness(i).lambda_1);
                bendStiffness2   [vtxOffset + i] = stripAutoDiff(r.bendingStiffness(i).lambda_2);
                twistStiffness   [vtxOffset + i] = stripAutoDiff(r.twistingStiffness(i));
            }

            vtxOffset  += r.numVertices();
            edgeOffset += r.numEdges();
        }
    }

    MSHFieldWriter writer(path, vertices, elements);

    writer.addField("rest len",       restLen,           DomainType::PER_ELEMENT);
    writer.addField("len",            len,               DomainType::PER_ELEMENT);
    writer.addField("theta",          theta,             DomainType::PER_ELEMENT);
    writer.addField("reference d1",   referenceD1,       DomainType::PER_ELEMENT);
    writer.addField("material d1",    materialD1,        DomainType::PER_ELEMENT);
    writer.addField("referenceTwist", referenceTwist,    DomainType::PER_NODE);
    writer.addField("kb",             curvatureBinormal, DomainType::PER_NODE);
    writer.addField("bendStiffness1", bendStiffness1,    DomainType::PER_NODE);
    writer.addField("bendStiffness2", bendStiffness2,    DomainType::PER_NODE);
    writer.addField("twistStiffness", twistStiffness,    DomainType::PER_NODE);
}

template<typename Real_>
void RodLinkage_T<Real_>::writeLinkageDebugData(const std::string &path) const {
    // Extract line mesh for each elastic rod in the linkage
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    const size_t nj = numJoints();
    for (const auto &j : m_joints)
        vertices.push_back(stripAutoDiff(j.pos()));

    // Convert each rod segment into a single line, creating vertices for the free ends.
    auto genVtx = [&vertices](const Pt3 &p) { vertices.emplace_back(stripAutoDiff(p)); return vertices.size() - 1; };
    for (const auto &s : m_segments) {
        size_t a = s.startJoint, b = s.endJoint;
        if (a == NONE) a = genVtx(s.rod.deformedPoints().front());
        if (b == NONE) b = genVtx(s.rod.deformedPoints().back());
        elements.emplace_back(a, b);
    }

    // Output joint information on the vertices.
    const size_t nv = vertices.size();
    VectorField<double, 3> eA(nv), eB(nv), n(nv);
    eA.clear(), eB.clear(), n.clear();

    for (size_t i = 0; i < nj; ++i) {
        eA(i) = stripAutoDiff(m_joints[i].e_A());
        eB(i) = stripAutoDiff(m_joints[i].e_B());
         n(i) = stripAutoDiff(m_joints[i].normal());
    }

    MSHFieldWriter writer(path, vertices, elements);

    writer.addField("eA",  eA, DomainType::PER_NODE);
    writer.addField("eB",  eB, DomainType::PER_NODE);
    writer.addField("n",    n, DomainType::PER_NODE);

    // Computing rivet torques will throw if the joint parametrization hasn't
    // been updated; just silently suppress rivet force output in this case.
    try {
        Eigen::MatrixXd rf = rivetNetForceAndTorques();
        VectorField<double, 3> jointForce(nj), jointTorque(nj);
        for (size_t j = 0; j < nj; ++j) {
            // pos, omega, alpha
            jointForce(j)  = rf.block<1, 3>(j, 0);
            jointTorque(j) = rf.block<1, 3>(j, 3);
        }

        writer.addField("joint force",  jointForce,  DomainType::PER_NODE);
        writer.addField("joint torque", jointTorque, DomainType::PER_NODE);
    }
    catch (...) { }
}

template<typename Real_>
template<class F>
void RodLinkage_T<Real_>::Joint::visitNeighbors(const F &f, const size_t restrict_AB) const {
    assert(m_linkage);
    auto neighbor = [&](size_t si, bool isStart) { return m_linkage->segment(si).joint(isStart); };
    for (size_t i = 0; i < 2; ++i) {
        if ((restrict_AB != 1) && (m_segmentsA[i] != NONE)) { size_t si = m_segmentsA[i]; size_t ni = neighbor(si, m_isStartA[i]); if (ni != NONE) f(ni, si, m_linkage->joint(ni).segmentABOffset(si)); }
        if ((restrict_AB != 0) && (m_segmentsB[i] != NONE)) { size_t si = m_segmentsB[i]; size_t ni = neighbor(si, m_isStartB[i]); if (ni != NONE) f(ni, si, m_linkage->joint(ni).segmentABOffset(si)); }
    }
}

template<typename Real_>
void RodLinkage_T<Real_>::triangulation(std::vector<MeshIO::IOVertex> &vertices, std::vector<MeshIO::IOElement> &tris, std::vector<size_t> &originJoint) const {
    // First, extract the quads formed by the rods. Vertices of this quad mesh are the linkage joints.
    // The quads touching a vertex contain that vertex and vertices 1 segment
    // away (on the same rod) and two segments away *on a different rod*:
    //  |   |   |
    // -+---o---+-
    //  |   |   |
    // -o---x---o-
    //  |   |   |
    // -+---o---+-
    //  |   |   |
    vertices.clear();
    vertices.reserve(numJoints());
    for (const auto &j : m_joints) vertices.emplace_back(stripAutoDiff(j.pos()));

    // Note: array has a lexicographic comparison
    std::vector<std::array<size_t, 4>> quads;

    // Pretty inefficient...
    for (size_t ji = 0; ji < numJoints(); ++ji) {
        // Locate the "o" vertices one segment away:
        std::set<size_t> adj;
        joint(ji).visitNeighbors([&](size_t ji_o, size_t, size_t) { adj.insert(ji_o); });

        // Create a quad with "x", two of the "o" vertices, and a "+" vertex
        auto visit_o = [&](size_t ji_o, size_t, size_t AB_o) {
            auto visit_plus = [&](size_t ji_p, size_t, size_t AB_plus) {
                auto add_quad = [&](size_t ji_o2, size_t, size_t) {
                    if (adj.count(ji_o2) && (ji_o2 != ji_o)) {
                        quads.emplace_back(std::array<size_t, 4>{{ji, ji_o, ji_p, ji_o2}}); // found a loop of four vertices in the stencil
                    }
                };
                if (ji_p != ji) joint(ji_p).visitNeighbors(add_quad, (AB_plus + 1) % 2);
            };
            joint(ji_o).visitNeighbors(visit_plus, (AB_o + 1) % 2);
        };
        joint(ji).visitNeighbors(visit_o);
    }

    // Deduplicate all the added quads by permuting their indices into a canonical order.
    // Cycle the quads' indices so that they start with the smallest index.
    for (auto &q : quads) std::rotate(q.begin(), std::min_element(q.begin(), q.end()), q.end());
    // Now the order is determined up to an orientation reversal (swap of second and last index)
    for (auto &q : quads) { if (q[1] > q[3]) std::swap(q[1], q[3]); }
    // Remove duplicates
    std::sort(quads.begin(), quads.end());
    quads.erase(std::unique(quads.begin(), quads.end()), quads.end());

    // Construct a regular triangulation of the quads (so that loop subdivision will produce a C2 surface):
    // +---+---+       +---+---+
    // |   |   |       | / | / |
    // +---+---+  ==>  +---+---+
    // |   |   |       | / | / |
    // +---+---+       +---+---+
    // Propagate the choice of diagonal from a single quad with a BFS.
    // First, we must construct the quads' an edge-based adjacency list
    std::vector<std::vector<size_t>> adj(quads.size());
    {
        std::map<UnorderedPair, size_t> edgeMatcher;
        for (size_t qi = 0; qi < quads.size(); ++qi) {
            const auto &q = quads[qi];
            for (size_t ei = 0; ei < 4; ++ei) {
                UnorderedPair key(q[ei], q[(ei + 1) % 4]);
                auto it = edgeMatcher.find(key);
                if (it == edgeMatcher.end()) { edgeMatcher.emplace(key, qi); }
                else                         { adj[qi].push_back(it->second); adj[it->second].push_back(qi); }
            }
        }
    }
    // Next we triangulate the quads in BFS order
    // We pick a seed quad and try both choices of diagonal. We pick
    // the triangulation with the lowest total edge length
    auto triangulate_surface = [&](size_t seed_diag) {
        std::vector<MeshIO::IOElement> candidate_tris;
        candidate_tris.reserve(2 * quads.size());
        // 3-2    3-2
        // |/| or |\|
        // 0-1    0-1
        // if diag = 0 or 1, respectively.
        auto triangulate_quad = [&](size_t qi, size_t diag) {
            candidate_tris.emplace_back(quads[qi][diag], quads[qi][diag + 1], quads[qi][diag + 2]);
            candidate_tris.emplace_back(quads[qi][diag], quads[qi][diag + 2], quads[qi][(diag + 3) % 4]);
        };

        std::vector<size_t> pickedDiagonal(quads.size());
        std::vector<bool> visited(quads.size(), false);
        std::queue<size_t> bfsQueue;
        bfsQueue.push(0);
        visited[0] = true;
        pickedDiagonal[0] = seed_diag;
        triangulate_quad(0, seed_diag);

        size_t numVisited = 1;
        while (!bfsQueue.empty()) {
            size_t u = bfsQueue.front();
            bfsQueue.pop();
            const auto &qu = quads[u];
            size_t pd_u = pickedDiagonal[u];
            for (size_t v : adj[u]) {
                if (visited[v]) continue;
                visited[v] = true;
                ++numVisited;
                // +---+---+
                // | / | / |
                // +---+---+
                // We want to pick the diagonal not touching our neighbor's diagonal.
                const auto &qv = quads[v];
                size_t blocked_vtx = std::min(std::distance(qv.begin(), std::find(qv.begin(), qv.end(), qu[pd_u])),
                                              std::distance(qv.begin(), std::find(qv.begin(), qv.end(), qu[(pd_u + 2) % 4])));
                if (blocked_vtx >= 4) throw std::runtime_error("Diagonal of neighboring quad is not incident this quad!");
                pickedDiagonal[v] = (blocked_vtx + 1) % 2;
                triangulate_quad(v, pickedDiagonal[v]);
                bfsQueue.push(v);
            }
        }

        if (numVisited < quads.size()) {
            std::cerr << "Warning: disconnected quads" << std::endl;
            // TODO: additional BFS passes. For now, just triangulate arbitrarily.
            for (size_t qi = 0; qi < quads.size(); ++qi)
                if (!visited[qi]) triangulate_quad(qi, 0);
        }
        return candidate_tris;
    };

    // Pick the triangulation yielding the shortest total edge length.
    {
        auto tris_0 = triangulate_surface(0);
        auto tris_1 = triangulate_surface(1);
        Real edgeLen0 = 0, edgeLen1 = 0;
        for (size_t ti = 0; ti < tris_0.size(); ++ti) {
            for (size_t ei = 0; ei < 3; ++ei) {
                edgeLen0 += (vertices[tris_0[ti][(ei + 1) % 3]].point - vertices[tris_0[ti][ei]].point).norm();
                edgeLen1 += (vertices[tris_1[ti][(ei + 1) % 3]].point - vertices[tris_1[ti][ei]].point).norm();
            }
        }
        if (edgeLen0 < edgeLen1) tris = std::move(tris_0);
        else                     tris = std::move(tris_1);
    }

#if 0
    // Old irregular triangulation approach.
    vertices.reserve(vertices.size() + quads.size());
    tris.clear();
    tris.reserve(4 * quads.size());
    for (const auto &q : quads) {
        size_t vnew = vertices.size();
        vertices.emplace_back((0.25 * (vertices.at(q[0]).point + vertices.at(q[1]).point + vertices.at(q[2]).point + vertices.at(q[3]).point)).eval());
        tris.emplace_back(q[0], q[1], vnew);
        tris.emplace_back(q[1], q[2], vnew);
        tris.emplace_back(q[2], q[3], vnew);
        tris.emplace_back(q[3], q[0], vnew);
    }
#endif

    originJoint.assign(vertices.size(), size_t(NONE));
    for (size_t i = 0; i < originJoint.size(); ++i) {
        if (i < numJoints()) originJoint[i] = i;
    }

    // Use the joint normals to orient the triangles (majority vote)
    for (size_t ti = 0; ti < tris.size(); ++ti) {
        auto &t = tris[ti];
        Vec3 n = (vertices.at(t[1]).point - vertices.at(t[0]).point)
           .cross(vertices.at(t[2]).point - vertices.at(t[0]).point);
#if 0
        // for midpoint insertion triangulation
        size_t qi = ti / 4;
        size_t agree = 0;
        for (size_t i = 0; i < 4; ++i) agree += (n.dot(joint(quads[qi][i]).normal()) > 0);
        if (agree < 2) std::swap(t[0], t[1]);
#else
        size_t agree = 0;
        for (size_t i = 0; i < 3; ++i) agree += (n.dot(joint(t[i]).normal()) > 0);
        if (agree < 2) std::swap(t[0], t[1]);
#endif
    }
}

////////////////////////////////////////////////////////////////////////////////
// Explicit instantiation for ordinary double type and autodiff type.
////////////////////////////////////////////////////////////////////////////////
template struct RodLinkage_T<Real>;
template struct RodLinkage_T<ADReal>;
// template RodLinkage_T<ADReal>::RodLinkage_T<Real>(const RodLinkage_T<Real> &);
