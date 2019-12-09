#include "TargetSurfaceFitter.hh"
#include "infer_target_surface.hh"
#include <MeshFEM/TriMesh.hh>
#include <MeshFEM/MeshIO.hh>
#include <igl/per_face_normals.h>
#include <igl/point_simplex_squared_distance.h>
#include <igl/AABB.h>

struct TargetSurfaceMesh : public TriMesh<> {
    using TriMesh<>::TriMesh;
};

struct TargetSurfaceAABB : public igl::AABB<Eigen::MatrixXd, 3> {
    using Base = igl::AABB<Eigen::MatrixXd, 3>;
    using Base::Base;
};

void TargetSurfaceFitter::constructTargetSurface(const RodLinkage &linkage, size_t loop_subdivisions) {
    try {
        infer_target_surface(linkage, m_tgt_surf_V, m_tgt_surf_F, /* smoothing iterations */ loop_subdivisions, /* num extension layers */ 2);
    }
    catch (std::exception &e) {
        std::cerr << "ERROR: failed to infer target surface: " << e.what() << std::endl;
        std::cerr << "You must load a new target surface or set the joint position weight to 1.0" << std::endl;

        std::vector<MeshIO::IOVertex > vertices;
        std::vector<MeshIO::IOElement> quads;
        linkage.visualizationGeometry(vertices, quads);
        BBox<Eigen::Vector3d> bb(vertices);
        Eigen::Vector3d minC = bb.minCorner;
        Eigen::Vector3d maxC = bb.maxCorner;

        // We need *some* target surface with the same general position/scale
        // as the linkage or the libigl-based viewer will not work properly.
        m_tgt_surf_V.resize(3, 3);
        m_tgt_surf_F.resize(1, 3);
        m_tgt_surf_V << minC[0], minC[1], minC[2],
                        maxC[0], minC[1], minC[2],
                        maxC[0], maxC[1], maxC[2];
        m_tgt_surf_F << 0, 1, 2;
    }
    setTargetSurface(linkage, m_tgt_surf_V, m_tgt_surf_F);
}

void TargetSurfaceFitter::setTargetSurface(const RodLinkage &linkage, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) {
    m_tgt_surf_V = V;
    m_tgt_surf_F = F;
    igl::per_face_normals(m_tgt_surf_V, m_tgt_surf_F, m_tgt_surf_N);

    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    igl_to_meshio(m_tgt_surf_V, m_tgt_surf_F, vertices, elements);
    target_surface = std::make_unique<TargetSurfaceMesh>(elements, vertices.size());

    m_tgt_surf_aabb_tree = std::make_unique<TargetSurfaceAABB>();
    m_tgt_surf_aabb_tree->init(m_tgt_surf_V, m_tgt_surf_F);

    updateClosestPoints(linkage);

    static size_t i = 0;
    igl_to_meshio(m_tgt_surf_V, m_tgt_surf_F, vertices, elements);
    MeshIO::save("target_surface_" + std::to_string(i) + ".msh", vertices, elements);
    linkage.writeLinkageDebugData("linkage_" + std::to_string(i) + ".msh");
    ++i;
}

void TargetSurfaceFitter::loadTargetSurface(const RodLinkage &linkage, const std::string &path) {
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    MeshIO::load(path, vertices, elements);
    std::cout << "Loaded " << vertices.size() << " vertices and " << elements.size() << " triangles" << std::endl;
    meshio_to_igl(vertices, elements, m_tgt_surf_V, m_tgt_surf_F);
    setTargetSurface(linkage, m_tgt_surf_V, m_tgt_surf_F);
}

void TargetSurfaceFitter::updateClosestPoints(const RodLinkage &linkage) {
    const size_t nj = linkage.numJoints();

    // If we have nonzero weights in the surface-fitting term,
    // or if the closest point array is uninitialized,
    // update each joint's closest surface point.
    if ((size_t(joint_closest_surf_pts.size()) == 3 * nj) && (Wsurf_diag_joint_pos.norm() == 0.0)) return;

    BENCHMARK_SCOPED_TIMER_SECTION timer("Update closest points");
    joint_closest_surf_pts.resize(3 * nj);
    joint_closest_surf_pt_sensitivities.resize(nj);
    joint_closest_surf_tris.resize(nj);
    int numInterior = 0, numBdryEdge = 0, numBdryVtx = 0;
    for (size_t ji = 0; ji < nj; ++ji) {
        int closest_idx;
        const auto &j = linkage.joint(ji);
        // Could be parallelized (libigl does this internally for multi-point queries)
        Eigen::RowVector3d p, query;
        query = j.pos().transpose();
        Real sqdist = m_tgt_surf_aabb_tree->squared_distance(m_tgt_surf_V, m_tgt_surf_F, query, closest_idx, p);
        joint_closest_surf_pts    .segment<3>(3 * ji) = p.transpose();

        // Compute the sensitivity of the closest point projection with respect to the query point (dp_dx).
        // There are three cases depending on whether the closest point lies in the target surface's
        // interior, on one of its boundary edges, or on a boundary vertex.
        Eigen::RowVector3d barycoords;
        igl::point_simplex_squared_distance<3>(query, m_tgt_surf_V, m_tgt_surf_F, closest_idx, sqdist, p, barycoords);

        std::array<int, 3> boundaryNonzeroLoc;
        int numNonzero = 0, numBoundaryNonzero = 0;
        for (int i = 0; i < 3; ++i) {
            if (barycoords[i] == 0.0) continue;
            ++numNonzero;
            // It is extremely unlikely a vertex will be closest to a point/edge if this is not a stable association.
            // Therefore we assume even for smoothish surfaces that points are constrained to lie on their closest
            // simplex.
            // Hack away the old boundry-snapping-only behavior: treat all non-boundary edges/vertices as active too...
			// TODO: decide on this!
            // if (target_surface->vertex(m_tgt_surf_F(closest_idx, i)).isBoundary())
                boundaryNonzeroLoc[numBoundaryNonzero++] = i;
        }
        assert(numNonzero >= 1);

        if ((numNonzero == 3) || (numNonzero != numBoundaryNonzero)) {
            // If the closest point lies in the interior, the sensitivity is (I - n n^T) (the query point perturbation is projected onto the tangent plane).
            joint_closest_surf_pt_sensitivities[ji] = Eigen::Matrix3d::Identity() - m_tgt_surf_N.row(closest_idx).transpose() * m_tgt_surf_N.row(closest_idx);
            ++numInterior;
        }
        else if ((numNonzero == 2) && (numBoundaryNonzero == 2)) {
            // If the closest point lies on a boundary edge, we assume it can only slide along this edge (i.e., the constraint is active)
            // (The edge orientation doesn't matter.)
            Eigen::RowVector3d e = m_tgt_surf_V.row(m_tgt_surf_F(closest_idx, boundaryNonzeroLoc[0])) -
                                   m_tgt_surf_V.row(m_tgt_surf_F(closest_idx, boundaryNonzeroLoc[1]));
            e.normalize();
            joint_closest_surf_pt_sensitivities[ji] = e.transpose() * e;
            ++numBdryEdge;
        }
        else if ((numNonzero == 1) && (numBoundaryNonzero == 1)) {
            // If the closest point coincides with a boundary vertex, we assume it is "stuck" there (i.e., the constraint is active)
            joint_closest_surf_pt_sensitivities[ji].setZero();
            ++numBdryVtx;
        }
        else {
            assert(false);
        }
    }

    // std::cout << "numInterior: " << numInterior << ", numBdryEdge: " << numBdryEdge << ", numBdryVtx: " << numBdryVtx << std::endl;
}

TargetSurfaceFitter:: TargetSurfaceFitter() = default;
TargetSurfaceFitter::~TargetSurfaceFitter() = default;
