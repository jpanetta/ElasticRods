#include "infer_target_surface.hh"
#include <MeshFEM/FEMMesh.hh>
#include <igl/loop.h>

void meshio_to_igl(const std::vector<MeshIO::IOVertex > &vertices, const std::vector<MeshIO::IOElement> &elements,
                   Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
    const size_t nv = vertices.size();
    V.resize(nv, 3);
    for (size_t i = 0; i < nv; ++i)
        V.row(i) = vertices[i].point;

    size_t ne = elements.size();
    if (ne == 0) return;
    size_t es = elements[0].size();

    if (es == 4) {
        ne *= 2; // triangulate quads
    }

    F.resize(ne, 3);

    for (size_t i = 0; i < elements.size(); ++i) {
        const auto &e = elements[i];
        if (es != e.size()) throw std::runtime_error("Mixed element types not supported");
        if (es == 4) {
            F.row(2 * i + 0) << e[0], e[1], e[2];
            F.row(2 * i + 1) << e[0], e[2], e[3];
        }
        else if (es == 3) {
            F.row(i) << e[0], e[1], e[2];
        }
        else throw std::runtime_error("Unsupported mesh type");
    }
}

void igl_to_meshio(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                   std::vector<MeshIO::IOVertex > &vertices, std::vector<MeshIO::IOElement> &elements) {
    const size_t nv = V.rows();
    const size_t ne = F.rows();

    vertices.clear();
    vertices.reserve(nv);
    for (size_t i = 0; i < nv; ++i)
        vertices.emplace_back(Vector3D(V.row(i).transpose()));

    if (F.cols() != 3) throw std::runtime_error("Currently only triangle meshes are supported");
    elements.clear();
    elements.reserve(ne);
    for (size_t i = 0; i < ne; ++i)
        elements.emplace_back(F(i, 0), F(i, 1), F(i, 2));
}

void infer_target_surface(const RodLinkage &l, Eigen::MatrixXd &V, Eigen::MatrixXi &F, size_t nsubdiv, size_t numExtensionLayers) {
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    // Start with a triangulation of the joints.
    {
        std::vector<size_t> dummy;
        l.triangulation(vertices, elements, dummy);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Create triangles to fill in each concave boundary corner.
    ////////////////////////////////////////////////////////////////////////////
    auto convexifyBoundary = [&]() {
        FEMMesh<2, 1, Vector3D> mesh(elements, vertices);
        const size_t nbe = mesh.numBoundaryEdges();
        std::vector<bool> visited(nbe);

        auto edge_vector = [&](auto e) -> Vector3D {
            return vertices.at(e. tip().volumeVertex().index()).point - 
                   vertices.at(e.tail().volumeVertex().index()).point;
        };

        // Traverse each boundary loop with the surface on the right (looking down
        // boundary half-edge). Add a triangle each time we take a (significant)
        // clockwise turn around the surface normal; this indicates a concave corner.
        for (size_t bei = 0; bei < nbe; ++bei) {
            if (visited[bei]) continue;
            auto curr = mesh.boundaryEdge(bei);
            bool filledPrevious = false;
            while (!visited[curr.index()]) {
                visited[curr.index()] = true;
                auto next = curr.next();
                // If the previous edge triggered a filled corner, filling the
                // corner at this edge's tip will result in a non-manifold
                // boundary. We must skip it.
                if (filledPrevious) { filledPrevious = false; curr = next; continue; }
                Vector3D e1 = edge_vector(curr).normalized(),
                         e2 = edge_vector(next).normalized();
                Vector3D n = e1.cross(e2);
                Real sinAngle = std::copysign(n.norm(), n.dot(curr.opposite().tri()->normal()));
                Real cosAngle = e1.dot(e2);
                if (atan2(sinAngle, cosAngle) > (M_PI / 16)) {
                    elements.emplace_back(curr.tail().volumeVertex().index(),
                                          curr. tip().volumeVertex().index(),
                                          next. tip().volumeVertex().index());
                    filledPrevious = true;
                }
                curr = next;
            }
        }
    };

    // MeshIO::save("initial.msh", vertices, elements);

    convexifyBoundary();

    // MeshIO::save("convexify1.msh", vertices, elements);

    for (size_t l = 0; l < numExtensionLayers; ++l) {
        ////////////////////////////////////////////////////////////////////////////
        // Reflect each boundary triangle (to create a new layer of jagged
        // triangles lining the surface's perimeter).
        ////////////////////////////////////////////////////////////////////////////
        {
            FEMMesh<2, 1, Vector3D> mesh(elements, vertices);

            for (const auto &be : mesh.boundaryEdges()) {
                auto vhe = be.volumeHalfEdge();
                size_t v1 = vhe.tail().index();
                size_t v2 = vhe. tip().index();
                size_t v3 = vhe.next().tip().index();

                Point3D p1 = vertices[v1];
                Point3D p2 = vertices[v2];
                Point3D p3 = vertices[v3];
                Vector3D e21 = p1 - p2;

                Point3D p_reflect = p2 + ((2.0 / e21.squaredNorm()) * e21.dot(p3 - p2)) * e21 - (p3 - p2);
                elements.emplace_back(v2, v1, vertices.size());
                vertices.emplace_back(p_reflect);
            }
        }

        // MeshIO::save("reflect.msh", vertices, elements);

        // Once more create triangles to fill in the concave corners,
        // finalizing the new layer.
        convexifyBoundary();

        // MeshIO::save("convexify2.msh", vertices, elements);
    }

    // Run loop subdivision
    if (nsubdiv > 0) {
        Eigen::MatrixXd VCage;
        Eigen::MatrixXi FCage;
        meshio_to_igl(vertices, elements, VCage, FCage);
        igl::loop(VCage, FCage, V, F, nsubdiv);
    }
    else {
        meshio_to_igl(vertices, elements, V, F);
    }
}

void infer_target_surface(const RodLinkage &l, std::vector<MeshIO::IOVertex > &vertices, std::vector<MeshIO::IOElement> &elements, size_t nsubdiv, size_t numExtensionLayers) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    infer_target_surface(l, V, F, nsubdiv, numExtensionLayers);
    igl_to_meshio(V, F, vertices, elements);
}
