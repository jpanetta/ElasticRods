#include "../RodLinkage.hh"

extern "C" void rod_linkage_grasshopper_interface(int numVertices, int numEdges,
    double *inCoords, int *inEdges, double *outCoordsClosed, double *outCoordsDeployed,
    double deploymentAngle, int openingSteps);

int main(int /* argc */, const char * /* argv */[]) {
    std::vector<MeshIO::IOVertex>  vertices, closedVertices, deployedVertices;
    std::vector<MeshIO::IOElement> elements;
    MeshIO::load("../examples/nonuniform_linkage_no_free_ends.obj", vertices, elements);

    const size_t nv = vertices.size(), ne = elements.size();
    std::vector<double>inCoords       (3 * nv),
                      outCoordsClosed (3 * nv),
                     outCoordsDeployed(3 * nv);
    for (size_t i = 0; i < nv; ++i)
        Eigen::Map<Vector3D>(inCoords.data() + 3 * i) = vertices[i].point;

    std::vector<int> inEdges(2 * elements.size());
    for (size_t i = 0; i < ne; ++i) {
        inEdges[2 * i + 0] = elements[i][0];
        inEdges[2 * i + 1] = elements[i][1];
    }

    rod_linkage_grasshopper_interface(nv, ne,
            inCoords.data(), inEdges.data(),
            outCoordsClosed.data(), outCoordsDeployed.data(),
            M_PI / 4,
            15);

    closedVertices.resize(nv), deployedVertices.resize(nv);
    for (size_t i = 0; i < nv; ++i) {
          closedVertices[i].point = Eigen::Map<Vector3D>(outCoordsClosed  .data() + 3 * i);
        deployedVertices[i].point = Eigen::Map<Vector3D>(outCoordsDeployed.data() + 3 * i);
    }

    MeshIO::save(  "closed_linkage.msh",   closedVertices, elements);
    MeshIO::save("deployed_linkage.msh", deployedVertices, elements);

    return 0;
}
