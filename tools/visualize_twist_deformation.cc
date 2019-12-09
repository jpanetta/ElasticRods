////////////////////////////////////////////////////////////////////////////////
// visualize_twist_deformation.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Given the psi field output by the twisting stiffness solver, construct a
//  visualization of the twisted, warped cross-section.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  12/30/2018 13:40:31
////////////////////////////////////////////////////////////////////////////////
#include "../RodMaterial.hh"
#include "../CrossSection.hh"
#include <MeshFEM/TriMesh.hh>
#include <MeshFEM/filters/subdivide.hh>
#include <MeshFEM/filters/extrude.hh>
#include <MeshFEM/filters/quad_subdiv_high_aspect.hh>
#include <MeshFEM/filters/quad_tri_subdiv.hh>
#include <MeshFEM/../../bin/tools/Sampler.hh>
#include <MeshFEM/MSHFieldWriter.hh>
#include <MeshFEM/MSHFieldParser.hh>

#include <iostream>

int main(int argc, const char *argv[]) {
    if (argc != 5) {
        std::cout << "usage: " << argv[0] << " psi.msh tau h out_surface.obj" << std::endl;
        exit(-1);
    }

    const std::string psi_path = argv[1];
    Real tau = std::stod(argv[2]),
           h = std::stod(argv[3]);
    const std::string out_path = argv[4];

    // Note: even though this is a 2D mesh, the displacement field
    // analyze_rod_profile outputs will have a nonzero z component. We must use
    // a 3D parser and permit a dimension mismatch for the mesh type.
    MSHFieldParser<3> fieldParser(psi_path, /* permit dim mismatch */ true);
    auto psiField = fieldParser.scalarField("psi");

    std::vector<MeshIO::IOVertex > inVertices, outVertices;
    std::vector<MeshIO::IOElement> inElements, outElements;
    inVertices = fieldParser.vertices();
    inElements = fieldParser.elements();

    // Generate a surface triangle mesh of the extrusion by height "h"
    {
        using VertexData = SubdivVertexData<3>;
        using HalfEdgeData = SubdivHalfedgeData;
        using Mesh = TriMesh<VertexData, HalfEdgeData, TMEmptyData, VertexData, TMEmptyData>;
        Mesh mesh(inElements, inVertices.size());

        // Store position on both volume and boundary vertices for ease of use.
        for (size_t vi = 0; vi < mesh.numVertices(); ++vi) {
            auto v = mesh.vertex(vi);
            v->p = inVertices[vi];
            if (v.isBoundary()) v.boundaryVertex()->p = inVertices[vi];
        }

        extrude(mesh, h, inVertices, inElements);

        // Center the mesh around the origin along the z axis.
        BBox<Point3D> bb(inVertices);
        Point3D middle = bb.center();
        for (auto &v : inVertices)
            v.point[2] -= middle[2];

        std::vector<size_t> dummy;
        while (quad_subdiv_high_aspect(inVertices, inElements,
                    outVertices, outElements,
                    dummy, /* quadAspectThreshold */ 1.45)) {
            inVertices.swap(outVertices);
            inElements.swap(outElements);
        }
        quad_tri_subdiv(inVertices, inElements, outVertices, outElements, dummy);
    }

    // Construct field sampler for the psi data.
    ElementSampler::Sampler<2> crossSectionSampler(fieldParser.vertices(), fieldParser.elements());
    crossSectionSampler.accelerate();

    MeshIO::save("debug.msh", outVertices, outElements);

    // Apply torsion deformation
    for (auto &v : outVertices) {
        auto sample = crossSectionSampler(Point2D(v[0], v[1]));
        Real z = v[2];

        Real psi = sample.baryCoords[0] * psiField(sample.nidx[0])
                 + sample.baryCoords[1] * psiField(sample.nidx[1])
                 + sample.baryCoords[2] * psiField(sample.nidx[2]);
        v[0]  = v[0] * cos(tau * z) - v[1] * sin(tau * z);
        v[1]  = v[1] * cos(tau * z) + v[0] * sin(tau * z);
        v[2] += tau * psi;
    }

    MeshIO::save(out_path, outVertices, outElements);

    return 0;
}
