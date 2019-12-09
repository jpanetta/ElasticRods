////////////////////////////////////////////////////////////////////////////////
// CrossSectionMesh.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  FEMMesh wrapper used to compute integrals over the cross-section geometry.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  08/29/2018 15:35:40
////////////////////////////////////////////////////////////////////////////////
#ifndef CROSSSECTIONMESH_HH
#define CROSSSECTIONMESH_HH
#include <MeshFEM/FEMMesh.hh>

class CrossSectionMesh : public FEMMesh<2, 2, Point2D> {
public:
    using FEMMesh<2, 2, Point2D>::FEMMesh;
};

#endif /* end of include guard: CROSSSECTIONMESH_HH */
