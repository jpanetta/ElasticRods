#ifndef INFER_TARGET_SURFACE_HH
#define INFER_TARGET_SURFACE_HH
#include <MeshFEM/MeshIO.hh>
#include "RodLinkage.hh"

void meshio_to_igl(const std::vector<MeshIO::IOVertex > &vertices, const std::vector<MeshIO::IOElement> &elements,
                   Eigen::MatrixXd &V, Eigen::MatrixXi &F);
void igl_to_meshio(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                   std::vector<MeshIO::IOVertex > &vertices, std::vector<MeshIO::IOElement> &elements);

void infer_target_surface(const RodLinkage &l, Eigen::MatrixXd &V, Eigen::MatrixXi &F, size_t nsubdiv = 0, size_t numExtensionLayers = 1);

void infer_target_surface(const RodLinkage &l, std::vector<MeshIO::IOVertex > &vertices, std::vector<MeshIO::IOElement> &elements, size_t nsubdiv = 0, size_t numExtensionLayers = 1);

#endif /* end of include guard: INFER_TARGET_SURFACE_HH */
