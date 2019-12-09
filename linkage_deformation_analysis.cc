#include "linkage_deformation_analysis.hh"
#include "RodLinkage.hh"
#include <MeshFEM/MSHFieldWriter.hh>
#include <MeshFEM/FEMMesh.hh>

void linkage_deformation_analysis(const RodLinkage &rest, const RodLinkage &defo, const std::string &path) {
    std::vector<MeshIO::IOVertex > vRest, vDefo;
    std::vector<MeshIO::IOElement> tRest, tDefo;
    std::vector<size_t> originJointRest, originJointDefo;

    rest.triangulation(vRest, tRest, originJointRest);
    defo.triangulation(vDefo, tDefo, originJointDefo);

    const size_t nv = vRest.size(), nt = tRest.size();
    if ((nv != vDefo.size()) || (nt != tDefo.size())) throw std::runtime_error("Triangulation mismatch");

    VectorField<Real, 3> u(nv);
    for (size_t i = 0; i < nv; ++i)
        u(i) = vDefo[i].point - vRest[i].point;

    const size_t NONE = RodLinkage::NONE;

    ScalarField<Real> restAngles(nv), defoAngles(nv);
    for (size_t i = 0; i < nv; ++i) {
        size_t ji = originJointRest[i];
        if (ji != NONE) {
            restAngles[i] = rest.joint(ji).alpha();
            defoAngles[i] = defo.joint(ji).alpha();
        }
    }

    FEMMesh<2, 1, Vector3D> mesh(tRest, vRest), defoMesh(tDefo, vDefo);

    // For the vertices created by triangulating the quads (where angles are undefined), average the
    // adjacent angles.
    for (auto v : mesh.vertices()) {
        size_t vi = v.index();
        if (originJointRest[vi] == NONE) {
            restAngles[vi] = 0;
            defoAngles[vi] = 0;

            auto curr = v.halfEdge();
            auto end = curr;
            size_t count = 0;
            do {
                size_t neighborJoint = originJointRest.at(curr.tail().index());
                assert(neighborJoint != NONE);
                restAngles[vi] += restAngles[neighborJoint];
                defoAngles[vi] += defoAngles[neighborJoint];
                curr = curr.ccw();
                ++count;
            } while (curr != end);
            restAngles[vi] /= count;
            defoAngles[vi] /= count;
        }
    }

    // Compute the finite strain measure
    using M3D = Eigen::Matrix3d;
    SymmetricMatrixField<Real, 3> greenStrain(nt), corotatedStrain(nt), stretch(nt);
    for (const auto triRest : mesh.elements()) {
        const size_t ti = triRest.index();
        const auto triDefo = defoMesh.element(ti);
        // pt(lambda(x)) = [p0 | p1 | p2](l0, l1, l2)^T
        M3D P;
        P.col(0) = triDefo.node(0)->p;
        P.col(1) = triDefo.node(1)->p;
        P.col(2) = triDefo.node(2)->p;
        M3D F = P * triRest->gradBarycentric().transpose() // each column of gradBarycentric is the gradient of a barycentric coordinate
              + triDefo->normal() * triRest->normal().transpose();
        greenStrain(ti) = SymmetricMatrixValue<Real, 3>(F.transpose() * F - M3D::Identity());

        Eigen::JacobiSVD<Eigen::Matrix3d> svd;
        svd.compute(F, Eigen::ComputeFullU | Eigen::ComputeFullV );
        M3D R = svd.matrixU() * svd.matrixV().transpose();
        if (R.determinant() < 0) {
            M3D W = svd.matrixV();
            W.col(svd.matrixV().cols() - 1) *= -1;
            R = svd.matrixU() * W.transpose();
        }
        M3D S = R.transpose() * F;
        corotatedStrain(ti) = SymmetricMatrixValue<Real, 3>(S - M3D::Identity());
        stretch(ti) = SymmetricMatrixValue<Real, 3>(S);
    }

    ScalarField<Real> diffAngles = defoAngles - restAngles,
                      strainMaxEigenvalue(nt), strainMinEigenvalue(nt);

    for (size_t i = 0; i < nt; ++i) {
        Vector3D lambdas = corotatedStrain(i).eigenvalues();
        strainMaxEigenvalue[i] = lambdas.maxCoeff();
        strainMinEigenvalue[i] = lambdas.minCoeff();
    }

    MSHFieldWriter writer(path, mesh);
    writer.addField("u", u);
    writer.addField("restAngles", restAngles);
    writer.addField("defoAngles", defoAngles);
    writer.addField("diffAngles", diffAngles);
    writer.addField("greenStrain", greenStrain);
    writer.addField("corotatedStrain", corotatedStrain);
    writer.addField("stretch", stretch);
    writer.addField("strainMaxEig", strainMaxEigenvalue);
    writer.addField("strainMinEig", strainMinEigenvalue);
}
