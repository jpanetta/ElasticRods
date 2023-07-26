////////////////////////////////////////////////////////////////////////////////
// TargetSurfaceFitter.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implementation of a target surface to which points are fit using the
//  distance to their closest point projections.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  01/02/2019 11:51:11
////////////////////////////////////////////////////////////////////////////////
#ifndef TARGETSURFACEFITTER_HH
#define TARGETSURFACEFITTER_HH

#include "RodLinkage.hh"
#include "compute_equilibrium.hh"

// Forward declare mesh data structure for holding the target surface (to avoid bringing in MeshFEM::TriMesh when unnecessary)
struct TargetSurfaceMesh;
// Forward declare AABB data structure
struct TargetSurfaceAABB;

struct TargetSurfaceFitter {
    TargetSurfaceFitter(); // Needed because target_surface is a smart pointer to an incomplete type.

    std::unique_ptr<TargetSurfaceMesh> target_surface;                // halfedge structure storing the target surface.
    Eigen::VectorXd joint_closest_surf_pts;                           // compressed, flattened version of p(x)  from the writeup: only include the joint position variables
    std::vector<Eigen::Matrix3d> joint_closest_surf_pt_sensitivities; // dp_dx(x) from writeup (sensitivity of closest point projection)
    std::vector<int> joint_closest_surf_tris;                         // for debugging: index of the closest triangle to each joint.
    
    void constructTargetSurface(const RodLinkage &linkage, size_t loop_subdivisions = 0);
    void setTargetSurface(const RodLinkage &linkage, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);
    void updateClosestPoints(const RodLinkage &linkage);
    void loadTargetSurface(const RodLinkage &linkage, const std::string &path);

    // Reflect the surface across the plane defined by joint "ji"'s position and normal.
    // This is useful in case the surface buckled in the opposite direction from the
    // loaded target surface.
    void reflect(const RodLinkage &linkage, size_t ji) {
        Point3D p = linkage.joint(ji).pos();
        Vector3D n = linkage.joint(ji).normal();
        for (int i = 0; i < m_tgt_surf_V.rows(); ++i) {
            Vector3D v = m_tgt_surf_V.row(i).transpose() - p;
            v -= (2 * n.dot(v)) * n;
            m_tgt_surf_V.row(i) = (p + v).transpose();
        }
        // Also reverse the orientation of each triangle (to flip the normals for proper visualization lighting)
        for (int i = 0; i < m_tgt_surf_F.rows(); ++i)
            std::swap(m_tgt_surf_F(i, 0), m_tgt_surf_F(i, 1));

        setTargetSurface(linkage, m_tgt_surf_V, m_tgt_surf_F);
    }

    const Eigen::MatrixXd &getV() const { return m_tgt_surf_V; }
    const Eigen::MatrixXi &getF() const { return m_tgt_surf_F; }
    const Eigen::MatrixXd &getN() const { return m_tgt_surf_N; }

    Real objective(const RodLinkage &linkage) const {
        Eigen::VectorXd jointPosDiff = linkage.jointPositions() - joint_pos_tgt;
        Eigen::VectorXd surfJointPosDiff = linkage.jointPositions() - joint_closest_surf_pts;
        return 0.5 * (jointPosDiff.dot(W_diag_joint_pos.cwiseProduct(jointPosDiff)) +
                      surfJointPosDiff.dot(Wsurf_diag_joint_pos.cwiseProduct(surfJointPosDiff)));
    }

    // Adjoint solve for the target fitting objective on the deployed linkage
    //      [H_3D a][w_x     ] = [b]   or    H_3D w_x = b
    //      [a^T  0][w_lambda]   [0]
    // where b = W * (x_3D - x_tgt) + Wsurf * (x_3D - p(x_3D))
    // depending on whether average angle actuation is applied.
    Eigen::VectorXd adjoint_solve(const RodLinkage &linkage, NewtonOptimizer &opt) const {
        Eigen::VectorXd b = m_apply_W    (linkage, linkage.jointPositions() - joint_pos_tgt)
                          + m_apply_Wsurf(linkage, linkage.jointPositions() - joint_closest_surf_pts);
        if (opt.get_problem().hasLEQConstraint())
            return opt.extractFullSolution(opt.kkt_solver(opt.solver(), opt.removeFixedEntries(b)));
        return opt.extractFullSolution(opt.solver().solve(opt.removeFixedEntries(b)));
    }

    // Solve for the change in adjoint state induced by a perturbation of the equilibrium state delta_x (and possibly the structure's design parameters p):
    //                                                                                                d3E_w
    //                                                                              _____________________________________________
    //                                                                             /                                             `.
    //      [H_3D a][delta w_x     ] = [W delta_x + W_surf (I - dp_dx) delta_x ] - [d3E/dx dx dx delta_x + d3E/dx dx dp delta_p] w
    //      [a^T  0][delta w_lambda]   [               0                       ]   [                     0                     ]
    //                                 \_________________________________________________________________________________________/
    //                                                                           b
    // Note that this equation is for when an average angle actuation is applied. If not, then the last row/column of the system is removed.
    Eigen::VectorXd delta_adjoint_solve(const RodLinkage &linkage, NewtonOptimizer &opt, const Eigen::VectorXd &delta_x, const Eigen::VectorXd &d3E_w) const {
        Eigen::VectorXd target_surf_term(Wsurf_diag_joint_pos.size());

        const size_t nj = linkage.numJoints();
        for (size_t ji = 0; ji < nj; ++ji) {
            Vector3D dx = delta_x.segment<3>(linkage.dofOffsetForJoint(ji));
            target_surf_term.segment<3>(3 * ji) = dx - joint_closest_surf_pt_sensitivities[ji] * dx;
        }

        auto b = (m_unpackJointPositions(linkage, W_diag_joint_pos).cwiseProduct(delta_x)
                + m_apply_Wsurf(linkage, target_surf_term)
                - d3E_w.head(delta_x.size())).eval();

        if (opt.get_problem().hasLEQConstraint())
            return opt.extractFullSolution(opt.kkt_solver(opt.solver(), opt.removeFixedEntries(b)));
        return opt.extractFullSolution(opt.solver().solve(opt.removeFixedEntries(b)));
    }

    ////////////////////////////////////////////////////////////////////////////
    // Public member variables
    ////////////////////////////////////////////////////////////////////////////
    // Fitting weights
    Eigen::VectorXd W_diag_joint_pos,       // compressed version of W from the writeup: only include weights corresponding to joint position variables.
                    Wsurf_diag_joint_pos;   // Similar to above, the weights for fitting each joint to its closest point on the surface.
                                            // WARNING: if this is changed from zero to a nonzero value, the joint_closest_surf_pts will not be updated
                                            // until the next equilibrium solve.
    Eigen::VectorXd joint_pos_tgt; // compressed, flattened version of x_tgt from the writeup: only include the joint position variables

    // Reset the target fitting weights to penalize all joints' deviations
    // equally and control the trade-off between fitting to the individual,
    // fixed joint targets and fitting to the target surface.
    // If valence2Multiplier > 1 we attempt to fit the valence 2 joints more closely
    // to their target positions than the rest of the joints.
    void setTargetJointPosVsTargetSurfaceTradeoff(const RodLinkage &linkage, Real jointPosWeight, Real valence2Multiplier = 1.0) {
        // Given the valence 2 vertices a valence2Multiplier times higher weight for fitting to their target positions.
        // (But leave the target surface fitting weights uniform).
        const size_t nj = linkage.numJoints();
        size_t numValence2 = 0;
        for (const auto &j : linkage.joints())
            numValence2 += (j.valence() == 2);

        size_t numNonValence2 = nj - numValence2;
        Real nonValence2Weight = 1.0 / (3.0 * (numValence2 * valence2Multiplier + numNonValence2));
        size_t numJointPosComponents = 3 * nj;
        W_diag_joint_pos.resize(numJointPosComponents);
        for (size_t ji = 0; ji < nj; ++ji)
            W_diag_joint_pos.segment<3>(3 * ji).setConstant(jointPosWeight * nonValence2Weight * ((linkage.joint(ji).valence() == 2) ? valence2Multiplier : 1.0));

        Wsurf_diag_joint_pos = Eigen::VectorXd::Ones(numJointPosComponents) * ((1.0 - jointPosWeight) / numJointPosComponents);
    }

    ~TargetSurfaceFitter(); // Needed because target_surface is a smart pointer to an incomplete type.

private:
    // Target surface to which the deployed joints are fit.
    std::unique_ptr<TargetSurfaceAABB> m_tgt_surf_aabb_tree;
    Eigen::MatrixXd m_tgt_surf_V, m_tgt_surf_N;
    Eigen::MatrixXi m_tgt_surf_F;

    // Apply the joint position weight matrix W to a compressed state vector that
    // contains only variables corresponding to joint positions.
    // Returns an uncompressed vector with an entry for each state variable.
    Eigen::VectorXd m_apply_W    (const RodLinkage &linkage, const Eigen::Ref<const Eigen::VectorXd> &x_joint_pos) const { return m_unpackJointPositions(linkage, W_diag_joint_pos.cwiseProduct(x_joint_pos)); }
    Eigen::VectorXd m_apply_Wsurf(const RodLinkage &linkage, const Eigen::Ref<const Eigen::VectorXd> &x_joint_pos) const { return m_unpackJointPositions(linkage, Wsurf_diag_joint_pos.cwiseProduct(x_joint_pos)); }

    // Extract a full state vector from a compressed version that only holds
    // variables corresponding to joint positions.
    Eigen::VectorXd m_unpackJointPositions(const RodLinkage &linkage, const Eigen::Ref<const Eigen::VectorXd> &x_joint_pos) const {
        Eigen::VectorXd result = Eigen::VectorXd::Zero(linkage.numDoF());

        for (size_t ji = 0; ji < linkage.numJoints(); ++ji)
            result.segment<3>(linkage.dofOffsetForJoint(ji)) = x_joint_pos.segment<3>(3 * ji);
        return result;
    }
};

#endif /* end of include guard: TARGETSURFACEFITTER_HH */
