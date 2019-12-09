#include <iostream>
#include <fstream>
#include <limits>
#include "../ActuationSparsifier.hh"
#include "../RodLinkage.hh"
#include "../restlen_solve.hh"
#include "../compute_equilibrium.hh"
#include "../open_linkage.hh"
#include <igl/point_simplex_squared_distance.h>

Eigen::VectorXd getPerturbation(size_t nvars, Real epsilon = 1.0) {
    Eigen::VectorXd perturbation;
    perturbation.setRandom(nvars);
    perturbation *= epsilon;
    return perturbation;
}

int main(int argc, const char *argv[]) {
    if ((argc != 4) && (argc != 5)) {
        std::cout << "Usage: test_actuation_sparsifier flat_linkage.obj cross_section.json initial_deployment_angle [rest_lengths.txt]" << std::endl;
        exit(-1);
    }

    const std::string linkage_path(argv[1]),
                cross_section_path(argv[2]);

    Real deployedActuationAngle = std::stod(argv[3]);

    RodLinkage linkage(linkage_path);
    linkage.setMaterial(RodMaterial(*CrossSection::load(cross_section_path)));

    if (argc == 5) {
        const std::string rl_path(argv[4]);
        std::ifstream rl_file(rl_path);
        if (!rl_file.is_open()) throw std::runtime_error("Failed to open input file '" + rl_path + "'");
        Real rl = 0.0;
        std::vector<Real> rlens;
        while (rl_file >> rl)
            rlens.push_back(rl);
        if (rlens.size() != linkage.numSegments()) throw std::runtime_error("Read incorrect number of rest lengths");
        linkage.setPerSegmentRestLength(Eigen::Map<const Eigen::VectorXd>(rlens.data(), rlens.size())); // These rest lengths should actually place the flat linkage in equilibrium if the input is valid...
    }
    else {
        std::cout << "Solving for rest lengths" << std::endl;
        restlen_solve(linkage);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Computed deployed equilibrium under full actuation
    ////////////////////////////////////////////////////////////////////////////
    // Constrain global rigid motion by fixing the position, orientation of the centermost joint
    const size_t jdo = linkage.dofOffsetForJoint(linkage.centralJoint());
    std::vector<size_t> rigidMotionFixedVars = { jdo + 0, jdo + 1, jdo + 2, jdo + 3, jdo + 4, jdo + 5 };

    // Compute undeployed equilibrium (preserving initial average actuation angle).
    NewtonOptimizerOptions eopts;
    eopts.beta = 1e-8;
    eopts.gradTol = 1e-8;
    eopts.verbose = 10;
    eopts.niter = 50;

    open_linkage(linkage, deployedActuationAngle, eopts, linkage.centralJoint());

    ////////////////////////////////////////////////////////////////////////////
    // Check that deformation doesn't change when we switch to using explicit
    // torques to hold the linkage open.
    ////////////////////////////////////////////////////////////////////////////
    std::cout << "Simulating the applied forces" << std::endl;
    auto custom_force_actuation_equilibrium = get_equilibrium_optimizer(linkage, TARGET_ANGLE_NONE, rigidMotionFixedVars);
    custom_force_actuation_equilibrium->options = eopts;
    auto &externalForces = dynamic_cast<EquilibriumProblem<RodLinkage>&>(custom_force_actuation_equilibrium->get_problem()).external_forces;
    const size_t nj = linkage.numJoints();
    {
        auto elasticForces = linkage.gradient();
        externalForces.setZero(elasticForces.size());
        for (size_t ji = 0; ji < nj; ++ji) {
            const size_t alpha_idx = linkage.dofOffsetForJoint(ji) + 6;
            externalForces[alpha_idx] = elasticForces[alpha_idx];
        }
    }
    std::cout << "Running external torque simulation" << std::endl;
    custom_force_actuation_equilibrium->optimize();
    linkage.saveVisualizationGeometry("pre_sparsification.msh");

    ActuationSparsifier sparsifier(linkage, *custom_force_actuation_equilibrium,
                                   ActuationSparsifier::Regularization::LP);
    auto vars = sparsifier.initialVarsFromElasticForces(externalForces);

    std::cout << "Rel error in parametrized force vector: "
              << (sparsifier.forceVectorFromVars(vars) - externalForces).norm() / externalForces.norm()
              << std::endl;

    Real fd_eps = 1e-5;

    std::cout.precision(19);
    std::cout << "J: " << sparsifier.eval(vars) << std::endl;

    auto perturb = getPerturbation(vars.size());
    auto grad_J = sparsifier.grad(vars);
    auto delta_grad_J = sparsifier.apply_hess(vars, perturb);
    auto delta_w = sparsifier.get_delta_w();
    auto delta_x = sparsifier.get_delta_x();
    auto closest_pts_sensitivity = sparsifier.target_surface_fitter.joint_closest_surf_pt_sensitivities;

    Real J_plus                 = sparsifier.eval(vars + fd_eps * perturb);
    Eigen::VectorXd grad_J_plus = sparsifier.grad(vars + fd_eps * perturb);
    Eigen::VectorXd x_plus = sparsifier.get_x();
    Eigen::VectorXd w_plus = sparsifier.get_w();
    sparsifier.linesearch_linkage.writeLinkageDebugData("linkage_plus.msh");
    auto closest_pts_plus = sparsifier.target_surface_fitter.joint_closest_surf_pts;
    auto closest_tris_plus = sparsifier.target_surface_fitter.joint_closest_surf_tris;

    Real J_minus                 = sparsifier.eval(vars - fd_eps * perturb);
    Eigen::VectorXd grad_J_minus = sparsifier.grad(vars - fd_eps * perturb);
    Eigen::VectorXd x_minus = sparsifier.get_x();
    Eigen::VectorXd w_minus = sparsifier.get_w();
    sparsifier.linesearch_linkage.writeLinkageDebugData("linkage_minus.msh");
    auto closest_pts_minus = sparsifier.target_surface_fitter.joint_closest_surf_pts;
    auto closest_tris_minus = sparsifier.target_surface_fitter.joint_closest_surf_tris;

    Real fd_delta_J = (J_plus - J_minus) / (2 * fd_eps);

    std::cout << "fd       delta J: " << fd_delta_J << std::endl;
    std::cout << "analytic delta J: " << grad_J.dot(perturb) << std::endl;

    auto delta_closest_pts = closest_pts_plus;
    for (size_t ji = 0; ji < nj; ++ji) {
        Vector3D dx = delta_x.segment<3>(linkage.dofOffsetForJoint(ji));
        delta_closest_pts.segment<3>(3 * ji) = closest_pts_sensitivity[ji] * dx;
    }

    auto fd_report = [&](const std::string &name, auto fd_result, auto an_result) {
        std::cout << std::endl;
        std::cout << "fd       " << name << ": " << fd_result.segment(0, 5).transpose() << "..." << std::endl;
        std::cout << "analytic " << name << ": " << an_result.segment(0, 5).transpose() << "..." << std::endl;
        std::cout << "fd       " << name << " norm: " << fd_result.norm() << std::endl;
        std::cout << "analytic " << name << " norm: " << an_result.norm() << std::endl;
        std::cout << name << " rel error: " << (fd_result - an_result).norm() / an_result.norm() << std::endl;

        int idx;
        Real err = (fd_result - an_result).cwiseAbs().maxCoeff(&idx);
        std::cout << "greatest abs error " << err << " at entry " << idx << ": "
                  << fd_result[idx] << " vs " << an_result[idx] << std::endl;
        int start = std::max(idx - 5, 0);
        int end   = std::min(idx + 5, int(fd_result.size()) - 1);
        std::cout << fd_result.segment(start, end - start + 1).transpose() << std::endl;
        std::cout << an_result.segment(start, end - start + 1).transpose() << std::endl;
        return idx;
    };

    // Warning: if updateRotationParametrizations() is called, the rotation
    // components of delta x will appear incorrect. But this is just because
    // the rotation variables are being reset to zero before the finite
    // difference evaluations; the full Hessian of J wrt p will still be
    // correct with this discrepancy.
    fd_report("delta x", ((x_plus - x_minus) / (2 * fd_eps)).eval(), delta_x);
    fd_report("delta cls pt", ((closest_pts_plus - closest_pts_minus) / (2 * fd_eps)).eval(), delta_closest_pts);
    fd_report("delta w", ((w_plus - w_minus) / (2 * fd_eps)).eval(), delta_w);

    fd_report("delta grad J", ((grad_J_plus - grad_J_minus) / (2 * fd_eps)).eval(), delta_grad_J);

    // Figure out what's happening with the worst closest point sensitivity.
    {
        // const size_t ji = worst / 3;
        const size_t ji = 175;

        const size_t jdo = linkage.dofOffsetForJoint(ji);
        Vector3D dx = delta_x.segment<3>(jdo);
        Vector3D cp_plus = closest_pts_plus.segment<3>(3 * ji),
                 cp_minus = closest_pts_minus.segment<3>(3 * ji);
        Vector3D xp = x_plus.segment<3>(jdo);
        Vector3D xm = x_minus.segment<3>(jdo);

        std::cout << std::endl;
        std::cout << "Analysis for bad joint " << ji << std::endl;
        std::cout << "FD       Joint " << ji << " motion: " << (x_plus.segment<3>(jdo) - x_minus.segment<3>(jdo)).transpose() / (2 * fd_eps) << std::endl;
        std::cout << "Analytic Joint " << ji << " motion: " << dx.transpose() << std::endl;
        std::cout << "FD       closest point motion: " << (cp_plus - cp_minus).transpose() / (2 * fd_eps) << std::endl;
        std::cout << "Analytic closest point motion: " << (closest_pts_sensitivity[ji] * dx).transpose() << std::endl;
        std::cout << "+ tri: " << closest_tris_plus [ji] << ", normal: " << sparsifier.target_surface_fitter.getN().row(closest_tris_plus [ji]) << std::endl;
        std::cout << "- tri: " << closest_tris_minus[ji] << ", normal: " << sparsifier.target_surface_fitter.getN().row(closest_tris_minus[ji]) << std::endl;

        std::cout << "+ closest point: " << cp_plus.transpose()  << std::endl;
        std::cout << "- closest point: " << cp_minus.transpose() << std::endl;
        std::cout << "+ x: " << xp.transpose()  << std::endl;
        std::cout << "- x: " << xm.transpose() << std::endl;

        size_t ntri = sparsifier.target_surface_fitter.getF().rows();
        Real minDist_plus  = std::numeric_limits<Real>::max(),
             minDist_minus = std::numeric_limits<Real>::max();
        Eigen::RowVector3d query_p = xp.transpose(), query_m = xm.transpose();
        Eigen::RowVector3d true_cp_plus, true_cp_plus_barycoords,
                           true_cp_minus, true_cp_minus_barycoords;
        for (size_t ti = 0; ti < ntri; ++ti) {
            Real dist;
            Eigen::RowVector3d pt, barycoords;
            igl::point_simplex_squared_distance<3>(query_p, sparsifier.target_surface_fitter.getV(),
                                                    sparsifier.target_surface_fitter.getF(),
                                                    closest_tris_plus[ji],
                                                    dist,
                                                    pt,
                                                    barycoords);
            if (dist < minDist_plus) {
                minDist_plus = dist;
                true_cp_plus = pt;
                true_cp_plus_barycoords = barycoords;
            }
            igl::point_simplex_squared_distance<3>(query_m, sparsifier.target_surface_fitter.getV(),
                                                    sparsifier.target_surface_fitter.getF(),
                                                    closest_tris_plus[ji],
                                                    dist,
                                                    pt,
                                                    barycoords);
            if (dist < minDist_minus) {
                minDist_minus = dist;
                true_cp_minus = pt;
                true_cp_minus_barycoords = barycoords;
            }
        }
        std::cout << "+ cp true: " << true_cp_plus << std::endl;
        std::cout << "+ cp true barycoords: " << true_cp_plus_barycoords << std::endl;

        std::cout << "- cp true: " << true_cp_minus << std::endl;
        std::cout << "- cp true barycoords: " << true_cp_minus_barycoords << std::endl;

        // auto tri = sparsifier.target_surface_fitter.getF().row(closest_tris_plus[ji]);
        // Vector3D pt1 = sparsifier.target_surface_fitter.getV().row(tri[0]).transpose(),
        //          pt2 = sparsifier.target_surface_fitter.getV().row(tri[1]).transpose(),
        //          pt3 = sparsifier.target_surface_fitter.getV().row(tri[2]).transpose();
        // TriangleClosestPoint<Real> tri_cp(pt1, pt2, pt3);
        // std::cout << "my + closest point: " << tri_cp(xp).transpose() << std::endl;
        // std::cout << "my - closest point: " << tri_cp(xm).transpose() << std::endl;
    }

    auto visualizeClosestPts = [&](const Eigen::VectorXd &closest_pts, const std::string &path) {
        // Extract line mesh for each elastic rod in the linkage
        std::vector<MeshIO::IOVertex > vertices(nj);
        std::vector<MeshIO::IOElement> elements;
        for (size_t ji = 0; ji < nj; ++ji)
            vertices[ji].point = closest_pts.segment<3>(3 * ji);

        for (const auto &s : linkage.segments()) {
            size_t a = s.startJoint, b = s.endJoint;
            if ((a > vertices.size()) || (b > vertices.size())) throw std::runtime_error("free ends not supported");
            elements.emplace_back(a, b);
        }
        MeshIO::save(path, vertices, elements);
    };

    visualizeClosestPts(closest_pts_plus, "closest_pts_plus.msh");
    visualizeClosestPts(closest_pts_minus, "closest_pts_minus.msh");

    return 0;
}
