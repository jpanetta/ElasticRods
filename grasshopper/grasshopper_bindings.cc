#include "../RodLinkage.hh"
#include "../restlen_solve.hh"
#include "../compute_equilibrium.hh"

#ifdef _WIN32
# ifdef WIN_EXPORT
#   define EXPORTED  __declspec( dllexport )
# else
#   define EXPORTED  __declspec( dllimport )
# endif
#else
# define EXPORTED __attribute__((visibility("default")))
#endif

extern "C" EXPORTED void rod_linkage_grasshopper_interface(int numVertices, int numEdges,
    double *inCoords, int *inEdges, double *outCoordsClosed,
    double *outCoordsDeployed, double deploymentAngle, int openingSteps)
{
    std::vector<MeshIO::IOVertex> vertices(numVertices);
    for (int i = 0; i < numVertices; ++i)
        vertices[i].point = Eigen::Map<Vector3D>(inCoords + 3 * i);

    std::vector<MeshIO::IOElement> edges;
    edges.reserve(numEdges);

    for (int i = 0; i < numEdges; ++i) {
        edges.emplace_back(inEdges[2 * i    ],
                           inEdges[2 * i + 1]);
    }

    // Construct flat linkage and solve for initial restlengths
    RodLinkage   closed_linkage(vertices, edges);
    closed_linkage.setMaterial(RodMaterial("+", 2000, 0.3, {0.02, 0.02, 0.002, 0.002}));

    restlen_solve(closed_linkage);

    // To support linkages with free ends we'll need to track the mapping from
    // input vertices to joints/rod free ends.
    if (closed_linkage.numJoints() != size_t(numVertices))
        throw std::runtime_error("Free ends currently unsupported");

    // Constrain global rigid motion by fixing the position, orientation of the centermost joint
    const size_t jdo = closed_linkage.dofOffsetForJoint(closed_linkage.centralJoint());
    std::vector<size_t> rigidMotionFixedVars = { jdo + 0, jdo + 1, jdo + 2, jdo + 3, jdo + 4, jdo + 5 };

    // Newton solver options for intermediate opening iterations and final equilibrium solve
    NewtonOptimizerOptions opening_newton_opts, final_newton_opts;
    opening_newton_opts.gradTol = 1e-8;
    opening_newton_opts.beta = 1e-8;
    opening_newton_opts.verbose = false;
    opening_newton_opts.niter = 15;
    // opening_newton_opts.useIdentityMetric = true;
    final_newton_opts = opening_newton_opts;
    final_newton_opts.niter = 100;

    // Compute flat equilibrium (no actuation for now)
    auto closed_equilibrium = get_equilibrium_optimizer(  closed_linkage, TARGET_ANGLE_NONE, rigidMotionFixedVars);
    closed_equilibrium->options = final_newton_opts;
    closed_equilibrium->optimize();

    // Compute deployed equilibrium incrementally
    RodLinkage deployed_linkage(closed_linkage);
    double initAngle = closed_linkage.getAverageJointAngle();
    auto deployed_equilibrium = get_equilibrium_optimizer(deployed_linkage, initAngle, rigidMotionFixedVars);
    deployed_equilibrium->options = opening_newton_opts;

    auto setTargetAngle = [&](double alpha)  { deployed_equilibrium->get_problem().setLEQConstraintRHS(alpha); };

    // Incrementally open the linkage
    for (int i = 0; i < openingSteps; ++i) {
        double frac = i / (openingSteps - 1.0);
        setTargetAngle(initAngle * (1 - frac) + deploymentAngle * frac);
        deployed_equilibrium->optimize();
    }

    // Solve the deployed equilibrium more accurately/handle case where openingSteps <= 0
    setTargetAngle(deploymentAngle);
    deployed_equilibrium->options = final_newton_opts;
    deployed_equilibrium->optimize();

    // Output the equilibrium coordinates for the closed and deployed linkages.
    Eigen::Map<VecX_T<double>>(outCoordsClosed,   3 * numVertices) =   closed_linkage.jointPositions();
    Eigen::Map<VecX_T<double>>(outCoordsDeployed, 3 * numVertices) = deployed_linkage.jointPositions();
}
