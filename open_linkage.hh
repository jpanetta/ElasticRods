#ifndef OPEN_LINKAGE_HH
#define OPEN_LINKAGE_HH

#include "RodLinkage.hh"
#include "compute_equilibrium.hh"

inline void open_linkage(RodLinkage &linkage, const Real deployedActuationAngle, NewtonOptimizerOptions eopts, const size_t rigidMotionConstrainedJoint, const Real maxStepSize = 0.02) {
    const size_t jdo = linkage.dofOffsetForJoint(rigidMotionConstrainedJoint);
    std::vector<size_t> rigidMotionFixedVars = { jdo + 0, jdo + 1, jdo + 2, jdo + 3, jdo + 4, jdo + 5 };

    const Real closedActuationAngle = linkage.getAverageJointAngle();
    auto full_actuation_equilibrium = get_equilibrium_optimizer(linkage, closedActuationAngle, rigidMotionFixedVars);
    full_actuation_equilibrium->options = eopts;
    full_actuation_equilibrium->optimize();

    int openingSteps = int(ceil(std::abs(deployedActuationAngle - closedActuationAngle) / maxStepSize));

    auto setTargetAngle = [&](double alpha)  { full_actuation_equilibrium->get_problem().setLEQConstraintRHS(alpha); };

    // Incrementally open the linkage
    std::cout << "Opening the linkage" << std::endl;
    for (int i = 1; i <= openingSteps; ++i) {
        double frac = double(i) / openingSteps;
        Real alpha_bar = closedActuationAngle * (1 - frac) + deployedActuationAngle * frac;
        std::cout << "Setting angle: " << alpha_bar << std::endl;
        setTargetAngle(alpha_bar);
        full_actuation_equilibrium->optimize();
    }
    // linkage.saveVisualizationGeometry("opened_linkage.msh");
}

#endif /* end of include guard: OPEN_LINKAGE_HH */
