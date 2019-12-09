////////////////////////////////////////////////////////////////////////////////
// analyze_rod_profile.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Analyze a rod cross-section whose boundary is given as a line mesh.
//  This line mesh should have a single dangling vertex inside each hole in the
//  cross-section (to mark the holes for Triangle).
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  12/30/2018 13:07:52
////////////////////////////////////////////////////////////////////////////////
#include "../RodMaterial.hh"
#include "../CrossSection.hh"
#include <iostream>

int main(int argc, const char *argv[]) {
    if ((argc < 2) || (argc > 4)) {
        std::cout << "usage: " << argv[0] << " input_line_mesh [psi.msh, triArea]" << std::endl;
        exit(-1);
    }

    std::string input_path = argv[1];
    std::string debug_psi_path;
    Real triArea = 0.001;
    if (argc > 2) debug_psi_path = argv[2];
    if (argc > 3) triArea = std::stod(argv[3]);

    RodMaterial mat;
    if (input_path.substr(input_path.size() - 4) == "json") {
        mat.set(*CrossSection::load(input_path), RodMaterial::StiffAxis::D1, false, debug_psi_path);
    }
    else {
        mat.setContour(20000, 0.3, input_path, 1.0, RodMaterial::StiffAxis::D1,
                       false, debug_psi_path, triArea);
    }

    std::cout.precision(19);
    std::cout << "stretching stiffness:\t" << mat.stretchingStiffness << std::endl;
    std::cout << "twisting   stiffness:\t" << mat.twistingStiffness << std::endl;
    std::cout << "bending    stiffness:\t" << mat.bendingStiffness.lambda_1 << '\t' << mat.bendingStiffness.lambda_2 << std::endl;

    return 0;
}
