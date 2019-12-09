#include "../RodMaterial.hh"
#include "../CrossSection.hh"
#include <iostream>

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cout << "usage: " << argv[0] << " cross_section.json" << std::endl;
        exit(-1);
    }

    auto cs = CrossSection::load(argv[1]);
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    std::tie(vertices, elements) = cs->interior(0.001);
    MeshIO::save("crosssec.msh", vertices, elements);

    RodMaterial mat;
    mat.set(*cs);

    std::cout.precision(19);
    std::cout << mat.stretchingStiffness << std::endl;
    std::cout << mat.twistingStiffness << std::endl;
    std::cout << mat.bendingStiffness.lambda_1 << '\t' << mat.bendingStiffness.lambda_2 << std::endl;

    return 0;
}
