#include <nlohmann/json.hpp>
#include <fstream>
#include <algorithm>
#include <string>
#include "CrossSection.hh"
#include <MeshFEM/Geometry.hh>
#include <MeshFEM/Triangulate.h>

#include "cross_sections/I.hh"
#include "cross_sections/L.hh"
#include "cross_sections/Plus.hh"
#include "cross_sections/Ellipse.hh"
#include "cross_sections/Rectangle.hh"

using json = nlohmann::json;

std::unique_ptr<CrossSection> CrossSection::load(const std::string &path) {
    std::ifstream inFile(path);
    if (!inFile.is_open()) throw std::runtime_error("Couldn't open " + path);

    json config;
    inFile >> config;

    return construct(config.at("type").get<std::string>(),
                     config.at("young").get<double>(),
                     config.at("poisson").get<double>(),
                     config.at("params").get<std::vector<Real>>());
}

std::unique_ptr<CrossSection> CrossSection::construct(std::string type, Real E, Real nu, const std::vector<Real> &params) {
    std::transform(type.begin(), type.end(), type.begin(), ::toupper);

    std::unique_ptr<CrossSection> c;
    if      (type == "I")         { c = std::make_unique<CrossSections::I        >(); }
    else if (type == "L")         { c = std::make_unique<CrossSections::L        >(); }
    else if (type == "+")         { c = std::make_unique<CrossSections::Plus     >(); }
    else if (type == "-")         { c = std::make_unique<CrossSections::Rectangle>(); }
    else if (type == "RECTANGLE") { c = std::make_unique<CrossSections::Rectangle>(); }
    else if (type == "ELLIPSE")   { c = std::make_unique<CrossSections::Ellipse  >(); }
    else throw std::runtime_error("Unknown cross-section " + type);

    c->E  = E;
    c->nu = nu;
    c->setParams(params);

    return c;
}

CrossSection::VRep CrossSection::interior(Real triArea) const {
    auto bdry = boundary(true);
    BBox<Point2D> bb(bdry.first);
    triArea *= bb.volume();

    VRep result;
    triangulatePSLG(bdry.first, bdry.second, std::vector<Vector2D>(),
                    result.first, result.second, triArea, "Q");

    return result;
}
