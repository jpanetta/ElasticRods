#include <iostream>
#include <iomanip>
#include <sstream>
#include <utility>
#include <memory>
#include "../ElasticRod.hh"
#include "../PeriodicRod.hh"
#include "../RodLinkage.hh"
#include "../compute_equilibrium.hh"
#include "../LinkageOptimization.hh"
#include "../restlen_solve.hh"
#include "../knitro_solver.hh"
#include "../linkage_deformation_analysis.hh"
#include "../DeploymentPathAnalysis.hh"

#include "visualization.hh"

#include <MeshFEM/GlobalBenchmark.hh>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
namespace py = pybind11;

template<typename T>
std::string hexString(T val) {
    std::ostringstream ss;
    ss << std::hex << val;
    return ss.str();
}

template <typename T>
std::string to_string_with_precision(const T &val, const int n = 6) {
    std::ostringstream ss;
    ss << std::setprecision(n) << val;
    return ss.str();

}
// Conversion of std::tuple to and from a py::tuple, since pybind11 doesn't seem to provide this...
template<typename... Args, size_t... Idxs>
py::tuple to_pytuple_helper(const std::tuple<Args...> &args, std::index_sequence<Idxs...>) {
    return py::make_tuple(std::get<Idxs>(args)...);
}

template<typename... Args>
py::tuple to_pytuple(const std::tuple<Args...> &args) {
    return to_pytuple_helper(args, std::make_index_sequence<sizeof...(Args)>());
}

template<class OutType>
struct FromPytupleImpl;

template<typename... Args>
struct FromPytupleImpl<std::tuple<Args...>> {
    template<size_t... Idxs>
    static auto run_helper(const py::tuple &t, std::index_sequence<Idxs...>) {
        return std::make_tuple((t[Idxs].cast<Args>())...);
    }
    static auto run(const py::tuple &t) {
        if (t.size() != sizeof...(Args)) throw std::runtime_error("Mismatched tuple size for py::tuple to std::tuple conversion.");
        return run_helper(t, std::make_index_sequence<sizeof...(Args)>());
    }
};

template<class OutType>
OutType from_pytuple(const py::tuple &t) {
    return FromPytupleImpl<OutType>::run(t);
}

PYBIND11_MODULE(elastic_rods, m) {
    m.doc() = "Elastic Rods Codebase";

    py::module::import("MeshFEM");
    py::module::import("sparse_matrices");

    ////////////////////////////////////////////////////////////////////////////////
    // ElasticRods and nested classes
    ////////////////////////////////////////////////////////////////////////////////
    auto elastic_rod = py::class_<ElasticRod>(m, "ElasticRod");

    py::enum_<ElasticRod::EnergyType>(m, "EnergyType")
        .value("Full",    ElasticRod::EnergyType::Full   )
        .value("Bend",    ElasticRod::EnergyType::Bend   )
        .value("Twist",   ElasticRod::EnergyType::Twist  )
        .value("Stretch", ElasticRod::EnergyType::Stretch)
        ;

    py::enum_<ElasticRod::BendingEnergyType>(m, "BendingEnergyType")
        .value("Bergou2010", ElasticRod::BendingEnergyType::Bergou2010)
        .value("Bergou2008", ElasticRod::BendingEnergyType::Bergou2008)
        ;

    py::class_<GradientStencilMaskCustom>(m, "GradientStencilMaskCustom")
        .def(py::init<>())
        .def_readwrite("edgeStencilMask", &GradientStencilMaskCustom::edgeStencilMask)
        .def_readwrite("vtxStencilMask",  &GradientStencilMaskCustom::vtxStencilMask)
        ;

    elastic_rod.def(py::init<std::vector<Point3D>>())
        .def("__repr__", [](const ElasticRod &e) { return "Elastic rod with " + std::to_string(e.numVertices()) + " points and " + std::to_string(e.numEdges()) + " edges"; })
        .def("setDeformedConfiguration", py::overload_cast<const std::vector<Point3D> &, const std::vector<Real> &>(&ElasticRod::setDeformedConfiguration))
        .def("setDeformedConfiguration", py::overload_cast<const ElasticRod::DeformedState &>(&ElasticRod::setDeformedConfiguration))
        .def("deformedPoints", &ElasticRod::deformedPoints)
        .def("thetas",         &ElasticRod::thetas)
        .def("setMaterial",    py::overload_cast<const             RodMaterial  &>(&ElasticRod::setMaterial))
        .def("setMaterial",    py::overload_cast<const std::vector<RodMaterial> &>(&ElasticRod::setMaterial))
        .def("material", py::overload_cast<size_t>(&ElasticRod::material, py::const_))
        .def("setRestKappas", &ElasticRod::setRestKappas)
        .def("restKappas", &ElasticRod::restKappas)
        .def("restPoints", &ElasticRod::restPoints)

        // Outputs mesh with normals
        .def("visualizationGeometry", &getVisualizationGeometry<ElasticRod>, py::arg("averagedMaterialFrames") = true)
        .def("rawVisualizationGeometry", [](ElasticRod &r, const bool averagedMaterialFrames) {
                std::vector<MeshIO::IOVertex > vertices;
                std::vector<MeshIO::IOElement> quads;
                r.visualizationGeometry(vertices, quads, averagedMaterialFrames);
                const size_t nv = vertices.size(),
                             ne = quads.size();
                Eigen::MatrixX3d V(nv, 3);
                Eigen::MatrixX4i F(ne, 4);

                for (size_t i = 0; i < nv; ++i) V.row(i) = vertices[i].point;
                for (size_t i = 0; i < ne; ++i) {
                    const auto &q = quads[i];
                    if (q.size() != 4) throw std::runtime_error("Expected quads");
                    F.row(i) << q[0], q[1], q[2], q[3];
                }

                return std::make_pair(V, F);
            }, py::arg("averagedMaterialFrames") = false)
        .def("saveVisualizationGeometry", &ElasticRod::saveVisualizationGeometry, py::arg("path"), py::arg("averagedMaterialFrames") = false)
        .def("writeDebugData", &ElasticRod::writeDebugData)

        .def("deformedConfiguration", py::overload_cast<>(&ElasticRod::deformedConfiguration, py::const_), py::return_value_policy::reference)
        .def("updateSourceFrame", &ElasticRod::updateSourceFrame)

        .def("numEdges",    &ElasticRod::numEdges)
        .def("numVertices", &ElasticRod::numVertices)

        .def("numDoF",  &ElasticRod::numDoF)
        .def("getDoFs", &ElasticRod::getDoFs)
        .def("setDoFs", py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&ElasticRod::setDoFs), py::arg("values"))

        .def("posOffset",     &ElasticRod::posOffset)
        .def("thetaOffset",   &ElasticRod::thetaOffset)
        .def("restLenOffset", &ElasticRod::restLenOffset)

        .def("getExtendedDoFs", &ElasticRod::getExtendedDoFs)
        .def("setExtendedDoFs", &ElasticRod::setExtendedDoFs)
        .def("lengthVars"     , &ElasticRod::lengthVars, py::arg("variableRestLen") = false)

        .def("restLength",   &ElasticRod::restLength)
        .def("restLengths",  &ElasticRod::restLengths)
        // Determine the deformed position at curve parameter 0.5
        .def_property_readonly("midpointPosition", [](const ElasticRod &e) -> Point3D {
                size_t ne = e.numEdges();
                // Midpoint is in the middle of an edge for odd numbers of edges,
                // at a vertex for even numbers of edges.
                if (ne % 2) return 0.5 * (e.deformedPoint(ne / 2) + e.deformedPoint(ne / 2 + 1));
                else        return e.deformedPoint(ne / 2);
            })
        // Determine the deformed material frame vector d2 at curve parameter 0.5
        .def_property_readonly("midpointD2", [](const ElasticRod &e) -> Vector3D {
                size_t ne = e.numEdges();
                // Midpoint is in the middle of an edge for odd numbers of edges,
                // at a vertex for even numbers of edges.
                if (ne % 2) return e.deformedMaterialFrameD2(ne / 2);
                else        return 0.5 * (e.deformedMaterialFrameD2(ne / 2 - 1) + e.deformedMaterialFrameD2(ne / 2));
            })

        .def_property("bendingEnergyType", [](const ElasticRod &e) { return e.bendingEnergyType(); },
                                           [](ElasticRod &e, ElasticRod::BendingEnergyType type) { e.setBendingEnergyType(type); })
        .def("energyStretch", &ElasticRod::energyStretch, "Compute stretching energy")
        .def("energyBend",    &ElasticRod::energyBend   , "Compute bending    energy")
        .def("energyTwist",   &ElasticRod::energyTwist  , "Compute twisting   energy")
        .def("energy",        py::overload_cast<ElasticRod::EnergyType>(&ElasticRod::energy, py::const_), "Compute elastic energy", py::arg("energyType") = ElasticRod::EnergyType::Full)

        .def("gradEnergyStretch", &ElasticRod::gradEnergyStretch<GradientStencilMaskCustom>, "Compute stretching energy gradient"                                                                                        , py::arg("variableRestLen") = false, py::arg("restlenOnly") = false, py::arg("stencilMask") = GradientStencilMaskCustom())
        .def("gradEnergyBend",    &ElasticRod::gradEnergyBend   <GradientStencilMaskCustom>, "Compute bending    energy gradient", py::arg("updatedSource") = false                                                      , py::arg("variableRestLen") = false, py::arg("restlenOnly") = false, py::arg("stencilMask") = GradientStencilMaskCustom())
        .def("gradEnergyTwist",   &ElasticRod::gradEnergyTwist  <GradientStencilMaskCustom>, "Compute twisting   energy gradient", py::arg("updatedSource") = false                                                      , py::arg("variableRestLen") = false, py::arg("restlenOnly") = false, py::arg("stencilMask") = GradientStencilMaskCustom())
        .def("gradient",          &ElasticRod::gradient         <GradientStencilMaskCustom>, "Compute elastic    energy gradient", py::arg("updatedSource") = false, py::arg("energyType") = ElasticRod::EnergyType::Full, py::arg("variableRestLen") = false, py::arg("restlenOnly") = false, py::arg("stencilMask") = GradientStencilMaskCustom())

        .def("hessianNNZ",             &ElasticRod::hessianNNZ,             "Tight upper bound for nonzeros in the Hessian.", py::arg("variableRestLen") = false)
        .def("hessianSparsityPattern", &ElasticRod::hessianSparsityPattern, "Compressed column matrix containing all potential nonzero Hessian entries", py::arg("variableRestLen") = false, py::arg("val") = 0.0)

        .def("hessian",           [](const ElasticRod &e, ElasticRod::EnergyType eType, bool variableRestLen) { return e.hessian(eType, variableRestLen); }, "Compute elastic energy Hessian", py::arg("energyType") = ElasticRod::EnergyType::Full, py::arg("variableRestLen") = false)
        .def("massMatrix",        py::overload_cast<>(&ElasticRod::massMatrix, py::const_))
        .def("lumpedMassMatrix",  &ElasticRod::lumpedMassMatrix)

        .def("characteristicLength", &ElasticRod::characteristicLength)
        .def("approxLinfVelocity",   &ElasticRod::approxLinfVelocity)

        .def("gravityForce",         &ElasticRod::gravityForce, py::arg("rho"), py::arg("g") = Vector3D(0, 0, 9.80635))

        .def("bendingStiffnesses",  py::overload_cast<>(&ElasticRod::bendingStiffnesses,  py::const_), py::return_value_policy::reference)
        .def("twistingStiffnesses", py::overload_cast<>(&ElasticRod::twistingStiffnesses, py::const_), py::return_value_policy::reference)

        .def("stretchingStresses",  &ElasticRod::stretchingStresses)
        .def("bendingStresses",     &ElasticRod::   bendingStresses)
        .def("minBendingStresses",  &ElasticRod::minBendingStresses)
        .def("maxBendingStresses",  &ElasticRod::maxBendingStresses)
        .def("twistingStresses",    &ElasticRod::  twistingStresses)

        .def("visualizationField", [](const ElasticRod &r, const Eigen::VectorXd  &f) { return getVisualizationField(r, f); }, "Convert a per-vertex or per-edge field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
        .def("visualizationField", [](const ElasticRod &r, const Eigen::MatrixX3d &f) { return getVisualizationField(r, f); }, "Convert a per-vertex or per-edge field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))

        .def(py::pickle([](const ElasticRod &r) { return py::make_tuple(r.restPoints(), r.restDirectors(), r.restKappas(), r.restTwists(), r.restLengths(),
                                                         r.edgeMaterials(),
                                                         r.bendingStiffnesses(),
                                                         r.twistingStiffnesses(),
                                                         r.stretchingStiffnesses(),
                                                         r.bendingEnergyType(),
                                                         r.deformedConfiguration(),
                                                         r.densities(),
                                                         r.initialMinRestLength()); },
                        [](const py::tuple &t) {
                        if ((t.size() < 11) || (t.size() > 13)) throw std::runtime_error("Invalid state!");
                            ElasticRod r              (t[ 0].cast<std::vector<Point3D>              >());
                            r.setRestDirectors        (t[ 1].cast<std::vector<ElasticRod::Directors>>());
                            r.setRestKappas           (t[ 2].cast<ElasticRod::StdVectorVector2D     >());
                            r.setRestTwists           (t[ 3].cast<std::vector<Real>                 >());
                            r.setRestLengths          (t[ 4].cast<std::vector<Real>                 >());

                            // Support old pickling format where only a RodMaterial was written instead of a vector of rod materials.
                            try         { r.setMaterial(t[ 5].cast<std::vector<RodMaterial>>()); }
                            catch (...) { r.setMaterial(t[ 5].cast<            RodMaterial >()); }

                            r.setBendingStiffnesses   (t[ 6].cast<std::vector<RodMaterial::BendingStiffness>>());
                            r.setTwistingStiffnesses  (t[ 7].cast<std::vector<Real>                         >());
                            r.setStretchingStiffnesses(t[ 8].cast<std::vector<Real>                         >());
                            r.setBendingEnergyType    (t[ 9].cast<ElasticRod::BendingEnergyType             >());
                            r.setDeformedConfiguration(t[10].cast<ElasticRod::DeformedState                 >());

                            // Support old pickling format where densities were absent.
                            if (t.size() > 11)
                                r.setDensities(t[11].cast<std::vector<Real>>());

                            // Support old pickling format where densities were absent.
                            if (t.size() > 12)
                                r.setInitialMinRestLen(t[12].cast<Real>());

                            return r;
                        }))
        ;

    // Note: the following bindings do not get used because PyBind thinks ElasticRod::Gradient is
    // just an Eigen::VectorXd. Also, they produce errors on Intel compilers.
    // py::class_<ElasticRod::Gradient>(elastic_rod, "Gradient")
    //     .def("__repr__", [](const ElasticRod::Gradient &g) { return "Elastic rod gradient with l2 norm " + to_string_with_precision(g.norm()); })
    //     .def_property_readonly("values", [](const ElasticRod::Gradient &g) { return Eigen::VectorXd(g); })
    //     .def("gradPos",   [](const ElasticRod::Gradient &g, size_t i) { return g.gradPos(i); })
    //     .def("gradTheta", [](const ElasticRod::Gradient &g, size_t j) { return g.gradTheta(j); })
    //     ;

    py::class_<ElasticRod::DeformedState>(elastic_rod, "DeformedState")
        .def("__repr__", [](const ElasticRod::DeformedState &) { return "Deformed state of an elastic rod (ElasticRod::DeformedState)."; })
        .def_readwrite("referenceDirectors", &ElasticRod::DeformedState::referenceDirectors)
        .def_readwrite("referenceTwist",     &ElasticRod::DeformedState::referenceTwist)
        .def_readwrite("tangent",            &ElasticRod::DeformedState::tangent)
        .def_readwrite("materialFrame",      &ElasticRod::DeformedState::materialFrame)
        .def_readwrite("kb",                 &ElasticRod::DeformedState::kb)
        .def_readwrite("kappa",              &ElasticRod::DeformedState::kappa)
        .def_readwrite("len",                &ElasticRod::DeformedState::len)

        .def_readwrite("sourceTangent"           , &ElasticRod::DeformedState::sourceTangent)
        .def_readwrite("sourceReferenceDirectors", &ElasticRod::DeformedState::sourceReferenceDirectors)
        .def_readwrite("sourceMaterialFrame"     , &ElasticRod::DeformedState::sourceMaterialFrame)
        .def_readwrite("sourceReferenceTwist"    , &ElasticRod::DeformedState::sourceReferenceTwist)

        .def("updateSourceFrame", &ElasticRod::DeformedState::updateSourceFrame)

        .def("setReferenceTwist", &ElasticRod::DeformedState::setReferenceTwist)

        .def(py::pickle([](const ElasticRod::DeformedState &dc) { return py::make_tuple(dc.points(), dc.thetas(), dc.sourceTangent, dc.sourceReferenceDirectors, dc.sourceTheta, dc.sourceReferenceTwist); },
                        [](const py::tuple &t) {
                        // sourceReferenceTwist is optional for backwards compatibility
                        if (t.size() != 5 && t.size() != 6) throw std::runtime_error("Invalid state!");
                            ElasticRod::DeformedState dc;
                            const auto &pts             = t[0].cast<std::vector<Point3D              >>();
                            const auto &thetas          = t[1].cast<std::vector<Real                 >>();
                            dc.sourceTangent            = t[2].cast<std::vector<Vector3D             >>();
                            dc.sourceReferenceDirectors = t[3].cast<std::vector<ElasticRod::Directors>>();
                            dc.sourceTheta              = t[4].cast<std::vector<Real                 >>();
                            if (t.size() > 5)
                                dc.sourceReferenceTwist = t[5].cast<std::vector<Real                 >>();
                            else dc.sourceReferenceTwist.assign(thetas.size(), 0);

                            dc.update(pts, thetas);
                            return dc;
                        }))
        ;

    py::class_<ElasticRod::Directors>(elastic_rod, "Directors")
        .def("__repr__", [](const ElasticRod::Directors &dirs) { return "{ d1: [" + to_string_with_precision(dirs.d1.transpose()) + "], d2: [" + to_string_with_precision(dirs.d2.transpose()) + "] }"; })
        .def_readwrite("d1", &ElasticRod::Directors::d1)
        .def_readwrite("d2", &ElasticRod::Directors::d2)
        .def(py::pickle([](const ElasticRod::Directors &d) { return py::make_tuple(d.d1, d.d2); },
                        [](const py::tuple &t) {
                        if (t.size() != 2) throw std::runtime_error("Invalid state!");
                        return ElasticRod::Directors(
                                t[0].cast<Vector3D>(),
                                t[1].cast<Vector3D>());
                        }))
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // RodMaterial
    ////////////////////////////////////////////////////////////////////////////////
    py::enum_<RodMaterial::StiffAxis>(m, "StiffAxis")
        .value("D1", RodMaterial::StiffAxis::D1)
        .value("D2", RodMaterial::StiffAxis::D2)
        ;
    py::class_<RodMaterial>(m, "RodMaterial")
        .def(py::init<const std::string &, RodMaterial::StiffAxis, bool>(),
                py::arg("cross_section_path.json"), py::arg("stiffAxis") = RodMaterial::StiffAxis::D1, py::arg("keepCrossSectionMesh") = false)
        .def(py::init<const std::string &, Real, Real, const std::vector<Real> &, RodMaterial::StiffAxis>(),
                py::arg("type"), py::arg("E"), py::arg("nu"),py::arg("params"), py::arg("stiffAxis") = RodMaterial::StiffAxis::D1)
        .def(py::init<>())
        .def("set", py::overload_cast<const std::string &, Real, Real, const std::vector<Real> &, RodMaterial::StiffAxis, bool>(&RodMaterial::set),
                py::arg("type"), py::arg("E"), py::arg("nu"),py::arg("params"), py::arg("stiffAxis") = RodMaterial::StiffAxis::D1, py::arg("keepCrossSectionMesh") = false)
        .def("setEllipse", &RodMaterial::setEllipse, "Set elliptical cross section")
        .def("setContour", &RodMaterial::setContour, "Set using a custom profile whose boundary is read from a line mesh file",
                py::arg("E"), py::arg("nu"), py::arg("path"), py::arg("scale") = 1.0, py::arg("stiffAxis") = RodMaterial::StiffAxis::D1, py::arg("keepCrossSectionMesh") = false, py::arg("debug_psi_path") = std::string(), py::arg("triArea") = 0.001, py::arg("simplifyVisualizationMesh") = 0)
        .def_readwrite("stretchingStiffness",       &RodMaterial::stretchingStiffness)
        .def_readwrite("twistingStiffness",         &RodMaterial::twistingStiffness)
        .def_readwrite("torsionStressCoefficient",  &RodMaterial::torsionStressCoefficient)
        .def_readwrite("bendingStiffness",          &RodMaterial::bendingStiffness)
        .def_readwrite("momentOfInertia",           &RodMaterial::momentOfInertia)
        .def_readwrite("crossSectionBoundaryPts",   &RodMaterial::crossSectionBoundaryPts)
        .def_readwrite("crossSectionBoundaryEdges", &RodMaterial::crossSectionBoundaryEdges)
        .def_readwrite("area",                      &RodMaterial::area)
        .def("bendingStresses", &RodMaterial::bendingStresses, py::arg("curvatureNormal"))
        .def(py::pickle([](const RodMaterial &mat) {
                    return py::make_tuple(mat.area, mat.stretchingStiffness, mat.twistingStiffness,
                                          mat.bendingStiffness, mat.momentOfInertia,
                                          mat.crossSectionBoundaryPts,
                                          mat.crossSectionBoundaryEdges);
                },
                [](const py::tuple &t) {
                    if (t.size() != 7) throw std::runtime_error("Invalid state!");
                    RodMaterial mat;
                    mat.area                      = t[0].cast<Real>();
                    mat.stretchingStiffness       = t[1].cast<Real>();
                    mat.twistingStiffness         = t[2].cast<Real>();
                    mat.bendingStiffness          = t[3].cast<RodMaterial::DiagonalizedTensor>();
                    mat.momentOfInertia           = t[4].cast<RodMaterial::DiagonalizedTensor>();
                    mat.crossSectionBoundaryPts   = t[5].cast<RodMaterial::StdVectorPoint2D>();
                    mat.crossSectionBoundaryEdges = t[6].cast<std::vector<std::pair<size_t, size_t>>>();

                    return mat;
                }))
        ;
    py::class_<RodMaterial::DiagonalizedTensor>(m, "DiagonalizedTensor")
        .def_readwrite("lambda_1", &RodMaterial::DiagonalizedTensor::lambda_1)
        .def_readwrite("lambda_2", &RodMaterial::DiagonalizedTensor::lambda_2)
        .def("trace", &RodMaterial::DiagonalizedTensor::trace)
        .def(py::pickle([](const RodMaterial::DiagonalizedTensor &d) { return py::make_tuple(d.lambda_1, d.lambda_2); },
                        [](const py::tuple &t) {
                            if (t.size() != 2) throw std::runtime_error("Invalid state!");
                            RodMaterial::DiagonalizedTensor result;
                            result.lambda_1 = t[0].cast<Real>();
                            result.lambda_2 = t[1].cast<Real>();
                            return result;
                        }))
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // RectangularBoxCollection
    ////////////////////////////////////////////////////////////////////////////////
    auto rectangular_box_collection = py::class_<RectangularBoxCollection>(m, "RectangularBoxCollection")
        .def(py::init<std::vector<RectangularBoxCollection::Corners>>(), py::arg("box_corners"))
        .def(py::init<const std::string>(), py::arg("path"))
        .def("contains", &RectangularBoxCollection::contains, py::arg("p"))
        .def("visualizationGeometry", &getVisualizationGeometry<RectangularBoxCollection>)
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // PeriodicRod
    ////////////////////////////////////////////////////////////////////////////////
    py::enum_<PeriodicRod::CurvatureDiscretizationType>(m, "CurvatureDiscretizationType")
        .value("Tangent",           PeriodicRod::CurvatureDiscretizationType::Tangent)
        .value("Sine",              PeriodicRod::CurvatureDiscretizationType::Sine)
        .value("Angle",             PeriodicRod::CurvatureDiscretizationType::Angle)
        ;

    auto periodic_rod = py::class_<PeriodicRod, std::shared_ptr<PeriodicRod>>(m, "PeriodicRod")
        .def(py::init<std::vector<Point3D>, bool>(), py::arg("pts"), py::arg("zeroRestCurvature") = false)
        .def(py::init<ElasticRod, Real>(), py::arg("rod"), py::arg("twist"))

        // Output mesh
        .def("visualizationGeometry", &getVisualizationGeometry<PeriodicRod>, py::arg("averagedMaterialFrames") = true)

        .def("setMaterial",              &PeriodicRod::setMaterial, py::arg("material"))
        .def("numDoF",                   &PeriodicRod::numDoF)
        .def("numEdges",                 &PeriodicRod::numEdges,    py::arg("countGhost") = false)
        .def("numVertices",              &PeriodicRod::numVertices, py::arg("countGhost") = false)
        .def("restLengths",              &PeriodicRod::restLengths)
        .def("restLength",               &PeriodicRod::restLength)
        .def("thetas",                   &PeriodicRod::thetas)
        .def("totalTwistAngle",          &PeriodicRod::totalTwistAngle)
        .def("totalReferenceTwistAngle", &PeriodicRod::totalReferenceTwistAngle)
        .def("openingAngle",             &PeriodicRod::openingAngle)
        .def("writhe",                   &PeriodicRod::writhe)
        .def("binormals",                &PeriodicRod::binormals, py::arg("normalize") = true, py::arg("transport_on_straight") = false)
        .def("deformedLengths",          &PeriodicRod::deformedLengths)
        .def("deformedPoints",           &PeriodicRod::deformedPoints)
        .def("maxBendingStresses",       &PeriodicRod::maxBendingStresses)
        .def("curvature",                &PeriodicRod::curvature, py::arg("discretization") = PeriodicRod::CurvatureDiscretizationType::Sine, py::arg("pointwise") = true)
        .def("torsion",                  &PeriodicRod::torsion, py::arg("discretization") = PeriodicRod::CurvatureDiscretizationType::Sine, py::arg("pointwise") = true)
        .def("setDeformedConfiguration", py::overload_cast<const std::vector<Eigen::Vector3d> &, const std::vector<Real> &>      (&PeriodicRod::setDeformedConfiguration), py::arg("points"), py::arg("thetas"))
        .def("setDeformedConfiguration", py::overload_cast<const std::vector<Eigen::Vector3d> &, const std::vector<Real> &, Real>(&PeriodicRod::setDeformedConfiguration), py::arg("points"), py::arg("thetas"), py::arg("twist"))
        .def("setDoFs",                  &PeriodicRod::setDoFs,  py::arg("dofs"))
        .def("getDoFs",                  &PeriodicRod::getDoFs)
        .def("energy",                   &PeriodicRod::energy,   py::arg("energyType") = ElasticRod::EnergyType::Full)
        .def("energyStretch",            &PeriodicRod::energyStretch)
        .def("energyBend",               &PeriodicRod::energyBend)
        .def("energyTwist",              &PeriodicRod::energyTwist)
        .def("gradient",                 &PeriodicRod::gradient, py::arg("updatedSource") = false, py::arg("energyType") = ElasticRod::EnergyType::Full)
        .def("hessianSparsityPattern",   &PeriodicRod::hessianSparsityPattern, py::arg("val") = 0.0)
        .def("hessian",                [](const PeriodicRod &r, PeriodicRod::EnergyType etype) {
                PeriodicRod::CSCMat H;
                r.hessian(H, etype);
                ElasticRod::TMatrix Htrip = H.getTripletMatrix();
                Htrip.symmetry_mode = ElasticRod::TMatrix::SymmetryMode::UPPER_TRIANGLE;
                return Htrip; },  py::arg("energyType") = ElasticRod::EnergyType::Full)
        .def("thetaOffset",  &PeriodicRod::thetaOffset)
        .def_readonly("rod", &PeriodicRod::rod, py::return_value_policy::reference)
        .def_property("twist", &PeriodicRod::twist, &PeriodicRod::setTwist, "Twist discontinuity passing from last edge back to (overlapping) first")
        .def(py::pickle([](const PeriodicRod &pr) { return py::make_tuple(pr.rod, pr.twist()); },
                        [](const py::tuple &t) {
                            if (t.size() != 2) throw std::runtime_error("Invalid state!");
                            PeriodicRod pr(t[0].cast<ElasticRod>(), t[1].cast<Real>());
                            return pr;
                        })
            )
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // RodLinkage
    ////////////////////////////////////////////////////////////////////////////////
    auto rod_linkage = py::class_<RodLinkage>(m, "RodLinkage")
        .def(py::init<const Eigen::MatrixX3d &, const Eigen::MatrixX2i &, size_t, bool>(), py::arg("points"), py::arg("edges"), py::arg("subdivision") = 10, py::arg("initConsistentAngle") = true)
        .def(py::init<const std::string &, size_t, bool>(), py::arg("path"), py::arg("subdivision") = 10, py::arg("initConsistentAngle") = true)
        .def(py::init<const RodLinkage &>(), "Copy constructor", py::arg("rod"))

        .def("set", (void (RodLinkage::*)(const Eigen::MatrixX3d &, const Eigen::MatrixX2i &, size_t, bool))(&RodLinkage::set), py::arg("points"), py::arg("edges"), py::arg("subdivision") = 10, py::arg("initConsistentAngle") = true) // py::overload_cast fails
        .def("set", (void (RodLinkage::*)(const std::string &, size_t, bool))(&RodLinkage::set), py::arg("path"), py::arg("subdivision") = 10, py::arg("initConsistentAngle") = true) // py::overload_cast fails

        .def("setBendingEnergyType", &RodLinkage::setBendingEnergyType, py::arg("betype"), "Configure the rods' bending energy type.")

        .def("energyStretch", &RodLinkage::energyStretch, "Compute stretching energy")
        .def("energyBend",    &RodLinkage::energyBend   , "Compute bending    energy")
        .def("energyTwist",   &RodLinkage::energyTwist  , "Compute twisting   energy")
        .def("energy",        py::overload_cast<ElasticRod::EnergyType>(&RodLinkage::energy, py::const_), "Compute elastic energy", py::arg("energyType") = ElasticRod::EnergyType::Full)

        .def("updateSourceFrame", &RodLinkage::updateSourceFrame, "Use the current reference frame as the source for parallel transport")
        .def("updateRotationParametrizations", &RodLinkage::updateRotationParametrizations, "Update the joint rotation variables to represent infinitesimal rotations around the current frame")

        .def("rivetForces", &RodLinkage::rivetForces, "Compute the forces exerted by the A rods on the system variables.", py::arg("energyType") = ElasticRod::EnergyType::Full)
        .def("rivetNetForceAndTorques", &RodLinkage::rivetNetForceAndTorques, "Compute the forces/torques exerted by the A rods on the center of each joint.", py::arg("energyType") = ElasticRod::EnergyType::Full)

        .def("gradient", &RodLinkage::gradient, "Elastic energy gradient", py::arg("updatedSource") = false, py::arg("energyType") = ElasticRod::EnergyType::Full, py::arg("variableRestLen") = false, py::arg("restlenOnly") = false, py::arg("skipBRods") = false)
        .def("hessian",  py::overload_cast<ElasticRod::EnergyType, bool>(&RodLinkage::hessian, py::const_), "Elastic energy  hessian", py::arg("energyType") = ElasticRod::EnergyType::Full, py::arg("variableRestLen") = false)

        .def("massMatrix",        py::overload_cast<bool, bool>(&RodLinkage::massMatrix, py::const_), py::arg("updatedSource") = false, py::arg("useLumped") = false)

        .def("characteristicLength", &RodLinkage::characteristicLength)
        .def("approxLinfVelocity",   &RodLinkage::approxLinfVelocity)

        .def("gravityForce",         &RodLinkage::gravityForce, py::arg("rho"), py::arg("g") = Vector3D(0, 0, 9.80635))

        .def("hessianNNZ",             &RodLinkage::hessianNNZ,             "Tight upper bound for nonzeros in the Hessian.",                            py::arg("variableRestLen") = false)
        .def("hessianSparsityPattern", &RodLinkage::hessianSparsityPattern, "Compressed column matrix containing all potential nonzero Hessian entries", py::arg("variableRestLen") = false, py::arg("val") = 0.0)

        .def("segment", py::overload_cast<size_t>(&RodLinkage::segment), py::return_value_policy::reference)
        .def("joint",   py::overload_cast<size_t>(&RodLinkage::joint),   py::return_value_policy::reference)

        .def("segments", [](const RodLinkage &l) { return py::make_iterator(l.segments().cbegin(), l.segments().cend()); })
        .def("joints",   [](const RodLinkage &l) { return py::make_iterator(l.joints  ().cbegin(), l.joints  ().cend()); })

        .def("traceRods",   &RodLinkage::traceRods)
        .def("rodStresses", &RodLinkage::rodStresses)
        .def("florinVisualizationGeometry", [](const RodLinkage &l) {
                std::vector<std::vector<size_t>> polylinesA, polylinesB;
                std::vector<Eigen::Vector3d> points, normals;
                std::vector<double> stresses;
                l.florinVisualizationGeometry(polylinesA, polylinesB, points, normals, stresses);
                return py::make_tuple(polylinesA, polylinesB, points, normals, stresses);
            })

        .def("getDoFs",   &RodLinkage::getDoFs)
        .def("setDoFs", py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, bool>(&RodLinkage::setDoFs), py::arg("values"), py::arg("spatialCoherence") = false)

        .def("getExtendedDoFs", &RodLinkage::getExtendedDoFs)
        .def("setExtendedDoFs", &RodLinkage::setExtendedDoFs, py::arg("values"), py::arg("spatialCoherence") = false)

        .def("getExtendedDoFsPSRL", &RodLinkage::getExtendedDoFsPSRL)
        .def("setExtendedDoFsPSRL", &RodLinkage::setExtendedDoFsPSRL, py::arg("values"), py::arg("spatialCoherence") = false)
        .def("getPerSegmentRestLength", &RodLinkage::getPerSegmentRestLength)

        .def("setPerSegmentRestLength", &RodLinkage::setPerSegmentRestLength, py::arg("values"))

        .def("swapJointAngleDefinitions", &RodLinkage::swapJointAngleDefinitions)
        .def_property("averageJointAngle", [](const RodLinkage &l)                   { return l.getAverageJointAngle(); },
                                           [](      RodLinkage &l, const Real alpha) { l.setAverageJointAngle(alpha);   })

        .def("setMaterial",               &RodLinkage::setMaterial)
        .def("stiffenRegions",            &RodLinkage::stiffenRegions)
        .def("saveVisualizationGeometry", &RodLinkage::saveVisualizationGeometry, py::arg("path"), py::arg("averagedMaterialFrames") = false)
        .def("saveStressVisualization",   &RodLinkage::saveStressVisualization)
        .def("writeRodDebugData",         py::overload_cast<const std::string &, const size_t>(&RodLinkage::writeRodDebugData, py::const_), py::arg("path"), py::arg("singleRod") = size_t(RodLinkage::NONE))
        .def("writeLinkageDebugData",     &RodLinkage::writeLinkageDebugData)
        .def("writeTriangulation",        &RodLinkage::writeTriangulation)

        // Outputs mesh with normals
        .def("visualizationGeometry", &getVisualizationGeometry<RodLinkage>, py::arg("averagedMaterialFrames") = true)

        .def("sqrtBendingEnergies", &RodLinkage::sqrtBendingEnergies)
        .def("stretchingStresses",  &RodLinkage:: stretchingStresses)
        .def("maxBendingStresses",  &RodLinkage:: maxBendingStresses)
        .def("minBendingStresses",  &RodLinkage:: minBendingStresses)
        .def("twistingStresses",    &RodLinkage::   twistingStresses)

        .def("visualizationField", [](const RodLinkage &r, const std::vector<Eigen::VectorXd>  &f) { return getVisualizationField(r, f); }, "Convert a per-vertex or per-edge field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
        .def("visualizationField", [](const RodLinkage &r, const std::vector<Eigen::MatrixX3d> &f) { return getVisualizationField(r, f); }, "Convert a per-vertex or per-edge field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))

        .def("numDoF",                  &RodLinkage::numDoF)
        .def("numSegments",             &RodLinkage::numSegments)
        .def("numJoints",               &RodLinkage::numJoints)
        .def("jointPositionDoFIndices", &RodLinkage::jointPositionDoFIndices)
        .def("jointAngleDoFIndices",    &RodLinkage::jointAngleDoFIndices)
        .def("jointDoFIndices",         &RodLinkage::jointDoFIndices)
        .def("restLenFixedVars",        &RodLinkage::restLenFixedVars)
        .def("lengthVars",              &RodLinkage::lengthVars, py::arg("variableRestLen") = false)

        .def("restLengthLaplacianEnergy", &RodLinkage::restLengthLaplacianEnergy)
        .def("getRestLengths",            &RodLinkage::getRestLengths)
        .def("minRestLength",             &RodLinkage::minRestLength)

        .def("numExtendedDoF", &RodLinkage::numExtendedDoF)
        .def("restLenOffset",  &RodLinkage::restLenOffset)
        .def("numRestLengths", &RodLinkage::numRestLengths)

        .def("centralJoint",   &RodLinkage::centralJoint)
        .def("jointPositions", &RodLinkage::jointPositions)
        .def("deformedPoints", &RodLinkage::deformedPoints)

        .def("dofOffsetForJoint",          &RodLinkage::dofOffsetForJoint,          py::arg("index"))
        .def("dofOffsetForSegment",        &RodLinkage::dofOffsetForSegment,        py::arg("index"))
        .def("restLenDofOffsetForJoint",   &RodLinkage::restLenDofOffsetForJoint,   py::arg("index"))
        .def("restLenDofOffsetForSegment", &RodLinkage::restLenDofOffsetForSegment, py::arg("index"))

        .def("segmentRestLenToEdgeRestLenMapTranspose", &RodLinkage::segmentRestLenToEdgeRestLenMapTranspose)

        .def(py::pickle([](const RodLinkage &l) { return py::make_tuple(l.joints(), l.segments(), l.material(), l.initialMinRestLength(), l.segmentRestLenToEdgeRestLenMapTranspose(), l.getPerSegmentRestLength()); },
                        [](const py::tuple &t) {
                            if (t.size() != 6) throw std::runtime_error("Invalid RodLinkage state!");
                            return std::make_unique<RodLinkage>(t[0].cast<std::vector<RodLinkage::Joint>>(),
                                                                t[1].cast<std::vector<RodLinkage::RodSegment>>(),
                                                                t[2].cast<RodMaterial>(),
                                                                t[3].cast<Real>(),
                                                                t[4].cast<SuiteSparseMatrix>(),
                                                                t[5].cast<Eigen::VectorXd>());

                        }))
        ;

    py::class_<RodLinkage::Joint>(rod_linkage, "Joint")
        .def("valence",         &RodLinkage::Joint::valence)
        .def_property("position", [](const RodLinkage::Joint &j) { return j.pos  (); }, [](RodLinkage::Joint &j, const Vector3D &v) { j.set_pos  (v); })
        .def_property("omega",    [](const RodLinkage::Joint &j) { return j.omega(); }, [](RodLinkage::Joint &j, const Vector3D &v) { j.set_omega(v); })
        .def_property("alpha",    [](const RodLinkage::Joint &j) { return j.alpha(); }, [](RodLinkage::Joint &j,            Real a) { j.set_alpha(a); })
        .def_property("len_A",    [](const RodLinkage::Joint &j) { return j.len_A(); }, [](RodLinkage::Joint &j,            Real l) { j.set_len_A(l); })
        .def_property("len_B",    [](const RodLinkage::Joint &j) { return j.len_B(); }, [](RodLinkage::Joint &j,            Real l) { j.set_len_B(l); })
        .def_property_readonly("normal",        [](const RodLinkage::Joint &j) { return j.normal(); })
        .def_property_readonly("e_A",           [](const RodLinkage::Joint &j) { return j.e_A(); })
        .def_property_readonly("e_B",           [](const RodLinkage::Joint &j) { return j.e_B(); })
        .def_property_readonly("source_t_A",    [](const RodLinkage::Joint &j) { return j.source_t_A(); })
        .def_property_readonly("source_t_B",    [](const RodLinkage::Joint &j) { return j.source_t_B(); })
        .def_property_readonly("source_normal", [](const RodLinkage::Joint &j) { return j.source_normal(); })
        .def_property_readonly("segments_A", [](const RodLinkage::Joint &j) { return j.segmentsA(); })
        .def_property_readonly("segments_B", [](const RodLinkage::Joint &j) { return j.segmentsB(); })

        .def_property_readonly("numSegmentsA", [](const RodLinkage::Joint &j) { return j.numSegmentsA(); })
        .def_property_readonly("numSegmentsB", [](const RodLinkage::Joint &j) { return j.numSegmentsB(); })
        .def_property_readonly("isStartA", [](const RodLinkage::Joint &j) { return j.isStartA(); })
        .def_property_readonly("isStartB", [](const RodLinkage::Joint &j) { return j.isStartB(); })
        .def("terminalEdgeIdentification", &RodLinkage::Joint::terminalEdgeIdentification, py::arg("segmentIdx"))
        .def("continuationSegment", &RodLinkage::Joint::continuationSegment, py::arg("segmentIdx"))

        .def(py::pickle([](const RodLinkage::Joint &joint) { return to_pytuple(joint.getState()); },
                        [](const py::tuple &t) {
                            return from_pytuple<RodLinkage::Joint::SerializedState>(t);
                        }))
        ;

    py::class_<RodLinkage::RodSegment>(rod_linkage, "RodSegment")
        .def_readonly("rod",        &RodLinkage::RodSegment::rod, py::return_value_policy::reference)
        .def_readonly("startJoint", &RodLinkage::RodSegment::startJoint)
        .def_readonly("endJoint",   &RodLinkage::RodSegment::endJoint)
        .def(py::pickle([](const RodLinkage::RodSegment &s) { return py::make_tuple(s.startJoint, s.endJoint, s.rod); },
                        [](const py::tuple &t) {
                            if (t.size() != 3) throw std::runtime_error("Invalid RodLinkage::RodSegment state!");
                            return RodLinkage::RodSegment(t[0].cast<size_t>(), t[1].cast<size_t>(), t[2].cast<ElasticRod>());
                        }))
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // Equilibrium solver
    ////////////////////////////////////////////////////////////////////////////////
    using BC = NewtonProblem::BoundConstraint;
    py::class_<NewtonProblem::BoundConstraint>(m, "BoundConstraint")
        .def_readwrite("idx",      &BC::idx)
        .def_readwrite("val",      &BC::val)
        .def_readwrite("type",     &BC::type)
        .def("active",             &BC::active,             py::arg("vars"), py::arg("g"), py::arg("tol") = 1e-8)
        .def("feasible",           &BC::feasible,           py::arg("vars"))
        .def("apply",              &BC::apply,              py::arg("vars"))
        .def("feasibleStepLength", &BC::feasibleStepLength, py::arg("vars"), py::arg("step"))
        ;

    py::class_<NewtonProblem>(m, "NewtonProblem")
        .def("energy",                 &NewtonProblem::energy)
        .def("gradient",               &NewtonProblem::gradient, py::arg("freshIterate") = false)
        .def("hessian",                &NewtonProblem::hessian)
        .def("metric",                 &NewtonProblem::metric)
        .def("fixedVars",              &NewtonProblem::fixedVars)
        .def("addFixedVariables",      &NewtonProblem::addFixedVariables)
        .def("getVars",                &NewtonProblem::getVars)
        .def("setVars",                &NewtonProblem::setVars)
        .def("applyBoundConstraints",  &NewtonProblem::applyBoundConstraints)
        .def("activeBoundConstraints", &NewtonProblem::activeBoundConstraints)
        .def("boundConstraints",       &NewtonProblem::boundConstraints, py::return_value_policy::reference)
        .def("feasible",               &NewtonProblem::feasible)
        .def("feasibleStepLength",     py::overload_cast<const Eigen::VectorXd &>(&NewtonProblem::feasibleStepLength, py::const_))
        .def("iterationCallback",      &NewtonProblem::iterationCallback)
        ;

    py::class_<ConvergenceReport>(m, "ConvergenceReport")
        .def_readonly("success",          &ConvergenceReport::success)
        .def         ("numIters",         &ConvergenceReport::numIters)
        .def_readonly("energy",           &ConvergenceReport::energy)
        .def_readonly("gradientNorm",     &ConvergenceReport::gradientNorm)
        .def_readonly("freeGradientNorm", &ConvergenceReport::freeGradientNorm)
        .def_readonly("stepLength",       &ConvergenceReport::stepLength)
        .def_readonly("indefinite",       &ConvergenceReport::indefinite)
        .def_readonly("customData",       &ConvergenceReport::customData)
        ;

    py::class_<NewtonOptimizerOptions>(m, "NewtonOptimizerOptions")
        .def(py::init<>())
        .def_readwrite("gradTol",                       &NewtonOptimizerOptions::gradTol)
        .def_readwrite("beta",                          &NewtonOptimizerOptions::beta)
        .def_readwrite("hessianScaledBeta",             &NewtonOptimizerOptions::hessianScaledBeta)
        .def_readwrite("niter",                         &NewtonOptimizerOptions::niter)
        .def_readwrite("useIdentityMetric",             &NewtonOptimizerOptions::useIdentityMetric)
        .def_readwrite("useNegativeCurvatureDirection", &NewtonOptimizerOptions::useNegativeCurvatureDirection)
        .def_readwrite("feasibilitySolve",              &NewtonOptimizerOptions::feasibilitySolve)
        .def_readwrite("verbose",                       &NewtonOptimizerOptions::verbose)
        ;

    py::class_<WorkingSet>(m, "WorkingSet")
        .def(py::init<NewtonProblem &>())
        .def("contains", &WorkingSet::contains)
        .def("fixesVariable", &WorkingSet::fixesVariable)
        .def("size", &WorkingSet::size)
        .def("getFreeComponent", &WorkingSet::getFreeComponent)
        ;

    py::class_<NewtonOptimizer>(m, "NewtonOptimizer")
        .def("optimize", &NewtonOptimizer::optimize)
        // For debugging the Newton step. TODO: support nonempty working sets, different betas
        .def("newton_step", [](NewtonOptimizer &opt, const bool feasibility) {
                Eigen::VectorXd step;
                auto &prob = opt.get_problem();
                prob.setVars(prob.applyBoundConstraints(prob.getVars()));
                WorkingSet workingSet(prob);

                Real beta = opt.options.beta;
                const Real betaMin = std::min(beta, 1e-6); // Initial shift "tau" to use when an indefinite matrix is detected.

                opt.newton_step(step, prob.gradient(false), workingSet, beta, betaMin, feasibility);
                return step;
            }, py::arg("feasibility") = false)
        .def("get_problem", py::overload_cast<>(&NewtonOptimizer::get_problem), py::return_value_policy::reference)
        .def_readwrite("options", &NewtonOptimizer::options)
        ;

    m.attr("TARGET_ANGLE_NONE") = py::float_(TARGET_ANGLE_NONE);

    m.def("compute_equilibrium",
          [](RodLinkage &linkage, Real targetAverageAngle, const NewtonOptimizerOptions &options, const std::vector<size_t> &fixedVars) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              return compute_equilibrium(linkage, targetAverageAngle, options, fixedVars);
          },
          py::arg("linkage"),
          py::arg("targetAverageAngle") = TARGET_ANGLE_NONE,
          py::arg("options") = NewtonOptimizerOptions(),
          py::arg("fixedVars") = std::vector<size_t>()
    );
    m.def("compute_equilibrium",
          [](RodLinkage &linkage, const Eigen::VectorXd &externalForces, const NewtonOptimizerOptions &options, const std::vector<size_t> &fixedVars) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              return compute_equilibrium(linkage, externalForces, options, fixedVars);
          },
          py::arg("linkage"),
          py::arg("externalForces"),
          py::arg("options") = NewtonOptimizerOptions(),
          py::arg("fixedVars") = std::vector<size_t>()
    );
    m.def("compute_equilibrium",
          [](ElasticRod &rod, const NewtonOptimizerOptions &options, const std::vector<size_t> &fixedVars) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              return compute_equilibrium(rod, options, fixedVars);
          },
          py::arg("rod"),
          py::arg("options") = NewtonOptimizerOptions(),
          py::arg("fixedVars") = std::vector<size_t>()
    );
    m.def("compute_equilibrium",
          [](ElasticRod &rod, const Eigen::VectorXd &externalForces, const NewtonOptimizerOptions &options, const std::vector<size_t> &fixedVars) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              return compute_equilibrium(rod, externalForces, options, fixedVars);
          },
          py::arg("rod"),
          py::arg("externalForces"),
          py::arg("options") = NewtonOptimizerOptions(),
          py::arg("fixedVars") = std::vector<size_t>()
    );
    m.def("get_equilibrium_optimizer",
          [](RodLinkage &linkage, Real targetAverageAngle, const std::vector<size_t> &fixedVars) { return get_equilibrium_optimizer(linkage, targetAverageAngle, fixedVars); },
          py::arg("linkage"),
          py::arg("targetAverageAngle") = TARGET_ANGLE_NONE,
          py::arg("fixedVars") = std::vector<size_t>()
    );
    m.def("get_equilibrium_optimizer",
          [](ElasticRod &rod, const std::vector<size_t> &fixedVars) { return get_equilibrium_optimizer(rod, TARGET_ANGLE_NONE, fixedVars); },
          py::arg("rod"),
          py::arg("fixedVars") = std::vector<size_t>()
    );
    m.def("restlen_solve",
          [](RodLinkage &linkage, const NewtonOptimizerOptions &opts, const std::vector<size_t> &fixedVars) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              return restlen_solve(linkage, opts, fixedVars);
          },
          py::arg("linkage"),
          py::arg("options") = NewtonOptimizerOptions(),
          py::arg("fixedVars") = std::vector<size_t>()
    );
    m.def("restlen_solve",
          [](ElasticRod &rod, const NewtonOptimizerOptions &opts, const std::vector<size_t> &fixedVars) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              return restlen_solve(rod, opts, fixedVars);
          },
          py::arg("rod"),
          py::arg("options") = NewtonOptimizerOptions(),
          py::arg("fixedVars") = std::vector<size_t>()
    );
    m.def("restlen_problem",
          [](RodLinkage &linkage, const std::vector<size_t> &fixedVars) -> std::unique_ptr<NewtonProblem> {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              return restlen_problem(linkage, fixedVars);
          },
          py::arg("linkage"),
          py::arg("fixedVars") = std::vector<size_t>()
    );
    m.def("equilibrium_problem",
          [](RodLinkage &linkage, Real targetAverageAngle, const std::vector<size_t> &fixedVars) -> std::unique_ptr<NewtonProblem> {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              return equilibrium_problem(linkage, targetAverageAngle, fixedVars);
          },
          py::arg("linkage"),
          py::arg("targetAverageAngle") = TARGET_ANGLE_NONE,
          py::arg("fixedVars") = std::vector<size_t>()
    );
#if HAS_KNITRO
    m.def("compute_equilibrium_knitro",
          [](RodLinkage &linkage, size_t niter, int /* verbose */, const std::vector<size_t> &fixedVars, Real gradTol) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              knitro_compute_equilibrium(linkage, niter, fixedVars, gradTol);
          },
          py::arg("linkage"),
          py::arg("niter") = 100,
          py::arg("verbose") = 0,
          py::arg("fixedVars") = std::vector<size_t>(),
          py::arg("gradTol") = 2e-8
    );
    m.def("compute_equilibrium_knitro",
          [](ElasticRod &rod, size_t niter, int /* verbose */, const std::vector<size_t> &fixedVars, Real gradTol) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              knitro_compute_equilibrium(rod, niter, fixedVars, gradTol);
          },
          py::arg("rod"),
          py::arg("niter") = 100,
          py::arg("verbose") = 0,
          py::arg("fixedVars") = std::vector<size_t>(),
          py::arg("gradTol") = 2e-8
    );
    m.def("restlen_solve_knitro",
          [](RodLinkage &linkage, Real laplacianRegWeight, size_t niter, const std::vector<size_t> &fixedVars, Real gradTol) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              knitro_restlen_solve(linkage, laplacianRegWeight, niter, fixedVars, gradTol);
          },
          py::arg("linkage"),
          py::arg("laplacianRegWeight") = 1.0,
          py::arg("niter") = 100,
          py::arg("fixedVars") = std::vector<size_t>(),
          py::arg("gradTol") = 2e-8
    );
    m.def("restlen_solve_knitro",
          [](ElasticRod &rod, Real laplacianRegWeight, size_t niter, const std::vector<size_t> &fixedVars, Real gradTol) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              knitro_restlen_solve(rod, laplacianRegWeight, niter, fixedVars, gradTol);
          },
          py::arg("rod"),
          py::arg("laplacianRegWeight") = 1.0,
          py::arg("niter") = 100,
          py::arg("fixedVars") = std::vector<size_t>(),
          py::arg("gradTol") = 2e-8
    );
#endif // HAS_KNITRO

    ////////////////////////////////////////////////////////////////////////////////
    // Analysis
    ////////////////////////////////////////////////////////////////////////////////
    m.def("linkage_deformation_analysis", &linkage_deformation_analysis, py::arg("rest_linkge"), py::arg("defo_linkge"), py::arg("path"));

    py::class_<DeploymentEnergyIncrement>(m, "DeploymentEnergyIncrement")
        .def_readonly("linearTerm",    &DeploymentEnergyIncrement::linearTerm)
        .def_readonly("quadraticTerm", &DeploymentEnergyIncrement::quadraticTerm)
        .def("__call__",               &DeploymentEnergyIncrement::operator())
        ;

    py::class_<DeploymentPathAnalysis>(m, "DeploymentPathAnalysis")
        .def(py::init<NewtonOptimizer &>(), py::arg("opt"))
        .def(py::init<RodLinkage &, const std::vector<size_t> &>(), py::arg("linkage"), py::arg("fixedVars"))
        .def_readonly("deploymentStep",            &DeploymentPathAnalysis::deploymentStep)
        .def_readonly("secondBestDeploymentStep",  &DeploymentPathAnalysis::secondBestDeploymentStep)
        .def_readonly("relativeStiffnessGap",      &DeploymentPathAnalysis::relativeStiffnessGap)
        .def_readonly("bestEnergyIncrement",       &DeploymentPathAnalysis::bestEnergyIncrement)
        .def_readonly("secondBestEnergyIncrement", &DeploymentPathAnalysis::secondBestEnergyIncrement)
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // Linkage Optimization
    ////////////////////////////////////////////////////////////////////////////////
    auto linkage_optimization = py::class_<LinkageOptimization>(m, "LinkageOptimization")
        .def(py::init<RodLinkage &, RodLinkage &, const NewtonOptimizerOptions &>(), py::arg("flat_linkage"), py::arg("deployed_linkage"), py::arg("equilibrium_options") = NewtonOptimizerOptions())
        .def("newPt",          &LinkageOptimization::newPt,          py::arg("params"))
        .def("c",              py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&LinkageOptimization::c),              py::arg("params"))
        .def("J",              py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&LinkageOptimization::J),              py::arg("params"))
        .def("J_target",       py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&LinkageOptimization::J_target),       py::arg("params"))
        .def("gradp_J",        py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&LinkageOptimization::gradp_J),        py::arg("params"))
        .def("gradp_J_target", py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&LinkageOptimization::gradp_J_target), py::arg("params"))
        .def("gradp_c",        &LinkageOptimization::gradp_c,        py::arg("params"))
        .def("apply_hess_J",   &LinkageOptimization::apply_hess_J, py::arg("params"), py::arg("delta_p"))
        .def("apply_hess_c",   &LinkageOptimization::apply_hess_c, py::arg("params"), py::arg("delta_p"))
        .def("apply_hess_angle_constraint",   &LinkageOptimization::apply_hess_angle_constraint, py::arg("params"), py::arg("delta_p"))

        .def("numParams",     &LinkageOptimization::numParams)
        .def("get_w_x",       &LinkageOptimization::get_w_x)
        .def("get_y",         &LinkageOptimization::get_y)

        .def("get_s_x",       &LinkageOptimization::get_s_x)
        .def("get_delta_x3d", &LinkageOptimization::get_delta_x3d)
        .def("get_delta_x2d", &LinkageOptimization::get_delta_x2d)
        .def("get_delta_w_x", &LinkageOptimization::get_delta_w_x)
        .def("get_delta_s_x", &LinkageOptimization::get_delta_s_x)

        .def("getLinesearchFlatLinkage",     &LinkageOptimization::getLinesearchFlatLinkage,     py::return_value_policy::reference)
        .def("getLinesearchDeployedLinkage", &LinkageOptimization::getLinesearchDeployedLinkage, py::return_value_policy::reference)

        // .def_readwrite("W_diag_joint_pos",    &LinkageOptimization::W_diag_joint_pos)
        .def_readwrite("beta",                &LinkageOptimization::beta)
        .def_readwrite("gamma",               &LinkageOptimization::gamma)
        // .def_readwrite("joint_pos_tgt",       &LinkageOptimization::joint_pos_tgt)
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // Benchmarking
    ////////////////////////////////////////////////////////////////////////////////
    m.def("benchmark_reset", &BENCHMARK_RESET);
    m.def("benchmark_start_timer_section", &BENCHMARK_START_TIMER_SECTION, py::arg("name"));
    m.def("benchmark_stop_timer_section",  &BENCHMARK_STOP_TIMER_SECTION,  py::arg("name"));
    m.def("benchmark_start_timer",         &BENCHMARK_START_TIMER,         py::arg("name"));
    m.def("benchmark_stop_timer",          &BENCHMARK_STOP_TIMER,          py::arg("name"));
    m.def("benchmark_report", [](bool includeMessages) {
            py::scoped_ostream_redirect stream(std::cout, py::module::import("sys").attr("stdout"));
            if (includeMessages) BENCHMARK_REPORT(); else BENCHMARK_REPORT_NO_MESSAGES();
        },
        py::arg("include_messages") = false)
        ;

    ////////////////////////////////////////////////////////////////////////////
    // Free-standing output functions
    ////////////////////////////////////////////////////////////////////////////
    m.def("save_mesh", [](const std::string &path, Eigen::MatrixX3d &V, Eigen::MatrixXi &F) {
        std::vector<MeshIO::IOVertex > vertices;
        std::vector<MeshIO::IOElement> elements;
        const size_t nv = V.rows();
        vertices.reserve(nv);
        for (size_t i = 0; i < nv; ++i)
           vertices.emplace_back(V.row(i).transpose().eval());
        const size_t ne = F.rows();
        const size_t nc = F.cols();
        elements.reserve(ne);
        for (size_t i = 0; i < ne; ++i) {
            elements.emplace_back(nc);
            for (size_t c = 0; c < nc; ++c)
                elements.back()[c] = F(i, c);
        }

        MeshIO::MeshType type;
        if (nc == 3) { type = MeshIO::MeshType::MESH_TRI; }
        else if (nc == 4) { type = MeshIO::MeshType::MESH_QUAD; }
        else {throw std::runtime_error("unsupported element type"); }

        MeshIO::save(path, vertices, elements, MeshIO::Format::FMT_GUESS, type);
    });
}
