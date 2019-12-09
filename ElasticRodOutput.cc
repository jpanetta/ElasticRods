#include <MeshFEM/MSHFieldWriter.hh>
#include "ElasticRod.hh"
#include "AutomaticDifferentiation.hh"

using VField = VectorField<Real, 3>;
using SField = ScalarField<Real>;

template<typename Real_>
void ElasticRod_T<Real_>::writeDebugData(const std::string &path) const {
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;

    const size_t nv = numVertices(), ne = numEdges();

    for (auto &p : deformedPoints())
        vertices.emplace_back(stripAutoDiff(p));
    for (size_t i = 0; i < ne; ++i) 
        elements.emplace_back(i, i + 1);

    MSHFieldWriter writer(path, vertices, elements);

    const auto &dc = deformedConfiguration();
    VField referenceD1(ne), materialD1(ne);
    VField restD1(ne);

    for (size_t j = 0; j < ne; ++j) {
        restD1(j)      = stripAutoDiff(m_restDirectors[j].d1);
        referenceD1(j) = stripAutoDiff(dc.referenceDirectors[j].d1);
        materialD1(j)  = stripAutoDiff(dc.materialFrame[j].d1);
    }

    SField referenceTwist(nv);
    VField curvatureBinormal(nv);
    for (size_t i = 0; i < nv; ++i) {
        referenceTwist[i] = stripAutoDiff(dc.referenceTwist[i]);
        curvatureBinormal(i) = stripAutoDiff(dc.kb[i]);
    }

    SField restLen(ne), len(ne);
    for (size_t j = 0; j < ne; ++j) {
        restLen[j] = stripAutoDiff(m_restLen[j]);
        len[j] = stripAutoDiff(dc.len[j]);
    }

    writer.addField("rest len",       restLen,           DomainType::PER_ELEMENT);
    writer.addField("len",            len,               DomainType::PER_ELEMENT);
    writer.addField("rest d1",        restD1,            DomainType::PER_ELEMENT);
    writer.addField("reference d1",   referenceD1,       DomainType::PER_ELEMENT);
    writer.addField("material d1",    materialD1,        DomainType::PER_ELEMENT);
    writer.addField("referenceTwist", referenceTwist,    DomainType::PER_NODE);
    writer.addField("kb",             curvatureBinormal, DomainType::PER_NODE);

    auto gradPToVField = [nv](const Gradient &g) {
        VField result(nv);
        for (size_t i = 0; i < nv; ++i)
            result(i) = stripAutoDiff(g.gradPos(i).eval());
        return result;
    };

    writer.addField("grad stretch energy", gradPToVField(gradEnergyStretch()), DomainType::PER_NODE);
    writer.addField("grad bend energy",    gradPToVField(gradEnergyBend()),    DomainType::PER_NODE);
    writer.addField("grad twist energy",   gradPToVField(gradEnergyTwist()),   DomainType::PER_NODE);
    writer.addField("grad energy",         gradPToVField(gradEnergy()),        DomainType::PER_NODE);
}

template<typename Real_>
void ElasticRod_T<Real_>::visualizationGeometry(std::vector<MeshIO::IOVertex > &vertices,
                                                std::vector<MeshIO::IOElement> &quads,
                                                const bool averagedMaterialFrames) const {
    const size_t ne = numEdges();
    const auto &dc = deformedConfiguration();

    // Construct a generalized cylinder for each edge in the rod.
    for (size_t j = 0; j < ne; ++j) {
        const auto &crossSectionPts   = material(j).crossSectionBoundaryPts;
        const auto &crossSectionEdges = material(j).crossSectionBoundaryEdges;

        Vec3 d1_a = dc.materialFrame[j].d1,
             d2_a = dc.materialFrame[j].d2;
        Vec3 d1_b = d1_a,
             d2_b = d2_a;

        if (averagedMaterialFrames) {
            if (j >      0) { d1_a += dc.materialFrame[j - 1].d1; d2_a += dc.materialFrame[j - 1].d2; d1_a *= 0.5; d2_a *= 0.5; }
            if (j < ne - 1) { d1_b += dc.materialFrame[j + 1].d1; d2_b += dc.materialFrame[j + 1].d2; d1_b *= 0.5; d2_b *= 0.5; }
        }

        size_t offset = vertices.size();
        // First, create copies of the cross section points for both cylinder end caps
        for (size_t k = 0; k < crossSectionPts.size(); ++k) {
            // Vec3 bdryVec = dc.materialFrame[j].d1 * crossSectionPts[k][0] + dc.materialFrame[j].d2 * crossSectionPts[k][1];
            vertices.emplace_back(stripAutoDiff((dc.point(j    ) + d1_a * crossSectionPts[k][0] + d2_a * crossSectionPts[k][1]).eval()));
            vertices.emplace_back(stripAutoDiff((dc.point(j + 1) + d1_b * crossSectionPts[k][0] + d2_b * crossSectionPts[k][1]).eval()));
        }

        for (const auto &ce : crossSectionEdges) {
            // The cross-section edges are oriented ccw in the d1-d2 plane,
            // (looking along the rod's minus tangent vector).
            quads.emplace_back(offset + 2 * ce.first  + 1,
                               offset + 2 * ce.first  + 0,
                               offset + 2 * ce.second + 0,
                               offset + 2 * ce.second + 1);
        }
    }
}

// Append this rod's data to existing geometry/scalar field.
template<typename Real_>
void ElasticRod_T<Real_>::stressVisualizationGeometry(std::vector<MeshIO::IOVertex > &vertices,
                                                      std::vector<MeshIO::IOElement> &quads,
                                                      Eigen::VectorXd &sqrtBendingEnergy,
                                                      Eigen::VectorXd &stretchingStress,
                                                      Eigen::VectorXd &maxBendingStress,
                                                      Eigen::VectorXd &minBendingStress,
                                                      Eigen::VectorXd &twistingStress) const {
    size_t rod_output_offset = vertices.size();
    visualizationGeometry(vertices, quads);
    sqrtBendingEnergy.conservativeResize(vertices.size());
    stretchingStress .conservativeResize(vertices.size());
    maxBendingStress .conservativeResize(vertices.size());
    minBendingStress .conservativeResize(vertices.size());
    twistingStress   .conservativeResize(vertices.size());

    auto              bendingEnergy = energyBendPerVertex();
    auto stretchingStressCenterline = stretchingStresses();
    auto    bendingStressCenterline =    bendingStresses();
    auto   twistingStressCenterline =   twistingStresses();

    const size_t ne = numEdges();
    size_t edge_offset = 0;
    for (size_t j = 0; j < ne; ++j) {
        const size_t numCrossSectionPts = material(j).crossSectionBoundaryPts.size();
        for (size_t i = 0; i < numCrossSectionPts; ++i) {
            for (size_t adj_vtx = 0; adj_vtx < 2; ++adj_vtx) {
                int outIdx = rod_output_offset + edge_offset + 2 * i + adj_vtx;
                sqrtBendingEnergy(outIdx) = std::sqrt(stripAutoDiff(bendingEnergy[j + adj_vtx]));
                 stretchingStress(outIdx) = stretchingStressCenterline[j];
                 maxBendingStress(outIdx) =    bendingStressCenterline(j + adj_vtx, 0);
                 minBendingStress(outIdx) =    bendingStressCenterline(j + adj_vtx, 1);
                   twistingStress(outIdx) =   twistingStressCenterline[j + adj_vtx];
            }
        }
        edge_offset += 2 * numCrossSectionPts;
    }
}

template<typename Real_>
void ElasticRod_T<Real_>::saveVisualizationGeometry(const std::string &path, const bool averagedMaterialFrames) const {
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> quads;
    visualizationGeometry(vertices, quads, averagedMaterialFrames);
    MeshIO::save(path, vertices, quads, MeshIO::FMT_GUESS, MeshIO::MESH_QUAD);
}

////////////////////////////////////////////////////////////////////////////////
// Explicit instantiation for ordinary double type and autodiff type.
////////////////////////////////////////////////////////////////////////////////
template struct ElasticRod_T<double>;
template struct ElasticRod_T<ADReal>;
