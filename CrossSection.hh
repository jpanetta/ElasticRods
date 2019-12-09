////////////////////////////////////////////////////////////////////////////////
// CrossSection.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Abstract base class representing the interface to the cross-section
//  geometry.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  06/22/2018 11:14:23
////////////////////////////////////////////////////////////////////////////////
#ifndef CROSSSECTION_HH
#define CROSSSECTION_HH
#include <vector>
#include <stdexcept>
#include <memory>
#include <MeshFEM/Types.hh>
#include <MeshFEM/MeshIO.hh>

struct CrossSection {
    using Edge = std::pair<size_t, size_t>;
    using EdgeCollection = std::vector<Edge>;
    using AlignedPointCollection = std::vector<Point2D,
                                               Eigen::aligned_allocator<Point2D>>; // Work around alignment issues.
    using BRep = std::pair<AlignedPointCollection, EdgeCollection>;
    using VRep = std::pair<std::vector<MeshIO::IOVertex>, std::vector<MeshIO::IOElement>>;

    static std::unique_ptr<CrossSection> load(const std::string &path);
    static std::unique_ptr<CrossSection> construct(std::string type, Real E, Real nu, const std::vector<Real> &params);

    ////////////////////////////////////////////////////////////////////////////
    /*! Get the cross-section's boundary.
    //  @param[in]  highRes    whether this is the low-res
    //                         visualization geometry or the high-res
    //                         geometry used for calculating moduli.
    *///////////////////////////////////////////////////////////////////////////
    virtual BRep boundary(bool highRes = false) const = 0;

    ////////////////////////////////////////////////////////////////////////////
    /*! Mesh the interior of the cross-section with triangles (to be used for
    //  calculating moduli).
    //  @param[in]  triArea     triangulation size (relative to the
    //                          cross-section's bounding box.
    *///////////////////////////////////////////////////////////////////////////
    VRep interior(Real triArea) const;

    virtual size_t numParams() const = 0;
    void setParams(const std::vector<Real> &p) {
        if (p.size() != numParams()) throw std::runtime_error("Trying to set incorrect number of parameters");
        m_params = p;
    }
    const std::vector<Real> &params() const { return m_params; }

    virtual ~CrossSection() { }

    Real E = 0, nu = 0;

protected:
    void m_validateParams() const { if (m_params.size() != numParams()) throw std::runtime_error("Incorrect number of parameters"); }

    std::vector<Real> m_params;
};

#endif /* end of include guard: CROSSSECTION_HH */
