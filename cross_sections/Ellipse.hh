////////////////////////////////////////////////////////////////////////////////
// Ellipse.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  An axis-aligned ellipse defined by parameters (width, height)
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  06/24/2018 15:35:39
////////////////////////////////////////////////////////////////////////////////
#ifndef ELLIPSE_HH
#define ELLIPSE_HH

namespace CrossSections {

struct Ellipse : public ::CrossSection {
    virtual size_t numParams() const override { return 2; }

    virtual BRep boundary(bool hiRes) const override {
        m_validateParams();

        Real w = m_params[0],
             h = m_params[1];

        const size_t nsubdiv = hiRes ? 1000 : 20;
        BRep result;
        AlignedPointCollection &bdryPts = result.first;
        std::vector<Edge>    &bdryEdges = result.second;
        bdryPts.reserve(nsubdiv);
        bdryEdges.reserve(nsubdiv);
        for (size_t k = 0; k < nsubdiv; ++k) {
            // Generate points in ccw order in the d1-d2 plane.
            Real phi = (2.0 * M_PI * k) / nsubdiv;
            bdryPts.emplace_back(w * cos(phi), h * sin(phi));
            bdryEdges.push_back({k, (k + 1) % nsubdiv});
        }

        return result;
    }
};

}

#endif /* end of include guard: ELLIPSE_HH */
