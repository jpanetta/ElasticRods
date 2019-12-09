////////////////////////////////////////////////////////////////////////////////
// L.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//   h2
//  +---+
//  |   |
//  |   |
// b|   +----+
//  |        | h1
//  +--------+
//       a
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  06/22/2018 11:22:25
////////////////////////////////////////////////////////////////////////////////
#ifndef L_HH
#define L_HH
#include <utility>

namespace CrossSections {

struct L : public ::CrossSection {
    virtual size_t numParams() const override { return 4; }

    virtual BRep boundary(bool /* hiRes */) const override {
        m_validateParams();

        Real a = m_params[0],
             b = m_params[1],
            h1 = m_params[2],
            h2 = m_params[3];

        AlignedPointCollection bdryPts = {{0, 0}, {a, 0}, {a, h1}, {h2, h1}, {h2, b}, {0, b}};

        std::vector<Edge> bdryEdges;
        for (size_t i = 0; i < bdryPts.size(); ++i)
            bdryEdges.push_back({i, (i + 1) % bdryPts.size()});
        return std::make_pair(bdryPts, bdryEdges);
    }
};

}

#endif /* end of include guard: L_HH */
