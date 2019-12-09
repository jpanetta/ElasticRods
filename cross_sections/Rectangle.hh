////////////////////////////////////////////////////////////////////////////////
// Rectangle.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  +--------+
//  |        | b
//  +--------+
//       a
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  06/24/2018 15:35:39
////////////////////////////////////////////////////////////////////////////////
#ifndef RECTANGLE_HH
#define RECTANGLE_HH

namespace CrossSections {

struct Rectangle : public ::CrossSection {
    virtual size_t numParams() const override { return 2; }

    virtual BRep boundary(bool /* hiRes */) const override {
        m_validateParams();

        Real a = m_params[0],
             b = m_params[1];

        AlignedPointCollection bdryPts = 
            { { -a / 2, -b / 2},
              {  a / 2, -b / 2},
              {  a / 2,  b / 2},
              { -a / 2,  b / 2} };

        std::vector<Edge> bdryEdges;
        for (size_t i = 0; i < bdryPts.size(); ++i)
            bdryEdges.push_back({i, (i + 1) % bdryPts.size()});
        return std::make_pair(bdryPts, bdryEdges);
    }
};

}

#endif /* end of include guard: RECTANGLE_HH */
