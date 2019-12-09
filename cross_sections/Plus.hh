////////////////////////////////////////////////////////////////////////////////
// Plus.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//            h2
//   /       +---+
//   |       |   |
//   |       |   |
//   |  +----+   +----+
// b |  |             | h1
//   |  +----+   +----+
//   |       |   |
//   |       |   |
//   \       +---+
//     \_______________/
//             a
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  06/24/2018 15:29:43
////////////////////////////////////////////////////////////////////////////////
#ifndef PLUS_HH
#define PLUS_HH

namespace CrossSections {

struct Plus : public ::CrossSection {
    virtual size_t numParams() const override { return 4; }

    virtual BRep boundary(bool /* hiRes */) const override {
        m_validateParams();

        Real a = m_params[0],
             b = m_params[1],
            h1 = m_params[2],
            h2 = m_params[3];

        AlignedPointCollection bdryPts = 
           { { h2/2, -h1/2},
             {  a/2, -h1/2},
             {  a/2,  h1/2},
             { h2/2,  h1/2},
             { h2/2,   b/2},
             {-h2/2,   b/2},
             {-h2/2,  h1/2},
             {- a/2,  h1/2},
             {- a/2, -h1/2},
             {-h2/2, -h1/2},
             {-h2/2, - b/2},
             { h2/2, - b/2} };

        std::vector<Edge> bdryEdges;
        for (size_t i = 0; i < bdryPts.size(); ++i)
            bdryEdges.push_back({i, (i + 1) % bdryPts.size()});
        return std::make_pair(bdryPts, bdryEdges);
    }
};

}

#endif /* end of include guard: PLUS_HH */
