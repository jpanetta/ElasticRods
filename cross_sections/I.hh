////////////////////////////////////////////////////////////////////////////////
// I.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      +--------------+
//   h3 |              |
//      +----+    +----+
//           |    |
//           |    |
//   h2      | w2 |
//           |    |
//           |    |
//      +-w1-+    +-w3-+
//   h1 |              |
//      +--------------+
//     \_______________/
//        w1 + w2 + w3
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  06/24/2018 15:38:16
////////////////////////////////////////////////////////////////////////////////
#ifndef I_HH
#define I_HH

namespace CrossSections {

struct I : public ::CrossSection {
    virtual size_t numParams() const override { return 6; }

    virtual BRep boundary(bool /* hiRes */) const override {
        m_validateParams();

        Real w1 = m_params[0],
             w2 = m_params[1],
             w3 = m_params[2],
             h1 = m_params[3],
             h2 = m_params[4],
             h3 = m_params[5];

        AlignedPointCollection bdryPts = 
            { {0, 0},
              {w1 + w2 + w3, 0},
              {w1 + w2 + w3, h1}, 
              {w1 + w2, h1},
              {w1 + w2, h1 + h2},
              {w1 + w2 + w3, h1 + h2},
              {w1 + w2 + w3, h1 + h2 + h3},
              {0, h1 + h2 + h3},
              {0, h1 + h2},
              {w1, h1 + h2},
              {w1, h1},
              {0, h1} };

        std::vector<Edge> bdryEdges;
        for (size_t i = 0; i < bdryPts.size(); ++i)
            bdryEdges.push_back({i, (i + 1) % bdryPts.size()});
        return std::make_pair(bdryPts, bdryEdges);
    }
};

}

#endif /* end of include guard: I_HH */
