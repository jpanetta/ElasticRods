////////////////////////////////////////////////////////////////////////////////
// RectangularBox.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implement a rectangular region used to select entities in 3D (e.g., to
//  stiffen regions of a linkage that fall within a box).
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  07/19/2019 10:55:47
////////////////////////////////////////////////////////////////////////////////
#ifndef RECTANGULARBOX_HH
#define RECTANGULARBOX_HH
#include <Eigen/Dense>
#include <MeshFEM/MeshIO.hh>
#include <vector>

// Node ordering from GMSH:
//        v
// 3----------2
// |\     ^   |\
// | \    |   | \
// |  \   |   |  \
// e3  7------+---6
// |   |  +-- |-- | -> u
// 0---+e1-\--1   |
//  \  |    \  \  |
//  e4 |     \  \ |
//    \|      w  \|
//     4----------5
struct RectangularBox {
    using V3d = Eigen::Vector3d;
    using Corners = Eigen::Matrix<double, 8, 3>;

    auto corner(size_t i) const { return m_c.row(i).transpose(); }
    auto corner(size_t i)       { return m_c.row(i).transpose(); }

    RectangularBox(const Corners &c) : m_c(c) {
        V3d  o = corner(0);
        V3d e1 = corner(1) - o,
            e3 = corner(3) - o,
            e4 = corner(4) - o;
        m_u = e3.cross(e4).normalized();
        m_v = e4.cross(e1).normalized();
        m_w = e1.cross(e3).normalized();

        m_ulen = e1.dot(m_u);
        m_vlen = e3.dot(m_v);
        m_wlen = e4.dot(m_w);
    }

    bool contains(const V3d &p) const {
        V3d uvw = getUVW(p);
        if ((uvw[0] < 0) || (uvw[0] > m_ulen)) return false;
        if ((uvw[1] < 0) || (uvw[1] > m_vlen)) return false;
        if ((uvw[2] < 0) || (uvw[2] > m_wlen)) return false;
        return true;
    }

    V3d getUVW(const V3d &p) const {
        V3d pvec = p - corner(0);
        return V3d(pvec.dot(m_u), pvec.dot(m_v), pvec.dot(m_w));
    }

    void visualizationGeometry(std::vector<MeshIO::IOVertex > &vertices,
                               std::vector<MeshIO::IOElement> &quads,
                               const bool /* averagedMaterialFrames */ = false) const {
        const size_t offset = vertices.size();
        for (size_t i = 0; i < 8; ++i) vertices.emplace_back(corner(i).eval());
        quads.emplace_back(offset + 0, offset + 3, offset + 2, offset + 1);
        quads.emplace_back(offset + 0, offset + 4, offset + 7, offset + 3);
        quads.emplace_back(offset + 4, offset + 5, offset + 6, offset + 7);
        quads.emplace_back(offset + 1, offset + 2, offset + 6, offset + 5);
        quads.emplace_back(offset + 3, offset + 7, offset + 6, offset + 2);
        quads.emplace_back(offset + 0, offset + 1, offset + 5, offset + 4);
    }

private:
    Corners m_c;
    V3d m_u, m_v, m_w;
    double m_ulen, m_vlen, m_wlen;
};

struct RectangularBoxCollection {
    using Corners = RectangularBox::Corners;
    using V3d     = RectangularBox::V3d;

    RectangularBoxCollection(const std::string &path) {
        const std::vector<Corners> corners;

		std::ifstream indata(path);
		std::string line;
		std::vector<double> values;
		size_t rows = 0;
        size_t cols = 0;
		while (std::getline(indata, line)) {
			std::stringstream lineStream(line);
			double tmp;
            size_t numRead = 0;
            while (lineStream >> tmp) {
                values.push_back(tmp);
                ++numRead;
            }
            if (cols == 0) cols = numRead;
            if (cols != numRead) throw std::runtime_error("Non-square input file");
			++rows;
		}

        if (cols != 24) throw std::runtime_error("Expected 24 values per box (row)");

        boxes.reserve(rows);
        for (size_t i = 0; i < rows; ++i) {
            using CornersRowMajor = Eigen::Matrix<double, 8, 3, Eigen::RowMajor>;
            boxes.emplace_back(Eigen::Map<CornersRowMajor>(&values[i * cols]));
        }
    }

    RectangularBoxCollection(const std::vector<Corners> &corners) {
        boxes.reserve(corners.size());
        for (const auto &c : corners) boxes.emplace_back(c);
    }

    bool contains(const V3d &p) const {
        for (const auto &b : boxes)
            if (b.contains(p)) return true;
        return false;
    }

    void visualizationGeometry(std::vector<MeshIO::IOVertex > &vertices,
                               std::vector<MeshIO::IOElement> &quads,
                               const bool /* averagedMaterialFrames */ = false) const {
        for (const auto &b : boxes)
            b.visualizationGeometry(vertices, quads);
    }

    std::vector<RectangularBox> boxes;
};

#endif /* end of include guard: RECTANGULARBOX_HH */
