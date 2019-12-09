#ifndef TEMPLATEDTYPES_HH
#define TEMPLATEDTYPES_HH

#include <MeshFEM/Types.hh>

template<typename Real_> using Vec3_T = Eigen::Matrix<Real_, 3, 1>;
template<typename Real_> using  Pt3_T = Vec3_T<Real_>;
template<typename Real_> using Vec2_T = Eigen::Matrix<Real_, 2, 1>;
template<typename Real_> using VecX_T = Eigen::Matrix<Real_, Eigen::Dynamic, 1>;
template<typename Real_> using Mat3_T = Eigen::Matrix<Real_, 3, 3>;
template<typename Real_> using Mat2_T = Eigen::Matrix<Real_, 2, 2>;

#endif /* end of include guard: TEMPLATEDTYPES_HH */
