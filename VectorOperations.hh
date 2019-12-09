#ifndef VECTOROPERATIONS_HH
#define VECTOROPERATIONS_HH

#include <MeshFEM/Types.hh>
#include "AutomaticDifferentiation.hh"

// Get an arbitrary vector in the plane perpendicular to "t"
template<typename Real_>
Vec3_T<Real_> getPerpendicularVector(const Vec3_T<Real_> &t) {
    Vec3_T<Real_> candidate1 = Vec3_T<Real_>(1, 0, 0).cross(t),
                  candidate2 = Vec3_T<Real_>(0, 1, 0).cross(t);
    return (candidate1.norm() > candidate2.norm()) ?
        candidate1.normalized() : candidate2.normalized();
}

// Compute the curvature binormal for a vertex between two edges with tangents
// e0 and e1, respectively
// (edge tangent vectors not necessarily normalized)
template<typename Real_>
Vec3_T<Real_> curvatureBinormal(const Vec3_T<Real_> &e0, const Vec3_T<Real_> &e1) {
    return e0.cross(e1) * (2.0 / (e0.norm() * e1.norm() + e0.dot(e1)));
}

// Rotate v around axis using Rodrigues' rotation formula
template<typename Real_>
Vec3_T<Real_> rotatedVector(const Vec3_T<Real_> &sinThetaAxis, Real_ cosTheta, const Vec3_T<Real_> &v) {
    Real_ sinThetaSq = sinThetaAxis.squaredNorm();
    // Robust handling of small rotations:
    // Plugging theta ~= 0 into (1 - cos(theta)) / sin(theta)^2, we would compute nearly 0/0.
    // Instead, we can use the following approximation:
    //      (1 - cos(theta)) / sin(theta)^2 = cos(theta)^2 / 2 + 5 sin(theta)^2 / 8 + sin(theta)^4 / 16 + O(theta^6)
    // For theta in [-1e-2, 1e-2], this approximation is accurate to at least
    // 13 digits--significantly more accurate than evaluating the formula in
    // double precision.
    Real_ normalization;
    if (sinThetaSq > 1e-4)
        normalization = (1 - cosTheta) / sinThetaSq;
    else { normalization = 0.5 * cosTheta * cosTheta + sinThetaSq * (5.0 / 8.0 + sinThetaSq / 16.0); }
    return sinThetaAxis * (sinThetaAxis.dot(v) * normalization) + cosTheta * v + (sinThetaAxis.cross(v));
}

template<typename Real_>
Vec3_T<Real_> rotatedVectorAngle(const Vec3_T<Real_> &axis, Real_ angle, const Vec3_T<Real_> &v) {
    return rotatedVector<Real_>((axis * sin(angle)).eval(), cos(angle), v);
}

// Get the sin of the signed angle from v1 to v2 around axis "a"; ccw angles are positive.
// Assumes all vectors are normalized.
template<typename Real_>
Real_ sinAngle(const Vec3_T<Real_> &a, const Vec3_T<Real_> &v1, const Vec3_T<Real_> &v2) {
    return v1.cross(v2).dot(a);
}

// Get the signed angle from v1 to v2 around axis "a"; ccw angles are positive.
// Assumes all vectors are normalized **and perpendicular to a**
// Return answer in the range [-pi, pi]
template<typename Real_>
Real_ angle(const Vec3_T<Real_> &a, const Vec3_T<Real_> &v1, const Vec3_T<Real_> &v2) {
    Real_ s = std::max(Real_(-1.0), std::min(Real_(1.0), sinAngle(a, v1, v2)));
    Real_ c = std::max(Real_(-1.0), std::min(Real_(1.0), v1.dot(v2)));
    return atan2(s, c);
}

// Transport vector "v" from edge with tangent vector "e0" to edge with tangent
// vector "e1" (edge tangent vectors are normalized)
template<typename Real_>
Vec3_T<Real_> parallelTransportNormalized(const Vec3_T<Real_> &t0, const Vec3_T<Real_> &t1, const Vec3_T<Real_> &v) {
    Vec3_T<Real_> sinThetaAxis = t0.cross(t1);
    Real_ cosTheta = t0.dot(t1);
    Real_ den = 1 + cosTheta;
    if (std::abs(stripAutoDiff(den)) < 1e-14) {
        // As t1 approaches -t0, the parallel transport operator becomes singular:
        // it approaches -(I - 2 a a^T), where a = (t0 x t1) / ||t0 x t1||.
        // In the neighborhood of t1 = -t0, this axis vector assumes all possible
        // values of unit vectors perpendicular to t0, so it is impossible to
        // define the parallel transport consistently in this neighborhood.
        // To avoid numerical blowups, we arbitrarily define the parallel transport
        // as the identity operator.
        // This case should only happen in practice when an edge length variable
        // inverts so that the edge it controls exactly flips. We have bound
        // constraints on the optimization to prevent this, but a naive finite
        // difference test can easily trigger this case).
        return v;
    }
    if (!isAutodiffType<Real_>()) {
        // Make parallelTransport(t, t, v) precisely the identity operation; this
        // is needed, e.g. to ensure rods with updated source frames can be
        // restored exactly from a file without small numerical perturbations.
        if ((t0 - t1).cwiseAbs().maxCoeff() == 0) return v; 
    }

    return (sinThetaAxis.dot(v) / (1 + cosTheta)) * sinThetaAxis
        + sinThetaAxis.cross(v)
        + cosTheta * v;
}

// Transport vector "v" from edge with tangent vector "e0" to edge with tangent
// vector "e1" (edge tangent vectors not necessarily normalized)
template<typename Real_>
Vec3_T<Real_> parallelTransport(Vec3_T<Real_> t0, Vec3_T<Real_> t1, const Vec3_T<Real_> &v) {
    t0.normalize();
    t1.normalize();
    return parallelTransportNormalized(t0, t1, v);
}

#endif /* end of include guard: VECTOROPERATIONS_HH */
