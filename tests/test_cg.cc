#include "../cg_solver.hh"

int main(int /* argc */, const char * /* argv */ []) {
    int size = 20;
    using VecX = Eigen::VectorXd;
    VecX Adiag, b, x;
    Adiag.setRandom(size);
    b.setRandom(size);
    // Adiag += 0.725 * VecX::Ones(size);
    Adiag += 1.0 * VecX::Ones(size);
    x.setZero(size);

    cg_solver([&](const VecX &v) { return Adiag.cwiseProduct(v).eval(); },
              b, x,
              [&](size_t k, const VecX &r) { std::cout << "residual " << k << " norm: " << r.norm() << ", " << " x norm: " << x.norm() << std::endl; },
              50, 0.0);

    return 0;
}
