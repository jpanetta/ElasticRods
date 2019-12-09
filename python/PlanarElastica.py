# Semi-analytic solution to the static equilibrium problem for an
# incompressible elastic rod with two fixed endpoints.
import scipy, numpy as np
from scipy.special import ellipe, ellipk, ellipeinc, ellipkinc
from scipy.optimize import newton

class AnalyticRod:
    def __init__(self, L, a):
        """ Construct an elastic rod of length L in equilibrium
        L -- Arc length of the rod
        a -- Distance the rod endpoints are held from each other.
        """
        self.L = L
        # Find the elliptic modulus describing the equilibrium shape of the rod constrained to span width 'a'
        self.m = newton(lambda m: 2 * ellipe(m) / ellipk(m) - 1 - a / L, 0.5)
        self.C = 1.0 / (2.0 * ellipk(self.m))

    def height(self):
        return 2 * self.L * self.C * np.sqrt(max(0, self.m))

    # Compute points on the deformed rod by evaluating its parametric curve representation at a list of phis
    # Can be called like:
    #   x, y = r.pts(np.linspace(-np.pi/2, np.pi/2, 100))
    def pts(self, phis): return self.L * self.C * np.row_stack((2 * ellipeinc(phis, self.m) - ellipkinc(phis, self.m),
                                                                2 * np.sqrt(self.m) * np.cos(phis)))

    def force(self, EI): return EI / ((self.L * self.C) ** 2)
