"""Covering constant of the stage-mixture efficiency theorem (RARECERT, PEIS version).

For a stage whose block-diagonal quadratic form has top eigenvalue 1 (dominant block H),
top eigenvalue varrho in [0, 1] on the other block (Lo) and all remaining eigenvalues
at most nu, the normalised second-moment exponent of a point with coordinates
a (along the H Perron direction), b (along the Lo Perron direction) and g (norm of the
rest) under the mixture {defensive, shift sqrt(2) e_H, shift sqrt(2/varrho) e_Lo} is

    F(a, b, g) = max( a^2 + b^2 + g^2,
                      (a + sqrt 2)^2 + b^2 + g^2 - 4,
                      a^2 + (b + sqrt(2/varrho))^2 + g^2 - 4/varrho ),

and the overflow region (scaled to level 1) is contained in
    a_+^2 + varrho b_+^2 + nu g^2 >= 2.
psi(varrho, nu) = inf F over that region. Since F increases in g, the infimum over g is
at g^2 = max(0, (2 - a_+^2 - varrho b_+^2)/nu), so psi is a two-dimensional minimum,
computed here on a grid and refined locally. use_lo=False drops the Lo component.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

SQ2 = math.sqrt(2.0)


def _F(a, b, varrho, nu, use_lo, pen_h=0.0, pen_lo=0.0):
    ap = np.maximum(a, 0.0)
    bp = np.maximum(b, 0.0)
    need = 2.0 - ap * ap - varrho * bp * bp
    if nu > 0:
        g2 = np.maximum(need, 0.0) / nu
    else:
        g2 = np.where(need > 1e-12, np.inf, 0.0)
    base = a * a + b * b + g2
    f = np.maximum(base, (a + SQ2) ** 2 + b * b + g2 - 4.0 - pen_h)
    if use_lo and varrho > 0:
        r = math.sqrt(2.0 / varrho)
        f = np.maximum(f, a * a + (b + r) ** 2 + g2 - 4.0 / varrho - pen_lo)
    return f


def psi(varrho: float, nu: float, use_lo: bool = True, n: int = 801,
        pen_h: float = 0.0, pen_lo: float = 0.0) -> float:
    """Covering constant psi(varrho, nu); 4 means the stage is covered at any scale.

    pen_h, pen_lo >= 0 subtract the normalised log-weight penalties 2 lam_k delta_m of
    components whose weights decay like exp(-c delta_m) (0 for constant weights).
    """
    if not (0.0 <= varrho <= 1.0 and nu >= 0.0):
        raise ValueError("need 0 <= varrho <= 1 and nu >= 0")
    if varrho == 0.0:
        bs = np.zeros(1)
    else:
        bs = np.linspace(-2.2, 2.2, n)
    as_ = np.linspace(-2.2, 2.2, n)
    A, B = np.meshgrid(as_, bs, indexing="ij")
    Fv = _F(A, B, varrho, nu, use_lo, pen_h, pen_lo)
    i = np.unravel_index(np.argmin(Fv), Fv.shape)
    best = float(Fv[i])
    x0 = np.array([A[i], B[i]])

    def fun(x):
        return float(_F(np.array(x[0]), np.array(x[1] if varrho > 0 else 0.0), varrho, nu, use_lo, pen_h, pen_lo))

    res = minimize(fun, x0, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000})
    return min(best, float(res.fun))
