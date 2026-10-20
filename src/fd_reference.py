"""Classical reference estimator for the two-parameter spatial Darcy benchmark.

Independent of DeepXDE and of the FEM code: a cell-centred-face, conservative
five-point finite-difference solver for -div(k grad p) = f on the unit square
with zero Dirichlet data, k = exp(alpha x + beta y), and the benchmark's known
forcing f (built from the true parameters). On top of it:

* ``jacobian``: sensitivities d p(x_i) / d(alpha, beta) at the sensors;
* ``fisher``: Gaussian Fisher information J^T J / sigma^2 and its inverse
  (Cramer-Rao bound for unbiased estimators);
* ``mle``: Gauss-Newton least squares, i.e. the maximum-likelihood estimate
  when the PDE holds exactly and the noise is i.i.d. N(0, sigma^2).

Used only for diagnostics (CPU, sparse linear solves; no training).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RectBivariateSpline

from src.deepxde_inverse_spatial_k import ALPHA_TRUE, BETA_TRUE, manufactured_forcing


class DarcyFD:
    """Finite-difference forward map theta -> p on an (n+1) x (n+1) node grid."""

    def __init__(self, n=200, alpha_true=ALPHA_TRUE, beta_true=BETA_TRUE):
        if n < 4:
            raise ValueError("n must be at least 4")
        self.n = n
        self.h = 1.0 / n
        self.nodes = np.linspace(0.0, 1.0, n + 1)
        inner = self.nodes[1:-1]
        xx, yy = np.meshgrid(inner, inner, indexing="ij")
        self._xx, self._yy = xx, yy
        pts = np.stack((xx.ravel(), yy.ravel()), axis=1)
        self.forcing = manufactured_forcing(pts, alpha_true, beta_true).ravel()

    def _matrix(self, alpha, beta):
        n, h = self.n, self.h
        m = n - 1
        x, y = self._xx, self._yy
        k = lambda xs, ys: np.exp(alpha * xs + beta * ys)
        ke, kw = k(x + h / 2, y), k(x - h / 2, y)
        kn, ks = k(x, y + h / 2), k(x, y - h / 2)
        idx = np.arange(m * m).reshape(m, m)
        rows, cols, vals = [idx.ravel()], [idx.ravel()], [(ke + kw + kn + ks).ravel()]
        for coeff, di, dj in ((ke, 1, 0), (kw, -1, 0), (kn, 0, 1), (ks, 0, -1)):
            i0, i1 = max(0, -di), m - max(0, di)
            j0, j1 = max(0, -dj), m - max(0, dj)
            rows.append(idx[i0:i1, j0:j1].ravel())
            cols.append(idx[i0 + di:i1 + di, j0 + dj:j1 + dj].ravel())
            vals.append(-coeff[i0:i1, j0:j1].ravel())
        a = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(m * m, m * m))
        return a / h**2

    def solve(self, alpha, beta):
        """Nodal pressure on the full grid (boundary zeros included)."""
        m = self.n - 1
        interior = spla.spsolve(self._matrix(alpha, beta).tocsc(), self.forcing)
        p = np.zeros((self.n + 1, self.n + 1))
        p[1:-1, 1:-1] = interior.reshape(m, m)
        return p

    def at(self, p, points):
        """Bicubic interpolation of a nodal field at arbitrary points."""
        spline = RectBivariateSpline(self.nodes, self.nodes, p, kx=3, ky=3)
        points = np.asarray(points)
        return spline.ev(points[:, 0], points[:, 1])

    def predict(self, theta, points):
        return self.at(self.solve(*theta), points)

    def jacobian(self, theta, points, step=1e-4):
        cols = []
        for j in range(2):
            d = np.zeros(2)
            d[j] = step
            cols.append((self.predict(theta + d, points) - self.predict(theta - d, points))
                        / (2 * step))
        return np.stack(cols, axis=1)

    def fisher(self, theta, points, sigma):
        jac = self.jacobian(np.asarray(theta, dtype=float), points)
        info = jac.T @ jac / sigma**2
        return info, np.linalg.inv(info)

    def mle(self, points, values, theta0=(0.0, 0.0), tol=1e-10, max_iter=50):
        """Gauss-Newton least squares; returns (theta, residual MSE, iterations)."""
        theta = np.asarray(theta0, dtype=float)
        values = np.asarray(values).ravel()
        for it in range(1, max_iter + 1):
            r = self.predict(theta, points) - values
            jac = self.jacobian(theta, points)
            step = np.linalg.lstsq(jac, -r, rcond=None)[0]
            theta = theta + step
            if np.max(np.abs(step)) < tol:
                break
        r = self.predict(theta, points) - values
        return theta, float(np.mean(r**2)), it
