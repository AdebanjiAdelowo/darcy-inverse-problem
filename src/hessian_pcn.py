"""Hessian-informed, prior-reversible pCN (HpCN) for a whitened Gaussian prior.

## Setting

Target on xi in R^d (here the d = 14 mean-free KL coefficients):

    pi(d xi)  propto  exp(-Phi(xi)) mu_0(d xi),     mu_0 = N(0, I_d).

## Which Gaussian proposals are prior-reversible

Let Q(xi, .) = N(A xi, S). Q is reversible with respect to mu_0, i.e.
mu_0(d xi) Q(xi, d xi') = mu_0(d xi') Q(xi', d xi), if and only if the joint law of
(xi, xi') under xi ~ mu_0 is exchangeable. That joint law is the zero-mean Gaussian with
covariance [[I, A^T], [A, A A^T + S]], which is invariant under swapping the blocks iff

    A = A^T   and   S = I - A^2   (so ||A|| <= 1).

pCN is the special case A = sqrt(1 - beta^2) I. A Hessian-preconditioned random walk
xi' = xi + s L eta (A = I, S = s^2 L L^T) is NOT prior-reversible (A^2 + S != I); it is
reversible with respect to Lebesgue measure and would need the prior-density ratio in its
acceptance. Proposals centred at the MAP, N(m + a(xi - m), .), are reversible with respect
to N(m, .), not mu_0. The generalised pCN of Rudolf & Sprungk (2018), N(sqrt(1-s^2) xi, s^2 C),
is reversible with respect to N(0, C), again not mu_0 unless C = I.

## HpCN

Take the Gauss-Newton misfit Hessian at the MAP in whitened coordinates,
H = G^T Gamma^-1 G = V diag(lambda) V^T (V orthogonal), and set

    A = V diag(a) V^T,  S = V diag(beta^2) V^T,  a_i = sqrt(1 - beta_i^2),
    beta_i = min(beta_max, rho / sqrt(1 + lambda_i)),   0 < beta_i <= 1.

Then A = A^T and A^2 + S = I, so Q is exactly mu_0-reversible. In eigen-coordinates
z = V^T xi the proposal is z_i' = a_i z_i + beta_i eta_i: an independent pCN step per
direction, whose size follows the Laplace posterior scale (1 + lambda_i)^{-1/2} in
data-informed directions and approaches the prior scale in prior-dominated ones.

## Acceptance and posterior invariance

For a mu_0-reversible proposal, the Metropolis-Hastings ratio is

    pi(xi') q(xi', xi) / (pi(xi) q(xi, xi')) = exp(Phi(xi) - Phi(xi')) *
        [mu_0(xi') q(xi', xi)] / [mu_0(xi) q(xi, xi')] = exp(Phi(xi) - Phi(xi')),

so alpha(xi, xi') = min(1, exp(Phi(xi) - Phi(xi'))), exactly as in pCN, and the kernel
satisfies detailed balance with respect to pi. Prior reversibility is a property of the
proposal alone; posterior invariance is a property of the full Metropolis-Hastings kernel,
and follows here from prior reversibility plus the likelihood-ratio acceptance. The proposal
is fixed (computed once at the MAP), so the chain is a time-homogeneous Markov chain.

A proposal whose Phi is not finite is rejected (this is the MH rule with Phi = +inf).

The constant KL mode is not part of xi: it is handled exactly by src/constant_mode.py.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class HessianPCNConfig:
    rho: float = 0.5               # global proposal scale (0, 1]
    beta_max: float = 1.0          # cap on per-direction step (<= 1)
    eig_floor: float = 0.0         # eigenvalues below this are raised to it (>= 0)
    eig_cap: float | None = None   # optional upper cap on eigenvalues
    sym_rtol: float = 1e-10        # tolerated relative asymmetry of the input Hessian
    psd_rtol: float = 1e-8         # tolerated negative eigenvalue relative to the largest

    def validate(self):
        if not (0.0 < self.rho <= 1.0):
            raise ValueError("rho must be in (0, 1]")
        if not (0.0 < self.beta_max <= 1.0):
            raise ValueError("beta_max must be in (0, 1]")
        if self.eig_floor < 0:
            raise ValueError("eig_floor must be >= 0")


class HessianInformedProposal:
    """Fixed mu_0-reversible Gaussian proposal N(A xi, I - A^2) built from a Hessian."""

    def __init__(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray, config: HessianPCNConfig,
                 raw_eigenvalues: np.ndarray | None = None):
        config.validate()
        self.config = config
        self.lam = np.asarray(eigenvalues, dtype=float)
        self.V = np.asarray(eigenvectors, dtype=float)
        self.raw_lam = self.lam if raw_eigenvalues is None else np.asarray(raw_eigenvalues, dtype=float)
        self.beta = np.minimum(config.beta_max, config.rho / np.sqrt(1.0 + self.lam))
        self.a = np.sqrt(1.0 - self.beta ** 2)
        if not np.allclose(self.V.T @ self.V, np.eye(len(self.lam)), atol=1e-10):
            raise ValueError("eigenvectors are not orthonormal")

    @classmethod
    def from_hessian(cls, H: np.ndarray, config: HessianPCNConfig) -> "HessianInformedProposal":
        config.validate()
        H = np.asarray(H, dtype=float)
        scale = max(np.max(np.abs(H)), 1e-300)
        asym = np.max(np.abs(H - H.T)) / scale
        if asym > config.sym_rtol:
            raise ValueError(f"Hessian not symmetric (relative asymmetry {asym:.2e})")
        lam, V = np.linalg.eigh(0.5 * (H + H.T))
        if lam.min() < -config.psd_rtol * max(lam.max(), 1.0):
            raise ValueError(f"Hessian not positive semi-definite (min eigenvalue {lam.min():.3e})")
        raw = lam.copy()
        lam = np.maximum(lam, config.eig_floor)
        if config.eig_cap is not None:
            lam = np.minimum(lam, config.eig_cap)
        return cls(lam, V, config, raw_eigenvalues=raw)

    @property
    def dim(self) -> int:
        return len(self.lam)

    def propose(self, xi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        eta = rng.standard_normal(self.dim)
        z = self.V.T @ xi
        return self.V @ (self.a * z + self.beta * eta)

    def matrices(self):
        A = (self.V * self.a) @ self.V.T
        S = (self.V * self.beta ** 2) @ self.V.T
        return A, S

    def log_q(self, xi: np.ndarray, xi_new: np.ndarray) -> float:
        """log density of N(A xi, S) at xi_new (for tests)."""
        z = self.V.T @ xi_new - self.a * (self.V.T @ xi)
        return float(-0.5 * np.sum((z / self.beta) ** 2) - np.sum(np.log(self.beta))
                     - 0.5 * self.dim * np.log(2 * np.pi))

    def to_dict(self) -> dict:
        return {"config": asdict(self.config), "eigenvalues": self.lam.tolist(),
                "raw_eigenvalues": self.raw_lam.tolist(), "eigenvectors": self.V.tolist(),
                "beta": self.beta.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "HessianInformedProposal":
        return cls(np.array(d["eigenvalues"]), np.array(d["eigenvectors"]), HessianPCNConfig(**d["config"]),
                   raw_eigenvalues=np.array(d["raw_eigenvalues"]))


class HpCNChain:
    """One restartable HpCN chain. The full state (current point, its Phi, the
    NumPy Generator state, counters, proposal) is captured by state_dict(), so a
    resumed chain continues bit for bit."""

    def __init__(self, neg_log_lik, proposal: HessianInformedProposal, x0: np.ndarray, seed: int,
                 phi0: float | None = None):
        self.nll = neg_log_lik
        self.proposal = proposal
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.x = np.array(x0, dtype=float)
        self.n_lik_evals = 0
        if phi0 is None:
            phi0 = self.nll(self.x)
            self.n_lik_evals += 1
        self.phi = float(phi0)
        self.n_steps = 0
        self.n_accept = 0
        self.n_nonfinite = 0
        self.seconds = 0.0

    def step(self, n: int, thin: int = 1):
        """Advance n proposals; return (draws, phis) at global steps divisible by thin."""
        xs, ps = [], []
        t0 = time.perf_counter()
        for _ in range(n):
            prop = self.proposal.propose(self.x, self.rng)
            phi_prop = self.nll(prop)
            self.n_lik_evals += 1
            log_u = np.log(self.rng.random())
            if not np.isfinite(phi_prop):
                self.n_nonfinite += 1
            elif log_u < self.phi - phi_prop:
                self.x, self.phi = prop, float(phi_prop)
                self.n_accept += 1
            self.n_steps += 1
            if self.n_steps % thin == 0:
                xs.append(self.x.copy())
                ps.append(self.phi)
        self.seconds += time.perf_counter() - t0
        return np.array(xs).reshape(-1, self.proposal.dim), np.array(ps)

    @property
    def acceptance_rate(self) -> float:
        return self.n_accept / self.n_steps if self.n_steps else float("nan")

    def state_dict(self) -> dict:
        return {"x": self.x.tolist(), "phi": self.phi, "rng_state": self.rng.bit_generator.state,
                "seed": self.seed, "n_steps": self.n_steps, "n_accept": self.n_accept,
                "n_lik_evals": self.n_lik_evals, "n_nonfinite": self.n_nonfinite, "seconds": self.seconds,
                "proposal": self.proposal.to_dict()}

    @classmethod
    def from_state(cls, neg_log_lik, state: dict) -> "HpCNChain":
        prop = HessianInformedProposal.from_dict(state["proposal"])
        ch = cls(neg_log_lik, prop, np.array(state["x"]), state["seed"], phi0=state["phi"])
        ch.rng.bit_generator.state = state["rng_state"]
        for k in ("n_steps", "n_accept", "n_lik_evals", "n_nonfinite", "seconds"):
            setattr(ch, k, state[k])
        return ch


def save_checkpoint(path, chain: HpCNChain, draws: np.ndarray, phis: np.ndarray, extra: dict | None = None):
    """Atomic checkpoint: chain state + all retained draws so far."""
    import os
    path = str(path)
    tmp = path + ".tmp.npz"
    np.savez(tmp, draws=draws, phis=phis,
             state=json.dumps(chain.state_dict()), extra=json.dumps(extra or {}))
    os.replace(tmp, path)


def load_checkpoint(path):
    with np.load(path) as d:
        return d["draws"], d["phis"], json.loads(str(d["state"])), json.loads(str(d["extra"]))
