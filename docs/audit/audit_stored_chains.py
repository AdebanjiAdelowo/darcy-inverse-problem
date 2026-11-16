"""Read-only audit of stored pCN chains. Usage: python -I audit_chains.py <repo_root>"""
import sys
import numpy as np

root = sys.argv[1]
sys.path.insert(0, root)
from src.pcn_sampler import effective_sample_size as ess_repo  # noqa: E402


def acf_fft(x):
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    a = np.fft.irfft(f * np.conj(f))[:n]
    return a / a[0]


def ess_geyer_full(x):
    """Geyer initial monotone sequence, no lag truncation."""
    rho = acf_fft(x)
    n = len(x)
    pairs = []
    k = 0
    while 2 * k + 1 < n:
        p = rho[2 * k] + rho[2 * k + 1]
        if p <= 0:
            break
        pairs.append(p)
        k += 1
    pairs = np.minimum.accumulate(np.array(pairs))
    tau = -1.0 + 2.0 * pairs.sum()
    return n / max(tau, 1.0)


def ess_batch_means(x, n_batches=20):
    n = len(x) // n_batches * n_batches
    b = x[:n].reshape(n_batches, -1).mean(axis=1)
    var_bm = (n // n_batches) * b.var(ddof=1)
    return n * x.var(ddof=1) / var_bm if var_bm > 0 else np.inf


def split_rhat(x, n_split=4):
    n = len(x) // n_split * n_split
    c = x[:n].reshape(n_split, -1)
    m = c.shape[1]
    W = c.var(axis=1, ddof=1).mean()
    B = m * c.mean(axis=1).var(ddof=1)
    return np.sqrt(((m - 1) / m * W + B / m) / W)


for truth in ["smooth", "structured"]:
    for cfg in ["full", "local"]:
        d = np.load(f"{root}/results/posterior_samples_{truth}_{cfg}.npz")
        xi = d["xi_samples"]
        n, r = xi.shape
        print(f"\n===== {truth} / {cfg}: n={n}, r={r}")
        print("lambda[:4] =", np.round(d["eigenvalues"][:4], 5))
        e_repo = np.array([ess_repo(xi[:, j]) for j in range(r)])
        e_full = np.array([ess_geyer_full(xi[:, j]) for j in range(r)])
        e_bm = np.array([ess_batch_means(xi[:, j]) for j in range(r)])
        rh = np.array([split_rhat(xi[:, j]) for j in range(r)])
        print("ESS repo (maxlag 1000) :", np.round(e_repo, 1).tolist())
        print("ESS Geyer untruncated  :", np.round(e_full, 1).tolist())
        print("ESS batch means (20)   :", np.round(e_bm, 1).tolist())
        print(f"  median repo={np.median(e_repo):.1f} untrunc={np.median(e_full):.1f} bm={np.median(e_bm):.1f}")
        print("split-Rhat (4 within-chain segments):", np.round(rh, 2).tolist(),
              f" max={rh.max():.2f}, #>1.01={int((rh > 1.01).sum())}")
        print(f"xi_0 (constant mode): mean={xi[:, 0].mean():+.3f}, sd={xi[:, 0].std():.3f}"
              f"  [exact posterior: N(0,1)]; range=[{xi[:, 0].min():+.3f},{xi[:, 0].max():+.3f}]")
        rho0 = acf_fft(xi[:, 0])
        print("  ACF xi_0 at lags 100/500/1000/2000:", np.round(rho0[[100, 500, 1000, 2000]], 3).tolist())
        print("sample sd per dim:", np.round(xi.std(axis=0), 3).tolist())

        mt, mmap, mpm, msd = d["m_true"], d["m_map"], d["m_post_mean"], d["m_post_std"]
        rel = lambda a: np.linalg.norm(a - mt) / np.linalg.norm(mt)
        c = lambda a: a - a.mean()
        relc = lambda a: np.linalg.norm(c(a) - c(mt)) / np.linalg.norm(c(mt))
        print(f"rel err MAP={rel(mmap):.4f}  postmean={rel(mpm):.4f}")
        print(f"  constant-offset (nodal mean) truth={mt.mean():+.4f} MAP={mmap.mean():+.4f} postmean={mpm.mean():+.4f}")
        print(f"  mean-removed rel err MAP={relc(mmap):.4f}  postmean={relc(mpm):.4f}")
        print(f"pointwise post sd: min={msd.min():.3f} median={np.median(msd):.3f} max={msd.max():.3f}"
              "  [exact posterior: >= 1 everywhere, from xi_0 alone]")
