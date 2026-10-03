"""
pstone_models.py  --  the synthetic-data generators (the "generator ladder").

  1. RandomWalk        extra baseline: i.i.d. normal returns
  2. BlockBootstrap    baseline: stationary bootstrap of real log returns
  3. GJRGarchFHS       MAIN generator: Filtered Historical Simulation on a
                       GJR-GARCH(1,1) with Student-t errors
  4. PegAR1            stablecoins: mean-reverting model of the distance from $1
  5. TCopulaFHS        several assets at once: GJR-GARCH per asset + t-copula

All generators share one interface:

    gen = SomeGenerator().fit(training_log_returns)
    r = gen.sample_log_returns(n_paths, length, rng)     # shape (n_paths, length)

Everything is fitted on the training window only. To keep the code runnable
without extra packages, GJR-GARCH is estimated here with scipy (maximum
likelihood, standardised Student-t errors, the same model as
arch_model(y, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")).
SYNTHETIC DATA: nothing produced here is a real price.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, special, stats
from scipy.signal import lfilter

SCALE = 100.0   # GARCH is fitted on 100 * log_ret (returns in percent), as the context file asks


# ----------------------------------------------------------------------------
# GJR-GARCH(1,1) with standardised Student-t errors
# ----------------------------------------------------------------------------
def _sigma2(eps: np.ndarray, omega: float, alpha: float, gamma: float, beta: float) -> np.ndarray:
    """Conditional variance recursion:  s2[t] = omega + (alpha + gamma*1[e<0]) e[t-1]^2 + beta s2[t-1]."""
    n = len(eps)
    u = np.empty(n)
    u[0] = np.mean(eps[: min(75, n)] ** 2)          # back-cast for the first variance
    u[1:] = omega + (alpha + gamma * (eps[:-1] < 0)) * eps[:-1] ** 2
    return lfilter([1.0], [1.0, -beta], u)


def _neg_loglik(params: np.ndarray, y: np.ndarray) -> float:
    mu, omega, alpha, gamma, beta, nu = params
    eps = y - mu
    s2 = _sigma2(eps, omega, alpha, gamma, beta)
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
        return 1e10
    c = (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2)))
    ll = c - 0.5 * np.log(s2) - 0.5 * (nu + 1) * np.log1p(eps ** 2 / (s2 * (nu - 2)))
    return -float(np.sum(ll))


class GJRGarch:
    """GJR-GARCH(1,1), Student-t errors, constant mean. Fit on returns in percent."""

    def fit(self, y_pct: np.ndarray) -> "GJRGarch":
        y = np.asarray(y_pct, dtype=float)
        y = y[np.isfinite(y)]
        if len(y) < 250:
            raise ValueError(f"need at least 250 returns to fit GJR-GARCH, got {len(y)}")
        var = float(np.var(y))
        mu0 = float(np.mean(y))
        cons = [
            {"type": "ineq", "fun": lambda p: 0.9999 - (p[2] + 0.5 * p[3] + p[4])},   # stationarity
            {"type": "ineq", "fun": lambda p: p[2] + p[3]},                            # variance stays >= 0
        ]
        bounds = [(-5 * np.std(y), 5 * np.std(y)), (1e-8 * var, 10 * var), (0.0, 0.6),
                  (-0.3, 0.8), (0.0, 0.9999), (2.1, 100.0)]
        best = None
        for a0, g0, b0, nu0 in [(0.03, 0.08, 0.90, 8.0), (0.08, 0.05, 0.80, 5.0), (0.02, 0.04, 0.94, 15.0)]:
            pers = a0 + 0.5 * g0 + b0
            x0 = np.array([mu0, var * (1 - pers), a0, g0, b0, nu0])
            res = optimize.minimize(_neg_loglik, x0, args=(y,), method="SLSQP", bounds=bounds,
                                    constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
            if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
                best = res
        if best is None or best.fun >= 1e9:
            raise RuntimeError("GJR-GARCH fit failed")
        self.mu, self.omega, self.alpha, self.gamma, self.beta, self.nu = [float(v) for v in best.x]
        self.loglik = -float(best.fun)
        eps = y - self.mu
        self.sigma2_ = _sigma2(eps, self.omega, self.alpha, self.gamma, self.beta)
        z = eps / np.sqrt(self.sigma2_)
        self.z_ = z / np.std(z)                       # filtered (standardised) residuals, unit variance
        self.persistence = self.alpha + 0.5 * self.gamma + self.beta
        self._prepare_simulation(eps)
        return self

    def _prepare_simulation(self, eps: np.ndarray, target_persistence: float = 0.995) -> None:
        """
        Filtered historical simulation reuses the REAL residuals, which are not symmetric:
        falls are bigger than rises, so the true persistence of the simulated process is
            alpha + gamma * E[z^2 * 1(z<0)] + beta,
        which can be above 1 even when the t-model's persistence (alpha + gamma/2 + beta) is
        below 1. Left alone, some paths then explode. If that happens we scale alpha, gamma,
        beta down together to reach `target_persistence` and reset omega so the long-run
        variance equals the training-sample variance.
        """
        z = self.z_
        p_emp = self.alpha + self.gamma * float(np.mean(z ** 2 * (z < 0))) + self.beta
        var_target = float(np.var(eps))
        self.persistence_empirical = float(p_emp)
        self.adjusted = bool(p_emp > target_persistence)
        if self.adjusted:
            k = target_persistence / p_emp
            self.sim_alpha, self.sim_gamma, self.sim_beta = self.alpha * k, self.gamma * k, self.beta * k
            self.sim_omega = var_target * (1.0 - target_persistence)
            self.sim_start_var = var_target
        else:
            self.sim_alpha, self.sim_gamma, self.sim_beta, self.sim_omega = self.alpha, self.gamma, self.beta, self.omega
            self.sim_start_var = min(self.omega / max(1.0 - p_emp, 1e-4), 3.0 * var_target)
        self.sim_cap = 25.0 * var_target          # volatility never above 5x the sample level

    @property
    def params(self) -> dict:
        return dict(mu=self.mu, omega=self.omega, alpha=self.alpha, gamma=self.gamma,
                    beta=self.beta, nu=self.nu, persistence=self.persistence,
                    persistence_empirical=self.persistence_empirical,
                    adjusted_for_simulation=self.adjusted, loglik=self.loglik)

    def simulate_from_shocks(self, shocks: np.ndarray, burn: int = 100) -> np.ndarray:
        """
        Push standardised shocks (n_paths, total) through the GARCH recursion.
        Returns returns in PERCENT (n_paths, total - burn).
        """
        n_paths, total = shocks.shape
        om, al, ga, be = self.sim_omega, self.sim_alpha, self.sim_gamma, self.sim_beta
        s2 = np.full(n_paths, self.sim_start_var)
        out = np.empty((n_paths, total))
        for t in range(total):
            eps = np.sqrt(s2) * shocks[:, t]
            out[:, t] = self.mu + eps
            s2 = om + (al + ga * (eps < 0)) * eps ** 2 + be * s2
            np.minimum(s2, self.sim_cap, out=s2)
        return out[:, burn:]


class GJRGarchFHS:
    """MAIN generator: Filtered Historical Simulation on a GJR-GARCH(1,1)-t."""
    name = "garch_fhs"

    def __init__(self, burn: int = 100):
        self.burn = burn

    def fit(self, log_ret: np.ndarray) -> "GJRGarchFHS":
        self.garch = GJRGarch().fit(SCALE * np.asarray(log_ret, dtype=float))
        return self

    def sample_log_returns(self, n_paths: int, length: int, rng: np.random.Generator) -> np.ndarray:
        z = self.garch.z_
        shocks = z[rng.integers(0, len(z), size=(n_paths, length + self.burn))]
        return self.garch.simulate_from_shocks(shocks, self.burn) / SCALE


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------
class RandomWalk:
    """Extra baseline: i.i.d. normal returns with the training mean and standard deviation."""
    name = "random_walk"

    def fit(self, log_ret: np.ndarray) -> "RandomWalk":
        r = np.asarray(log_ret, dtype=float)
        self.mu, self.sd = float(np.mean(r)), float(np.std(r))
        return self

    def sample_log_returns(self, n_paths: int, length: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.mu, self.sd, size=(n_paths, length))


def _acov(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = x - x.mean()
    n = len(x)
    return np.array([np.dot(x[: n - k], x[k:]) / n for k in range(max_lag + 1)])


def optimal_block_length(x: np.ndarray) -> float:
    """
    Politis-White (2004) automatic block length for the STATIONARY bootstrap, with the
    Patton-Politis-White (2009) correction. Uses arch.bootstrap.optimal_block_length when the
    `arch` package is installed, otherwise a direct implementation of the same formula.
    """
    x = np.asarray(x, dtype=float)
    try:  # pragma: no cover - only if arch is installed
        from arch.bootstrap import optimal_block_length as _obl
        return float(_obl(x).iloc[0]["stationary"])
    except Exception:
        pass
    n = len(x)
    k_n = max(5, int(np.ceil(np.sqrt(np.log10(n)))))
    m_max = int(np.ceil(np.sqrt(n))) + k_n
    g = _acov(x, m_max)
    rho = g[1:] / g[0]
    thr = 2.0 * np.sqrt(np.log10(n) / n)
    below = np.abs(rho) < thr
    m_hat = m_max
    for m in range(0, m_max - k_n + 1):
        if below[m: m + k_n].all():
            m_hat = m
            break
    big_m = int(min(2 * max(m_hat, 1), m_max))
    k = np.arange(1, big_m + 1)
    t = k / big_m
    lam = np.where(t <= 0.5, 1.0, 2.0 * (1.0 - t))
    g0 = g[0] + 2.0 * np.sum(lam * g[1: big_m + 1])
    gg = 2.0 * np.sum(lam * k * g[1: big_m + 1])
    d_sb = 2.0 * g0 ** 2
    if d_sb <= 0:
        return 1.0
    b = (2.0 * gg ** 2 / d_sb) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    b_max = np.ceil(min(3.0 * np.sqrt(n), n / 3.0))
    return float(min(max(b, 1.0), b_max))


class BlockBootstrap:
    """
    Baseline: stationary bootstrap of the training log returns.

    Block length: Politis-White on the training returns. Plain returns have almost no
    autocorrelation, which gives blocks of ~1 day and would erase volatility clustering,
    so with block_basis="max" (default) we take the larger of the lengths found on the
    returns and on the squared returns. Use block_basis="returns" for the literal rule.
    """
    name = "bootstrap"

    def __init__(self, block_basis: str = "max", block_length: float | None = None):
        self.block_basis = block_basis
        self.fixed = block_length

    def fit(self, log_ret: np.ndarray) -> "BlockBootstrap":
        r = np.asarray(log_ret, dtype=float)
        self.r = r
        if self.fixed is not None:
            self.block = float(self.fixed)
        else:
            b_ret = optimal_block_length(r)
            b_sq = optimal_block_length(r ** 2)
            self.block = b_ret if self.block_basis == "returns" else max(b_ret, b_sq)
        return self

    def sample_log_returns(self, n_paths: int, length: int, rng: np.random.Generator) -> np.ndarray:
        n = len(self.r)
        jump = rng.random((n_paths, length)) < 1.0 / max(self.block, 1.0)
        jump[:, 0] = True
        start = rng.integers(0, n, size=(n_paths, length))
        pos = np.arange(length)[None, :]
        last = np.maximum.accumulate(np.where(jump, pos, 0), axis=1)
        base = np.take_along_axis(start, last, axis=1)
        idx = (base + pos - last) % n
        return self.r[idx]


# ----------------------------------------------------------------------------
# Stablecoins
# ----------------------------------------------------------------------------
class PegAR1:
    """
    Stablecoin generator: AR(1) on the distance from the peg d_t = close_t - 1,
        d_t = c + phi * d_{t-1} + e_t,
    with GJR-GARCH-t errors and filtered historical simulation, so de-pegs have realistic
    size, clustering and speed of recovery. Input is peg_dev (NOT log returns).
    sample_log_returns() converts the simulated price (1 + d_t) back to log returns so that
    every generator can be evaluated the same way.
    """
    name = "peg_ar1"

    def fit(self, peg_dev: np.ndarray) -> "PegAR1":
        d = np.asarray(peg_dev, dtype=float)
        x, yv = d[:-1], d[1:]
        phi, c = np.polyfit(x, yv, 1)
        self.phi, self.c = float(phi), float(c)
        e = yv - (self.c + self.phi * x)
        self.e_scale = float(np.std(e))
        self.garch = GJRGarch().fit(e / self.e_scale)        # unit-scale errors
        self.d0 = float(d[-1])
        self.d_all = d
        return self

    def sample_peg_dev(self, n_paths: int, length: int, rng: np.random.Generator, burn: int = 100) -> np.ndarray:
        z = self.garch.z_
        shocks = z[rng.integers(0, len(z), size=(n_paths, length + burn))]
        e = self.garch.simulate_from_shocks(shocks, burn=0) * self.e_scale   # mu~0, scaled back
        d = np.empty((n_paths, length + burn))
        prev = np.full(n_paths, self.d0)
        for t in range(length + burn):
            prev = self.c + self.phi * prev + e[:, t]
            d[:, t] = prev
        d = np.clip(d, -0.95, 5.0)                               # a price cannot go to or below zero
        return d[:, burn:]

    def sample_log_returns(self, n_paths: int, length: int, rng: np.random.Generator) -> np.ndarray:
        d = self.sample_peg_dev(n_paths, length, rng)
        price = 1.0 + d
        prev = np.concatenate([np.full((n_paths, 1), 1.0 + self.d0), price[:, :-1]], axis=1)
        return np.log(price / prev)


# ----------------------------------------------------------------------------
# Copula (several assets on the same dates)
# ----------------------------------------------------------------------------
class TCopulaFHS:
    """
    Multi-asset generator for a group panel (same dates for every asset):
      1. fit a GJR-GARCH-t to every asset,
      2. take the filtered residuals, turn them into uniforms with the empirical CDF,
      3. fit a Student-t copula (correlation from Kendall's tau, degrees of freedom by
         maximum likelihood on a grid),
      4. simulate copula draws, map back to residuals with the empirical quantiles,
      5. push them through each asset's GARCH recursion.
    Do not put near-duplicates (SPY with SPX, an ETF with its index) in one panel.
    """
    name = "t_copula_fhs"

    def fit(self, panel_log_ret: np.ndarray, columns=None) -> "TCopulaFHS":
        x = np.asarray(panel_log_ret, dtype=float)
        n, d = x.shape
        self.columns = list(columns) if columns is not None else [f"a{i}" for i in range(d)]
        self.garches = [GJRGarch().fit(SCALE * x[:, j]) for j in range(d)]
        self.Z = np.column_stack([g.z_ for g in self.garches])
        # pseudo-observations
        ranks = stats.rankdata(self.Z, axis=0)
        u = ranks / (n + 1.0)
        tau = np.eye(d)
        for i in range(d):
            for j in range(i + 1, d):
                t, _ = stats.kendalltau(self.Z[:, i], self.Z[:, j])
                tau[i, j] = tau[j, i] = t
        corr = np.sin(np.pi * tau / 2.0)
        w, v = np.linalg.eigh(corr)                              # make it a valid correlation matrix
        corr = (v * np.clip(w, 1e-6, None)) @ v.T
        dd = np.sqrt(np.diag(corr))
        self.corr = corr / np.outer(dd, dd)
        best_nu, best_ll = None, -np.inf
        for nu in [2.5, 3, 4, 5, 6, 8, 10, 15, 30, 60]:
            q = stats.t.ppf(u, nu)
            ll = (stats.multivariate_t(loc=np.zeros(d), shape=self.corr, df=nu).logpdf(q).sum()
                  - stats.t.logpdf(q, nu).sum())
            if ll > best_ll:
                best_nu, best_ll = nu, ll
        self.nu = best_nu
        self.chol = np.linalg.cholesky(self.corr)
        self.sorted_Z = np.sort(self.Z, axis=0)
        return self

    def sample_log_returns(self, n_paths: int, length: int, rng: np.random.Generator,
                           burn: int = 100, chunk: int = 100) -> np.ndarray:
        """Returns (n_paths, length, n_assets)."""
        d = len(self.garches)
        n = self.Z.shape[0]
        total = length + burn
        out = np.empty((n_paths, length, d))
        for s in range(0, n_paths, chunk):
            m = min(chunk, n_paths - s)
            g = rng.standard_normal((m, total, d)) @ self.chol.T
            w = rng.chisquare(self.nu, size=(m, total, 1)) / self.nu
            u = stats.t.cdf(g / np.sqrt(w), self.nu)
            idx = np.minimum((u * n).astype(int), n - 1)
            for j in range(d):
                shocks = self.sorted_Z[idx[:, :, j], j]
                out[s: s + m, :, j] = self.garches[j].simulate_from_shocks(shocks, burn) / SCALE
        return out


# ----------------------------------------------------------------------------
# helpers shared by the run scripts
# ----------------------------------------------------------------------------
def make_generator(name: str):
    """Build an (unfitted) generator by name."""
    table = {"random_walk": RandomWalk, "bootstrap": BlockBootstrap,
             "garch_fhs": GJRGarchFHS, "peg_ar1": PegAR1}
    if name not in table:
        raise KeyError(f"unknown generator '{name}'. Choose from {sorted(table)}")
    return table[name]()


def fit_generator(name: str, train_df):
    """Fit a single-asset generator on a training DataFrame (needs log_ret, or peg_dev for peg_ar1)."""
    gen = make_generator(name)
    col = "peg_dev" if name == "peg_ar1" else "log_ret"
    return gen.fit(train_df[col].to_numpy())


def prices_from_log_returns(log_ret: np.ndarray, start_price: float) -> np.ndarray:
    """Synthetic price paths = start price * exp(cumulative sum of log returns)."""
    return start_price * np.exp(np.cumsum(log_ret, axis=-1))
