"""
test_pstone.py  --  automatic checks for the P Stone code. Run:   python -m unittest -v test_pstone

The tests use a fake market whose true rules we know, so they need no real data. One extra test
(skipped if the data folder is missing) checks that your real data files load cleanly.
"""
import unittest

import numpy as np
import pandas as pd

import pstone_config as cfg
import pstone_data as D
import pstone_gate as G
import pstone_utility as U
from pstone_models import (BlockBootstrap, GJRGarch, GJRGarchFHS, PegAR1, RandomWalk, TCopulaFHS,
                           optimal_block_length, prices_from_log_returns)
from pstone_leveraged import leveraged_paths


def fake_gjr_returns(n=6000, omega=0.02, alpha=0.03, gamma=0.10, beta=0.90, nu=6, seed=0):
    """Log returns from a known GJR-GARCH-t process (true rules known)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, size=n + 500) * np.sqrt((nu - 2) / nu)
    s2 = omega / (1 - alpha - gamma / 2 - beta)
    out = np.empty(n + 500)
    for t in range(n + 500):
        e = np.sqrt(s2) * z[t]
        out[t] = e
        s2 = omega + (alpha + gamma * (e < 0)) * e ** 2 + beta * s2
    return out[500:] / 100.0


class TestGarchAndFHS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = fake_gjr_returns()
        cls.fhs = GJRGarchFHS().fit(cls.r)

    def test_fit_recovers_persistence_and_asymmetry(self):
        p = self.fhs.garch.params
        self.assertAlmostEqual(p["persistence"], 0.03 + 0.05 + 0.90, delta=0.04)
        self.assertGreater(p["gamma"], 0.03)                  # leverage effect found
        self.assertTrue(4 < p["nu"] < 15)

    def test_fhs_reproduces_fat_tails_and_clustering(self):
        paths = self.fhs.sample_log_returns(200, 3000, np.random.default_rng(1))
        m = G.path_metrics(paths)
        self.assertGreater(np.median(m["excess_kurtosis"]), 1.0)
        self.assertGreater(np.median(m["acf1_sq"]), 0.05)

    def test_fhs_paths_do_not_explode(self):
        paths = self.fhs.sample_log_returns(300, 3000, np.random.default_rng(2))
        self.assertLess(np.abs(paths).max(), 0.6)
        self.assertLess(np.quantile(paths.std(axis=1), 0.99), 6 * self.r.std())

    def test_same_seed_same_paths(self):
        a = self.fhs.sample_log_returns(5, 50, np.random.default_rng(7))
        b = self.fhs.sample_log_returns(5, 50, np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)

    def test_too_few_observations_is_an_error(self):
        with self.assertRaises(ValueError):
            GJRGarch().fit(np.random.randn(100))


class TestBaselines(unittest.TestCase):
    def test_bootstrap_only_uses_real_values(self):
        r = np.random.default_rng(0).standard_normal(1000) / 100
        b = BlockBootstrap().fit(r)
        s = b.sample_log_returns(4, 300, np.random.default_rng(1))
        self.assertEqual(s.shape, (4, 300))
        self.assertTrue(np.isin(s, r).all())

    def test_block_length_is_small_for_noise_and_large_for_persistent_series(self):
        noise = np.random.default_rng(0).standard_normal(4000)
        ar = np.zeros(4000); e = np.random.default_rng(1).standard_normal(4000)
        for t in range(1, 4000):
            ar[t] = 0.9 * ar[t - 1] + e[t]
        self.assertLess(optimal_block_length(noise), 8)
        self.assertGreater(optimal_block_length(ar), optimal_block_length(noise))

    def test_random_walk_has_no_clustering(self):
        rw = RandomWalk().fit(fake_gjr_returns())
        m = G.path_metrics(rw.sample_log_returns(100, 3000, np.random.default_rng(0)))
        self.assertLess(abs(np.median(m["acf1_sq"])), 0.05)


class TestPeg(unittest.TestCase):
    def test_peg_model_stays_near_one_but_random_walk_drifts(self):
        rng = np.random.default_rng(0)
        d = np.zeros(2500); e = rng.standard_t(5, 2500) * 0.0005
        for t in range(1, 2500):
            d[t] = 0.7 * d[t - 1] + e[t]
        price = 1 + d
        log_ret = np.diff(np.log(price), prepend=np.log(price[0]))
        peg = PegAR1().fit(d)
        p_peg = 1 + peg.sample_peg_dev(200, 2500, np.random.default_rng(1))
        p_rw = prices_from_log_returns(RandomWalk().fit(log_ret).sample_log_returns(200, 2500, np.random.default_rng(1)), 1.0)
        end_peg = np.mean(np.abs(p_peg[:, -1] - 1) > 0.02)
        end_rw = np.mean(np.abs(p_rw[:, -1] - 1) > 0.02)
        self.assertLess(end_peg, 0.05)
        self.assertGreater(end_rw, 0.5)
        self.assertLess(peg.phi, 0.95)                        # mean-reverting

    def test_peg_log_returns_rebuild_the_price(self):
        d = 0.001 * np.sin(np.arange(1500) / 7.0) + 0.0003 * np.random.default_rng(0).standard_normal(1500)
        peg = PegAR1().fit(d)
        r = peg.sample_log_returns(3, 200, np.random.default_rng(2))
        price = (1 + peg.d0) * np.exp(np.cumsum(r, axis=1))
        self.assertTrue(np.all(price > 0))


class TestCopula(unittest.TestCase):
    def test_simulated_correlation_matches_input(self):
        a = fake_gjr_returns(4000, seed=1)
        b = 0.7 * a + 0.3 * fake_gjr_returns(4000, seed=2)    # strongly linked
        c = fake_gjr_returns(4000, seed=3)                     # independent
        X = np.column_stack([a, b, c])
        gen = TCopulaFHS().fit(X, ["a", "b", "c"])
        paths = gen.sample_log_returns(60, 2000, np.random.default_rng(0))
        self.assertEqual(paths.shape, (60, 2000, 3))
        real_ab = np.corrcoef(a, b)[0, 1]
        syn_ab = np.median([np.corrcoef(p[:, 0], p[:, 1])[0, 1] for p in paths])
        syn_ac = np.median([np.corrcoef(p[:, 0], p[:, 2])[0, 1] for p in paths])
        self.assertAlmostEqual(real_ab, syn_ab, delta=0.12)
        self.assertLess(abs(syn_ac), 0.15)


class TestGate(unittest.TestCase):
    def test_fidelity_fails_random_walk_and_passes_matching_generator(self):
        r = fake_gjr_returns()
        rw = RandomWalk().fit(r).sample_log_returns(150, len(r), np.random.default_rng(0))
        fhs = GJRGarchFHS().fit(r).sample_log_returns(150, len(r), np.random.default_rng(0))
        s_rw = G.fidelity_summary(G.fidelity_table(r, rw))
        s_fhs = G.fidelity_summary(G.fidelity_table(r, fhs))
        self.assertLess(s_rw["fidelity_pass_rate"], 0.5)
        self.assertGreater(s_fhs["fidelity_pass_rate"], s_rw["fidelity_pass_rate"])

    def test_novelty_flags_copies_of_history(self):
        r = fake_gjr_returns(3000)
        copies = np.stack([np.roll(r, 40 * k) for k in range(1, 6)])
        fresh = GJRGarchFHS().fit(r).sample_log_returns(5, 3000, np.random.default_rng(0))
        n_copy = G.novelty_report(r, copies)["near_copy_share"]
        n_fresh = G.novelty_report(r, fresh)["near_copy_share"]
        self.assertGreater(n_copy, 0.5)
        self.assertGreater(n_copy, n_fresh)


class TestUtilityPieces(unittest.TestCase):
    def test_target_uses_only_future_rows(self):
        r = np.random.default_rng(0).standard_normal(300) / 100
        X, y, ix = U.make_features(r, 5)
        i = ix[10]
        self.assertAlmostEqual(y[10], np.log(np.sqrt(np.sum(r[i + 1: i + 6] ** 2))), places=10)
        r2 = r.copy(); r2[i + 7:] = 0.5                       # change the far future: features at i must not move
        X2, _, ix2 = U.make_features(r2, 5)
        np.testing.assert_allclose(X[10], X2[list(ix2).index(i)])

    def test_diebold_mariano(self):
        rng = np.random.default_rng(0)
        ref = rng.chisquare(1, 500) + 0.5
        better = ref - 0.3 + 0.05 * rng.standard_normal(500)
        self.assertLess(U.diebold_mariano(ref, better, 5)["p_one_sided"], 0.01)
        self.assertGreater(U.diebold_mariano(better, ref, 5)["p_one_sided"], 0.9)


class TestData(unittest.TestCase):
    def _df(self, start="2000-01-03", n=7000):
        d = pd.bdate_range(start, periods=n)
        return pd.DataFrame({"date": d, "close": 100.0, "log_ret": 0.001})

    def test_walk_forward_has_no_leakage(self):
        df = self._df()
        train, test = D.walk_forward_split(df, 2022)
        self.assertLess(train["date"].max(), pd.Timestamp("2022-01-01"))
        self.assertTrue((test["date"].dt.year == 2022).all())

    def test_needs_three_years_of_history(self):
        df = self._df(start="2020-06-01", n=1500)
        self.assertIsNone(D.walk_forward_split(df, 2022))
        self.assertIsNotNone(D.walk_forward_split(df, 2024))

    def test_exclusion_keeps_price_level(self):
        import tempfile, os
        df = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=6), "close": [1, 1, 5, 1, 1, 1.1],
                           "log_ret": [0.0, 0.0, np.log(5), -np.log(5), 0.0, np.log(1.1)], "gap_days": [1] * 6})
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "ex.csv")
            pd.DataFrame({"asset_id": ["X"], "date": ["2020-01-03"], "reason": ["test"]}).to_csv(p, index=False)
            out = D.apply_exclusions(df, "X", p)
        self.assertEqual(len(out), 5)
        self.assertAlmostEqual(out["log_ret"].sum(), df["log_ret"].sum(), places=12)
        self.assertEqual(out["gap_days"].sum(), df["gap_days"].sum())

    @unittest.skipUnless((cfg.DATA_DIR / "manifest.csv").exists(), "data folder not found - skipping")
    def test_real_data_files_load_cleanly(self):
        m = D.load_manifest()
        for asset in m["asset_id"]:
            df = D.load_close(asset)
            self.assertTrue(np.isfinite(df["log_ret"]).all(), asset)
            self.assertTrue(df["date"].is_monotonic_increasing, asset)
            self.assertGreater(len(df), 1000, asset)


class TestLeveragedLayer(unittest.TestCase):
    def test_one_day_is_three_times_the_move(self):
        p = leveraged_paths(np.log([[1.01]]), 3.0, expense=0, rf=0, spread=0, start_price=100)
        self.assertAlmostEqual(p[0, 0], 103.0, places=6)

    def test_volatility_decay(self):
        r = np.tile(np.log([1.05, 1 / 1.05]), 100)[None, :]     # index goes up and down, ends flat
        p = leveraged_paths(r, 3.0, expense=0, rf=0, spread=0)
        self.assertAlmostEqual(float(np.exp(r.sum())), 1.0, places=9)
        self.assertLess(p[0, -1], 100.0)


if __name__ == "__main__":
    unittest.main()
