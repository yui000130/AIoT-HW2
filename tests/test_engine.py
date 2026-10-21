"""mlr_engine 單元測試：python -m unittest discover -s tests -v"""
import json
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mlr_engine import profile_columns, read_csv_any, run_analysis, run_config, suggest_target  # noqa: E402
from mlr_engine.prep import prepare  # noqa: E402


def synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "x1": rng.normal(10, 2, n),
        "x2": rng.normal(0, 1, n),
        "noise": rng.normal(0, 1, n),
        "city": rng.choice(["A", "B", "C"], n, p=[.5, .3, .2]),
        "when": pd.date_range("2020-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
    })
    df["y"] = 5 + 3 * df.x1 - 2 * df.x2 + np.where(df.city == "C", 4, 0) + rng.normal(0, 1, n)
    return df


class TestIO(unittest.TestCase):
    def test_semicolon_and_currency(self):
        text = "a;b;y\n1;$1,000;10%\n2;$2,000;20%\n3;$3,500;35%\n"
        df = read_csv_any(text.encode("utf-8"))
        self.assertEqual(list(df.columns), ["a", "b", "y"])
        P = prepare(df, "y", {"log_target": False})
        self.assertAlmostEqual(P.X["b"].iloc[2], 3500)
        self.assertAlmostEqual(P.y_raw[1], 0.2)

    def test_headerless(self):
        df = read_csv_any(b"1,2,3\n4,5,6\n7,8,10\n")
        self.assertEqual(list(df.columns), ["col_0", "col_1", "col_2"])
        self.assertEqual(len(df), 3)

    def test_big5(self):
        df = read_csv_any("房價,坪數\n100,30\n200,55\n".encode("cp950"))
        self.assertEqual(list(df.columns), ["房價", "坪數"])


class TestPrep(unittest.TestCase):
    def test_profile_and_target(self):
        prof = {p["name"]: p["kind"] for p in profile_columns(synthetic())}
        self.assertEqual(prof["id"], "id")
        self.assertEqual(prof["city"], "categorical")
        self.assertEqual(prof["when"], "date")
        self.assertEqual(suggest_target(profile_columns(synthetic())), "y")

    def test_dummies_use_most_frequent_baseline(self):
        P = prepare(synthetic(), "y")
        self.assertIn("city_B", P.features)
        self.assertIn("city_C", P.features)
        self.assertNotIn("city_A", P.features)
        self.assertNotIn("id", P.features)
        self.assertIn("when_year", P.features)


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = run_analysis(synthetic(), "y", model={"reference_model": False})

    def test_recovers_true_features(self):
        feats = set(self.r["main"]["features"])
        self.assertTrue({"x1", "x2", "city_C"} <= feats, feats)
        self.assertNotIn("noise", feats)
        coef = {c["term"]: c["coef"] for c in self.r["main"]["coef"]}
        self.assertAlmostEqual(coef["x1"], 3, delta=0.15)
        self.assertAlmostEqual(coef["x2"], -2, delta=0.15)

    def test_metrics_and_intervals(self):
        main = next(m for m in self.r["models"] if m["model"] == self.r["main"]["name"])
        self.assertGreater(main["test_r2"], 0.95)
        self.assertGreater(main["pi_coverage"], 0.88)
        self.assertEqual(len(self.r["selection"]["methods"]), 5)

    def test_report_is_strict_json(self):
        json.dumps(self.r, allow_nan=False)


@unittest.skipUnless(os.path.exists(os.path.join(ROOT, "data", "kc_house_data.csv")), "需要 KC 資料")
class TestHW2Reproduction(unittest.TestCase):
    """設定檔 configs/kc_house.json 應重現 HW2 原始作業的結果"""

    def test_kc_house(self):
        with open(os.path.join(ROOT, "configs", "kc_house.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["model"]["reference_model"] = False
        r = run_config(cfg, base_dir=ROOT)
        main = next(m for m in r["models"] if m["model"] == r["main"]["name"])
        self.assertEqual(r["main"]["k"], 11)
        self.assertAlmostEqual(main["test_r2"], 0.7595, places=3)
        self.assertAlmostEqual(main["pi_coverage"], 0.944, places=3)
        self.assertAlmostEqual(r["main"]["f_stat"], 5317, delta=1)


if __name__ == "__main__":
    unittest.main()
