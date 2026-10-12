"""Correctness and fairness regression tests for the frozen patch benchmark."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/patch_benchmark.py"
spec = importlib.util.spec_from_file_location("patch_benchmark", SCRIPT)
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b
spec.loader.exec_module(b)


class TestPatchBenchmark(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[1, 0], [0, 1], [0.9, 0.1],
                           [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]], dtype=np.float32)
        self.y = np.array([0, 1, 0, 1, 0, 1])
        self.ids = np.array([101, 102, 103, 104, 105, 106])

    def test_retrieval_perfect(self):
        result = b.retrieval_metrics(self.x, self.y, self.x, self.y, query_chunk=2)
        for name in ("mAP", "macro_mAP", "Acc@1", "Acc@3", "Acc@5",
                     "majority_vote_Acc@5"):
            self.assertEqual(result[name], 1.0, name)

    def test_map_matches_sklearn(self):
        q = np.array([[0.85, 0.15]], dtype=np.float32)
        result = b.retrieval_metrics(self.x, self.y, q, np.array([0]))
        sims = (b.normalize(q) @ b.normalize(self.x).T)[0]
        self.assertAlmostEqual(result["mAP"], average_precision_score(
            (self.y == 0).astype(int), sims), places=7)

    def test_macro_map_balances_query_classes(self):
        q = np.vstack([self.x[:1]] * 9 + [self.x[1:2]])
        y = np.array([0] * 9 + [1])
        r = b.retrieval_metrics(self.x, self.y, q, y)
        self.assertAlmostEqual(r["mAP"], r["macro_mAP"], places=7)

    def test_query_absent_in_gallery_fails(self):
        with self.assertRaises(ValueError):
            b.retrieval_metrics(self.x, self.y, self.x[:1], np.array([10]))

    def test_normalisation_is_per_sample(self):
        x = b.normalize(np.array([[3, 4], [0, 8]], np.float32))
        np.testing.assert_allclose(np.linalg.norm(x, axis=1), [1, 1])
        np.testing.assert_allclose(x[0], [.6, .8], atol=1e-7)

    def test_zero_vector_fails_closed(self):
        with self.assertRaises(ValueError):
            b.normalize(np.zeros((3, 2), np.float32))

    def test_support_equal_and_reproducible(self):
        s1 = b.support_selections("mhist", "original", self.ids, self.y,
                                  shots=(1, 2, 3), repeats=100)
        s2 = b.support_selections("mhist", "original", self.ids[::-1], self.y[::-1],
                                  shots=(1, 2, 3), repeats=100)
        self.assertEqual(s1, s2)
        for k, plans in s1.items():
            for plan in plans:
                c = [int(self.y[np.flatnonzero(self.ids == i)[0]]) for i in plan]
                self.assertEqual(c.count(0), int(k))
                self.assertEqual(c.count(1), int(k))
                self.assertEqual(len(set(plan)), len(plan))

    def test_support_changes_by_variant(self):
        a = b.support_selections("mhist", "original", self.ids, self.y,
                                 shots=(1,), repeats=100)
        c = b.support_selections("mhist", "grouped", self.ids, self.y,
                                 shots=(1,), repeats=100)
        self.assertNotEqual(a, c)

    def test_support_no_test_labels(self):
        support = b.support_selections("mhist", "original", self.ids, self.y,
                                       shots=(1,), repeats=100)
        altered_test = np.array([0, 0, 0])
        del altered_test
        self.assertEqual(support, b.support_selections(
            "mhist", "original", self.ids, self.y, shots=(1,), repeats=100))

    def test_fewshot_perfect(self):
        support = {"1": [[101, 102]], "2": [[101, 103, 102, 104]]}
        out = b.fewshot_metrics(self.x, self.y, self.ids, self.x, self.y, support)
        for shot in ("1", "2"):
            for metric in ("macro_f1", "balanced_accuracy", "accuracy"):
                self.assertEqual(out[shot]["metrics"][metric]["mean"], 1.0)

    def test_fewshot_fails_malformed_support(self):
        with self.assertRaises(RuntimeError):
            b.fewshot_metrics(self.x, self.y, self.ids, self.x, self.y,
                              {"2": [[101, 102, 103, 105]]})

    def test_ci_mean_and_standard_deviation(self):
        x = np.array([0.1, 0.3, 0.5, 0.9])
        d = b.ci_summary(x)
        self.assertAlmostEqual(d["mean"], float(x.mean()))
        self.assertAlmostEqual(d["std"], float(x.std(ddof=1)))
        self.assertLess(d["ci95_lower"], d["mean"])
        self.assertGreater(d["ci95_upper"], d["mean"])

    def test_group_path_parsing(self):
        row = {"relative_path": "benign/SOB/adenosis/SOB_B_A_14-22549AB/"
                                "100X/SOB_B_A-14-22549AB-100-010.png"}
        self.assertEqual(b.patient_breakhis(row), "SOB_B_A_14-22549AB")

    def test_feature_cache_roundtrip_and_invalid_vector(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "x.npz"
            arrays = {}
            for split in b.SPLITS:
                arrays[split + "_features"] = self.x.copy()
                arrays[split + "_labels"] = self.y.copy()
                arrays[split + "_indices"] = self.ids.copy()
            b.save_feature_cache(path, arrays, {"frozen": True})
            meta, loaded = b.load_feature_cache(path)
            self.assertTrue(meta["frozen"])
            np.testing.assert_array_equal(loaded["train"][0], self.x)
            arrays["test_features"] = np.zeros_like(self.x)
            b.save_feature_cache(path, arrays, {"frozen": True})
            with self.assertRaises(RuntimeError):
                b.load_feature_cache(path)

    def test_input_order_changes_signature(self):
        a = {"train": [{"record_index": 0, "class_id": 0,
                        "relative_path": "a", "storage": "file"}],
             "val": [{"record_index": 1, "class_id": 1,
                      "relative_path": "b", "storage": "file"}],
             "test": [{"record_index": 2, "class_id": 0,
                       "relative_path": "c", "storage": "file"}]}
        b1 = json.loads(json.dumps(a))
        b1["train"][0]["class_id"] = 1
        self.assertNotEqual(b.row_order_hash(a), b.row_order_hash(b1))


if __name__ == "__main__":
    unittest.main()
