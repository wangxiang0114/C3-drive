import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from c3drive import paper_eval


class PaperEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.ids = ["RouteScenario_11_rep0", "RouteScenario_12_rep0"]
        self.scores = {"score_composed": 48.0, "score_route": 60.0, "score_penalty": 0.8}
        scores = [self.scores, {"score_composed": 52.0, "score_route": 80.0, "score_penalty": 0.65}]
        self.report = {
            "entry_status": "Finished",
            "eligible": True,
            "_checkpoint": {
                "progress": [2, 2],
                "records": [{"route_id": route, "index": index, "scores": scores[index]}
                            for index, route in enumerate(self.ids)],
                "global_record": {"scores_mean": {"score_composed": 50, "score_route": 70, "score_penalty": 0.725}},
            },
        }

    def test_aggregate_matches_complete_report(self):
        scores = paper_eval.audit_report(self.report, self.ids)
        self.assertEqual(scores["score_composed"], 50)
        self.assertEqual(scores["score_penalty"], 0.725)

    def test_rejects_partial_proxy_and_duplicate_routes(self):
        partial = copy.deepcopy(self.report)
        partial["_checkpoint"]["progress"][0] = 1
        duplicate = copy.deepcopy(self.report)
        duplicate["_checkpoint"]["records"][1]["route_id"] = self.ids[0]
        for report in [partial, duplicate, {"results": {"clean": {"score": 50}}}]:
            with self.subTest(report=report), self.assertRaises(ValueError):
                paper_eval.audit_report(report, self.ids)

    def test_rejects_out_of_bounds_and_inconsistent_scores(self):
        self.report["_checkpoint"]["global_record"]["scores_mean"]["score_composed"] = 51
        with self.assertRaisesRegex(ValueError, "disagrees"):
            paper_eval.audit_report(self.report, self.ids)
        self.scores["score_penalty"] = float("nan")
        with self.assertRaisesRegex(ValueError, "between"):
            paper_eval.audit_report(self.report, self.ids)

    def test_fingerprint_and_report_duplicates(self):
        with TemporaryDirectory() as directory:
            xml = Path(directory) / "routes.xml"
            xml.write_bytes(b'<routes><route id="11" /><route id="12" /></routes>')
            with self.assertRaisesRegex(ValueError, "does not match"):
                paper_eval.route_ids(xml)
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(self.report), encoding="utf-8")
            with patch.object(paper_eval, "ROUTE_SHA256", paper_eval.digest(xml.read_bytes())), \
                    patch.object(paper_eval, "ROUTE_COUNT", 2):
                self.assertEqual(paper_eval.audit_files(xml, [path], 1)["unique_routes"], 2)
                with self.assertRaisesRegex(ValueError, "duplicates"):
                    paper_eval.audit_files(xml, [path, path], 1)
