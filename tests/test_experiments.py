import csv
import json
import sys
import unittest
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import torch

from c3drive.__main__ import command_parser
from c3drive.benchmark import run_benchmark, run_lock
from c3drive.data import SCHEMA, SPLITS
from c3drive.drift_eval import summarize, trace_episode
from c3drive.model import Driver
from c3drive.reporting import score_summary
from c3drive.train import load, save
from c3drive.workflow import train_test


def profile(dimension):
    return {
        "center": torch.tensor([1.0] + [0.0] * (dimension - 1)),
        "precision": torch.eye(dimension),
        "scores": torch.zeros(19),
    }


def write_episode(directory, domain, split, route, samples=3, name="0000.npz"):
    path = directory / domain / split / name
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.zeros((samples, *shape), dtype=dtype) for name, (shape, dtype) in SCHEMA.items()}
    np.savez_compressed(path, **arrays, samples=np.array(samples),
                        metadata=json.dumps({"town": "TestTown", "domain": domain, "route": route}))


class TrainingWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.args = command_parser().parse_args(
            ["train-test", "--output", str(root / "run"), "--data", str(root / "data"), "--limit", "1"]
        )
        for index, split in enumerate(("train", "validation", "test")):
            write_episode(self.args.data, "clean", split, {"start": 2 * index, "end": 2 * index + 1})

    def test_validates_both_checkpoints_and_tests_the_student(self):
        evaluate = Mock()
        with patch("c3drive.train.fit") as fit, patch("c3drive.carla_env.connect") as connect:
            train_test(self.args, evaluate)
        connect.assert_not_called()
        self.assertEqual(fit.call_args_list[1].args[1], self.args.output / "teacher.pt")
        self.assertEqual(
            [(call.args[0].split, call.args[0].checkpoint.name, call.args[0].limit)
             for call in evaluate.call_args_list],
            [("validation", "teacher.pt", None), ("validation", "student.pt", None), ("test", "student.pt", 1)],
        )

    def test_rejects_train_test_route_overlap_before_training(self):
        write_episode(self.args.data, "clean", "test", {"start": 0, "end": 1})
        with patch("c3drive.train.fit") as fit:
            with self.assertRaisesRegex(ValueError, "occurs in both train and test"):
                train_test(self.args, Mock())
        fit.assert_not_called()
        self.assertFalse(self.args.output.exists())


class TraceTests(unittest.TestCase):
    def test_clean_and_shifted_detection(self):
        state = {"profiles": {"clean": profile(2)}, "domain_slots": {"clean": -1},
                 "block_size": 2, "threshold": 2.5}
        clean = trace_episode(np.array([[1.0, 0.0]] * 4), state)
        shift = trace_episode(np.array([[0.0, 1.0]] * 4), state)
        self.assertIsNone(clean["first_alarm_block"])
        self.assertEqual(shift["first_alarm_block"], 2)
        self.assertEqual([x["block"] for x in shift["trace"]], [1, 2, 3, 4])
        metrics = summarize([clean], [shift, clean])
        self.assertEqual(metrics["auroc_peak_evidence"], 0.75)
        self.assertEqual(metrics["shift_detection_rate"], 0.5)
        self.assertEqual(metrics["median_first_alarm_block"], 2)
        self.assertEqual(metrics["clean_false_alarm_rate"], 0.0)

    def test_forgetting_and_backward_transfer_use_different_references(self):
        domains = ["clean", "camera-drop", "occlusion"]
        matrix = [{"clean": 0.8}, {"clean": 0.9, "camera-drop": 0.6},
                  {"clean": 0.75, "camera-drop": 0.65, "occlusion": 0.4}]
        value = score_summary(matrix, domains)
        self.assertAlmostEqual(value["forgetting_by_domain"]["clean"], 0.15)
        self.assertAlmostEqual(value["backward_transfer_by_domain"]["clean"], -0.05)
        self.assertAlmostEqual(value["average_forgetting"], 0.05)
        self.assertAlmostEqual(value["backward_transfer"], 0)
        self.assertIsNone(score_summary(matrix[:1], domains)["average_forgetting"])


class FakeEnv:
    def __init__(self, test_route):
        self.test_route = test_route

    def run(self, route, domain, ticks, traffic, policy):
        assert route == self.test_route
        scores = {"clean": 0.8, "camera-drop": 0.4, "occlusion": 0.3, "gps": 0.2}
        if "camera-drop" in policy.detector.mapping:
            scores.update({"clean": 0.75, "camera-drop": 0.65})
        if "gps" in policy.detector.mapping:
            scores.update({"clean": 0.7, "gps": 0.6})
        return {"route": route, "score": scores[domain], "completion": scores[domain],
                "collisions": 0, "success": False}


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        routes = {split: [{"start": i * 2, "end": i * 2 + 1, "seed": 41}]
                  for i, split in enumerate(SPLITS)}
        self.manifest = {"town": "TestTown", "routes": routes}
        path = self.root / "routes.json"
        path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.data = self.root / "data"
        for domain in ("clean", "camera-drop", "occlusion", "gps"):
            for split in ("train", "validation", "fit", "calibrate", "threshold", "test"):
                write_episode(self.data, domain, split, routes[split][0])
        model = Driver(width=4, dim=8, slots=2)
        self.checkpoint = self.root / "initial.pt"
        save(self.checkpoint, model, stage=0, domain_slots={"clean": -1}, profiles={"clean": profile(8)},
             block_size=2, threshold=2.5, alpha=0.05, memory_budget=16,
             memory={"clean": {"feature": torch.zeros(1, 8)}})
        self.teacher = self.root / "teacher.pt"
        save(self.teacher, Driver(width=4, dim=8, slots=0))
        self.output = self.root / "run"
        self.args = command_parser().parse_args([
            "benchmark", "--checkpoint", str(self.checkpoint), "--teacher", str(self.teacher),
            "--manifest", str(path), "--data", str(self.data), "--output", str(self.output),
            "--domains", "camera-drop", "occlusion", "gps", "--limit", "1", "--device", "cpu",
        ])
        self.env = FakeEnv(routes["test"][0])

    def test_resume_after_interruption_and_reuse_last_accepted_checkpoint(self):
        attempts = []

        def adapt(update, env, manifest):
            model, state = load(update.checkpoint, "cpu")
            attempts.append((update.domain, state["stage"]))
            if update.domain == "occlusion":
                if attempts.count(("occlusion", 1)) == 1:
                    save(update.output, model)
                    raise RuntimeError("interrupted after writing the candidate")
                decision = {"accepted": False, "reason": "failed_gate_or_no_merge"}
            else:
                fields = {key: value for key, value in state.items() if key not in ("spec", "weights")}
                fields["stage"] += 1
                fields["domain_slots"][update.domain] = fields["stage"] - 1
                fields["profiles"][update.domain] = profile(8)
                fields["memory"][update.domain] = {"feature": torch.zeros(2, 8)}
                save(update.output, model, **fields)
                decision = {"accepted": True, "reason": "passed_gate"}
            update.report.write_text(json.dumps(decision), encoding="utf-8")

        connect = lambda *_: self.env
        with patch("c3drive.benchmark.continual", side_effect=adapt):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_benchmark(self.args, self.manifest, connect)
            self.assertFalse((self.output / "stage-02-occlusion").exists())
            partial = json.loads((self.output / "report.json").read_text(encoding="utf-8"))
            self.assertFalse(partial["completed"])
            self.assertEqual(len(partial["stages"]), 2)
            self.args.resume = True
            run_benchmark(self.args, self.manifest, connect)
        self.assertEqual(attempts, [("camera-drop", 0), ("occlusion", 1), ("occlusion", 1), ("gps", 1)])
        self.assertFalse((self.output / "stage-02-occlusion" / "accepted.pt").exists())
        result = json.loads((self.output / "report.json").read_text(encoding="utf-8"))
        self.assertTrue(result["completed"])
        self.assertEqual(result["score_matrix"][0], {"clean": 0.8, "camera-drop": None, "occlusion": None, "gps": None})
        self.assertAlmostEqual(result["stages"][1]["current_domain_gain"], 0.25)
        self.assertEqual(result["stages"][2]["current_domain_gain"], 0)
        self.assertEqual(result["stages"][2]["checkpoint_stage"], 1)
        with (self.output / "scores.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[-1]["gps"], "0.6")
        final = json.loads((self.output / "stage-03-gps/stage.json").read_text(encoding="utf-8"))
        self.assertEqual(final["resources"]["occupied_slots"], 2)
        self.assertEqual(final["resources"]["replay_items_by_domain"]["gps"], 2)
        with patch("c3drive.benchmark.continual", side_effect=AssertionError("adaptation rerun")):
            run_benchmark(self.args, self.manifest, lambda *_: self.fail("complete run called CARLA"))
        self.args.ticks += 1
        with self.assertRaisesRegex(ValueError, "settings changed"):
            run_benchmark(self.args, self.manifest, connect)

    def test_misplaced_recording_is_rejected_before_connecting(self):
        route = self.manifest["routes"]["test"][0]
        write_episode(self.data, "camera-drop", "fit", route)
        with self.assertRaisesRegex(ValueError, "outside the benchmark manifest"):
            run_benchmark(self.args, self.manifest, lambda *_: self.fail("invalid setup called CARLA"))

    def test_same_output_directory_excludes_concurrent_writer(self):
        with run_lock(self.output):
            with self.assertRaisesRegex(RuntimeError, "Another benchmark"):
                with run_lock(self.output):
                    self.fail("second writer acquired the lock")
        with run_lock(self.output):
            self.assertTrue(self.output.is_dir())

    def test_detect_eval_cli_on_recorded_episodes(self):
        write_episode(self.data, "clean", "test", self.env.test_route, samples=1, name="0001.npz")
        report = self.root / "drift.json"
        completed = run([
            sys.executable, "-m", "c3drive", "detect-eval", "--checkpoint", str(self.checkpoint),
            "--data", str(self.data), "--domains", "camera-drop", "--output", str(report),
            "--device", "cpu",
        ], capture_output=True, text=True, timeout=90)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(result["comparisons"]["camera-drop"]["auroc_peak_evidence"], 0.5)
        self.assertEqual(result["episodes"]["clean"][0]["blocks"], 1)
        self.assertEqual(result["skipped_incomplete_episodes"]["clean"], ["clean/test/0001.npz"])
