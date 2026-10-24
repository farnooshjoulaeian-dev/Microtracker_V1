import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from detection import DetectionSettings
from linking import link_nearest, select_region_at_click, predict_position
from trajectory import Trajectory


def region(label=1, x=10., y=12., area=16, accepted=True):
    return {"label": label, "x_px": x, "y_px": y, "area_px": area,
            "accepted_for_auto": accepted, "area_status": "accepted" if accepted else "above_max_area",
            "touches_image_edge": False}


class LinkingTests(unittest.TestCase):
    def test_nearest_accepted_candidate_and_inclusive_radius(self):
        result = link_nearest([region(1, x=1, accepted=False), region(2, x=20)], (0, 12), 20)
        self.assertEqual(result.selected["label"], 2)
        self.assertEqual(result.selected["distance_px"], 20)

    def test_stop_reasons(self):
        self.assertEqual(link_nearest([], (0, 12)).reason, "no_segmented_regions")
        self.assertEqual(link_nearest([region(x=1, accepted=False)], (0, 12)).reason,
                         "nearby_regions_rejected_by_area")
        self.assertEqual(link_nearest([region(x=50)], (0, 12)).reason,
                         "no_accepted_candidate_within_radius")

    def test_multiple_candidates_choose_nearest_without_prediction(self):
        result = link_nearest([region(1, x=5), region(2, x=2)], (0, 12))
        self.assertEqual(result.selected["label"], 2)

    def test_single_nearby_candidate_must_pass_area_change(self):
        result = link_nearest([region(x=1, area=60)], (0, 12),
                              previous_area_px=40, max_area_change=0.30)
        self.assertIsNone(result.selected)
        self.assertEqual(result.reason, "area_change_too_large")

    def test_area_match_wins_only_inside_radius_and_area_limits(self):
        candidates = [region(1, x=1, area=48), region(2, x=5, area=42),
                      region(3, x=30, area=40),
                      region(4, x=2, area=40, accepted=False)]
        result = link_nearest(candidates, (0, 12), 20,
                              previous_area_px=40, max_area_change=0.30)
        self.assertEqual(result.selected["label"], 2)

    def test_area_tolerance_endpoints_and_distance_tie_break(self):
        for area in (28, 52):
            result = link_nearest([region(x=1, area=area)], (0, 12),
                                  previous_area_px=40, max_area_change=0.30)
            self.assertIsNotNone(result.selected)
        result = link_nearest([region(1, x=5, area=38), region(2, x=2, area=42)],
                              (0, 12), previous_area_px=40)
        self.assertEqual(result.selected["label"], 2)

    def test_prediction_uses_frame_intervals_and_has_a_gap_limit(self):
        first = {"frame": 0, "x_px": 2.0, "y_px": 4.0}
        last = {"frame": 2, "x_px": 6.0, "y_px": 2.0}
        self.assertEqual(predict_position(first, last, 3), (8.0, 1.0))
        self.assertEqual(predict_position(first, last, 5), (12.0, -1.0))
        self.assertIsNone(predict_position(first, last, 6))
        self.assertIsNone(predict_position(None, last, 3))
        self.assertIsNone(predict_position(first, last, 2))

    def test_manual_selection_bypasses_area_rejection_and_checks_bounds(self):
        labels = np.zeros((20, 20), dtype=int)
        labels[10:14, 8:12] = 1
        rejected = region(x=9.5, y=11.5, area=16, accepted=False)
        self.assertEqual(select_region_at_click(labels, [rejected], 9, 11), rejected)
        self.assertEqual(select_region_at_click(labels, [rejected], 6, 11), rejected)
        for x, y in [(-1, 11), (20, 11), (9, 20), (0, 0)]:
            self.assertIsNone(select_region_at_click(labels, [rejected], x, y))


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.traj = Trajectory()
        self.settings = DetectionSettings()

    def add(self, frame, source="auto", **kwargs):
        return self.traj.record(frame, region(), source, self.settings, 20, **kwargs)

    def test_frame_replacement_and_removal_of_future_are_undoable(self):
        self.add(0, "manual"); self.add(1); self.add(2)
        self.add(1, "manual", truncate_future=True)
        self.assertEqual([p["frame"] for p in self.traj.points], [0, 1])
        self.assertEqual(self.traj.get(1)["source"], "manual")
        self.assertTrue(self.traj.undo())
        self.assertEqual([p["frame"] for p in self.traj.points], [0, 1, 2])
        self.assertEqual(self.traj.get(1)["source"], "auto")

    def test_keep_future_and_duplicate_auto_replacement(self):
        self.add(0, "manual"); self.add(2); self.add(0, "manual")
        self.add(2); self.add(2)
        self.assertEqual([p["frame"] for p in self.traj.points], [0, 2])

    def test_clear_is_undoable(self):
        self.add(0, "manual"); self.add(1)
        self.traj.clear()
        self.assertEqual(self.traj.points, [])
        self.assertTrue(self.traj.undo())
        self.assertEqual(len(self.traj.points), 2)

    def test_estimated_auto_position_has_no_fabricated_area_or_label(self):
        point = region()
        point.update(area_px=None, label=None, area_status="not_measured")
        with self.assertRaises(ValueError):
            self.traj.record(1, point, "auto", self.settings, 20)
        stored = self.traj.record(1, point, "auto", self.settings, 20, predicted=True)
        self.assertEqual(stored["source"], "auto")
        self.assertIsNone(stored["area_px"])
        self.assertIsNone(stored["label"])
        self.assertIsNone(stored["position_method"])

    def test_csv_records_point_settings_and_json_metadata(self):
        self.add(0, "manual", time_s=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"track.csv"
            companion = self.traj.save(path, {"video": "example.mp4", "spatial_calibration": None})
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["source"], "manual")
            self.assertEqual(float(rows[0]["threshold_gray"]), 40)
            self.assertEqual(json.loads(companion.read_text())["video"], "example.mp4")


if __name__ == "__main__":
    unittest.main()
