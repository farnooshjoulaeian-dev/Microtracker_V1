"""Tests for detection measurements, area filtering and input validation."""

import unittest

import cv2
import numpy as np

from detection import DetectionSettings, detect_frame


class DetectionTests(unittest.TestCase):
    def test_size_rejection_preserves_labels_and_measurements(self):
        # Three regions with known areas: 4, 12 and 25 pixels².
        frame = np.zeros((30, 30), dtype=np.uint8)
        frame[3:5, 3:5] = 100
        frame[10:13, 10:14] = 100
        frame[20:25, 20:25] = 100

        settings = DetectionSettings(
            blur_sigma_px=0,  # Disable smoothing to preserve exact geometry.
            min_area_px=6,
            max_area_px=20,
        )
        result = detect_frame(frame, settings)

        # Rejected regions remain measured and labeled.
        self.assertEqual(
            [r["area_px"] for r in result.regions],
            [4, 12, 25],
        )
        self.assertEqual(
            [r["area_status"] for r in result.regions],
            ["below_min_area", "accepted", "above_max_area"],
        )
        self.assertEqual(int(result.labels.max()), 3)

        # Verify the accepted rectangle's centroid.
        self.assertEqual(result.regions[1]["x_px"], 11.5)
        self.assertEqual(result.regions[1]["y_px"], 11)

    def test_area_limits_include_endpoints(self):
        frame = np.zeros((10, 10), dtype=np.uint8)
        frame[3:5, 3:6] = 100  # Area is exactly 6 pixels².

        settings = DetectionSettings(
            blur_sigma_px=0,
            min_area_px=6,
            max_area_px=6,
        )
        result = detect_frame(frame, settings)

        self.assertTrue(result.regions[0]["accepted_for_auto"])

    def test_bgr_and_grayscale_agree_for_gray_content(self):
        # A fixed random seed makes the test reproducible.
        gray = np.random.default_rng(12).integers(
            0, 256, (30, 30), dtype=np.uint8
        )
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        grayscale_result = detect_frame(gray)
        bgr_result = detect_frame(bgr)

        np.testing.assert_array_equal(
            grayscale_result.labels,
            bgr_result.labels,
        )

    def test_blank_frame_has_no_regions(self):
        result = detect_frame(np.zeros((20, 20), dtype=np.uint8))

        self.assertEqual(result.regions, [])
        self.assertFalse(result.mask.any())

    def test_invalid_parameters_and_inputs_are_rejected(self):
        invalid_settings = (
            {"min_area_px": 10, "max_area_px": 6},
            {"threshold_gray": 256},
            {"blur_sigma_px": -1},
            {"closing_radius_px": -1},
        )

        for kwargs in invalid_settings:
            with self.subTest(settings=kwargs):
                with self.assertRaises(ValueError):
                    DetectionSettings(**kwargs)

        invalid_frames = (
            np.zeros((4, 4), dtype=float),
            np.zeros((0, 0), dtype=np.uint8),
            np.zeros((4, 4, 4), dtype=np.uint8),
        )

        for frame in invalid_frames:
            with self.subTest(shape=frame.shape, dtype=str(frame.dtype)):
                with self.assertRaises(ValueError):
                    detect_frame(frame)


if __name__ == "__main__":
    unittest.main()