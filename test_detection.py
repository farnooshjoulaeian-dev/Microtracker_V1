import unittest
import cv2
import numpy as np
from detection import DetectionSettings, detect_frame


class DetectionTests(unittest.TestCase):
    def test_size_rejection_preserves_labels_and_measurements(self):
        frame = np.zeros((30, 30), dtype=np.uint8)
        frame[3:5, 3:5] = 100
        frame[10:13, 10:14] = 100
        frame[20:25, 20:25] = 100
        settings = DetectionSettings(blur_kernel_px=1, min_area_px=6, max_area_px=20)
        result = detect_frame(frame, settings)
        self.assertEqual([r["area_px"] for r in result.regions], [4, 12, 25])
        self.assertEqual([r["area_status"] for r in result.regions],
                         ["below_min_area", "accepted", "above_max_area"])
        self.assertEqual(int(result.labels.max()), 3)
        self.assertEqual(result.regions[1]["x_px"], 11.5)
        self.assertEqual(result.regions[1]["y_px"], 11)

    def test_area_limits_include_endpoints(self):
        frame = np.zeros((10, 10), dtype=np.uint8)
        frame[3:5, 3:6] = 100
        result = detect_frame(frame, DetectionSettings(blur_kernel_px=1, min_area_px=6, max_area_px=6))
        self.assertTrue(result.regions[0]["accepted_for_auto"])

    def test_bgr_and_grayscale_agree_for_gray_content(self):
        gray = np.random.default_rng(12).integers(0, 256, (30, 30), dtype=np.uint8)
        a = detect_frame(gray)
        b = detect_frame(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        np.testing.assert_array_equal(a.labels, b.labels)

    def test_blank_frame_has_no_regions(self):
        result = detect_frame(np.zeros((20, 20), dtype=np.uint8))
        self.assertEqual(result.regions, [])
        self.assertFalse(result.mask.any())

    def test_invalid_parameters_and_inputs_are_rejected(self):
        for kwargs in ({"min_area_px": 10, "max_area_px": 6},
                       {"blur_kernel_px": 2}, {"threshold_gray": 256},
                       {"blur_sigma_px": 0}, {"closing_radius_px": -1}):
            with self.assertRaises(ValueError):
                DetectionSettings(**kwargs)
        for frame in (np.zeros((4, 4), dtype=float), np.zeros((0, 0), dtype=np.uint8),
                      np.zeros((4, 4, 4), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                detect_frame(frame)


if __name__ == "__main__":
    unittest.main()
