"""Small independent checks of the measurement contract."""
import unittest
import numpy as np
from segmentation import threshold_mask, measure_regions


class SegmentationTests(unittest.TestCase):
    def test_known_rectangle_has_expected_area_and_centroid(self):
        image = np.zeros((12, 12), dtype=np.uint8)
        image[2:5, 3:7] = 80
        mask, used = threshold_mask(image, 60)
        _, regions = measure_regions(mask)
        self.assertEqual(used, 60)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["area_px"], 12)
        self.assertEqual(regions[0]["x_px"], 4.5)
        self.assertEqual(regions[0]["y_px"], 3)
        self.assertFalse(regions[0]["touches_image_edge"])

    def test_pixels_equal_to_threshold_are_background(self):
        mask, _ = threshold_mask(np.full((4, 4), 80, dtype=np.uint8), 80)
        self.assertFalse(mask.any())

    def test_diagonal_pixels_are_connected_and_edge_is_reported(self):
        mask = np.zeros((6, 6), dtype=bool)
        mask[0, 0] = mask[1, 1] = True
        _, regions = measure_regions(mask)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["area_px"], 2)
        self.assertTrue(regions[0]["touches_image_edge"])

    def test_blank_image_has_no_foreground_with_otsu(self):
        mask, _ = threshold_mask(np.zeros((8, 8), dtype=np.uint8))
        _, regions = measure_regions(mask)
        self.assertEqual(regions, [])

    def test_invalid_inputs_are_rejected(self):
        for image in (np.array([]), np.array([[np.nan]])):
            with self.assertRaises(ValueError):
                threshold_mask(image, 30)
        with self.assertRaises(ValueError):
            measure_regions(np.zeros((4, 4), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
