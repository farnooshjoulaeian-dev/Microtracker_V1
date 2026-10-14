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

    def test_diagonal_pixels_are_separate_and_edge_is_reported(self):
        mask = np.zeros((6, 6), dtype=bool)
        mask[0, 0] = True
        mask[1, 1] = True

        labels, regions = measure_regions(mask)

        # Diagonal contact does not connect pixels with connectivity=1.
        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0]["area_px"], 1)
        self.assertEqual(regions[1]["area_px"], 1)

        # Only the first pixel touches the image border.
        self.assertTrue(regions[0]["touches_image_edge"])
        self.assertFalse(regions[1]["touches_image_edge"])

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


    def test_contour_center_uses_outer_boundary(self):
        mask = np.zeros((9, 9), dtype=bool)

        # A square with one off-center background pixel inside it.
        mask[2:7, 2:7] = True
        mask[3, 3] = False

        labels, regions = measure_regions(mask)
        region = regions[0]

        # The outer square's center is unchanged by the internal hole.
        self.assertAlmostEqual(region["x_px"], 4.0)
        self.assertAlmostEqual(region["y_px"], 4.0)
        self.assertEqual(region["position_method"], "contour_centroid")

        # Area still counts foreground pixels, excluding the hole.
        self.assertEqual(region["area_px"], 24)


    def test_single_pixel_uses_pixel_centroid_fallback(self):
        mask = np.zeros((6, 6), dtype=bool)
        mask[2, 3] = True

        labels, regions = measure_regions(mask)
        region = regions[0]

        # One pixel has no enclosed contour area.
        self.assertEqual(region["x_px"], 3.0)
        self.assertEqual(region["y_px"], 2.0)
        self.assertEqual(region["area_px"], 1)
        self.assertEqual(
            region["position_method"],
            "pixel_centroid_fallback",
        )

if __name__ == "__main__":

    unittest.main()