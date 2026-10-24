"""Threshold images and measure connected foreground regions.

Positions use contour centroids when possible.
Area remains the number of foreground pixels.
"""

import cv2
import numpy as np
from skimage import filters, measure, morphology

def threshold_mask(image, threshold=None, closing_radius_px=0):
    """Create a foreground mask and return the threshold used."""

    # Require a non-empty grayscale image.
    if image.ndim != 2:
        raise ValueError("Expected a two-dimensional image")

    if image.size == 0:
        raise ValueError("Expected a non-empty image")

    if not np.isfinite(image).all():
        raise ValueError("Image contains non-finite values")

    # Closing radius must be a non-negative integer.
    if not isinstance(closing_radius_px, (int, np.integer)):
        raise ValueError("Closing radius must be an integer")

    if closing_radius_px < 0:
        raise ValueError("Closing radius must be non-negative")

    # Use Otsu only when no threshold was supplied.
    if threshold is None:
        value = filters.threshold_otsu(image)
    else:
        value = threshold

    value = float(value)

    if not np.isfinite(value):
        raise ValueError("Threshold must be finite")

    # Pixels above the threshold become foreground.
    mask = image > value

    # Optionally close small gaps in the mask.
    if closing_radius_px > 0:
        footprint = morphology.disk(closing_radius_px)
        mask = morphology.binary_closing(mask, footprint=footprint)

    return mask, value


def contour_center(region_mask):
    """Find the centroid of the area enclosed by outer contours."""

    # Convert the boolean mask to an OpenCV-compatible image.
    image = region_mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    area = 0.0
    x_total = 0.0
    y_total = 0.0

    # Combine the moments of all outer contours in this region.
    for contour in contours:
        moments = cv2.moments(contour)      #calculates geometric measurements of the area enclosed by a contour.

        area = area + moments["m00"]
        x_total = x_total + moments["m10"]
        y_total = y_total + moments["m01"]

    # Thin lines and single pixels may enclose zero area.
    if area == 0:
        return None

    x = x_total / area
    y = y_total / area

    return x, y


def measure_regions(mask):
    """Find connected foreground regions and measure their positions and areas.

    
    Area filtering is applied later in detection.py.
    """

    if mask.ndim != 2:
        raise ValueError("Expected a two-dimensional mask")

    if mask.dtype != np.bool_:
        raise ValueError("Expected a boolean mask")

    # Pixels sharing an edge belong to the same object.
    labels = measure.label(mask, connectivity=1)

    rows = []
    height, width = mask.shape

    for region in measure.regionprops(labels):
        # Create a mask containing only this region.
        region_mask = labels == region.label

        # Try the contour centroid first.
        center = contour_center(region_mask)

        if center is None:
            # Use the pixel centroid when contour area is zero.
            y, x = region.centroid
            position_method = "pixel_centroid_fallback"
        else:
            x, y = center
            position_method = "contour_centroid"

        # Check whether the region touches the image border.
        y0, x0, y1, x1 = region.bbox

        touches_edge = False

        if y0 == 0 or x0 == 0:
            touches_edge = True

        if y1 == height or x1 == width:
            touches_edge = True

        # Store the region's measurements in a dictionary.
        row = {}
        row["label"] = int(region.label)
        row["x_px"] = float(x)
        row["y_px"] = float(y)
        row["area_px"] = int(region.area)
        row["touches_image_edge"] = touches_edge
        row["position_method"] = position_method

        rows.append(row)

    return labels, rows
