"""Small segmentation primitives, independent of video I/O and the GUI.

Thresholds are in the units of the supplied image. Masks contain all foreground
regions; acceptance by size belongs in a later explicit processing step.
"""
import numpy as np
from skimage import filters, measure, morphology


def threshold_mask(image, threshold=None, closing_radius_px=0):
    """Return (mask, threshold used); None selects whole-image Otsu."""
    if image.ndim != 2 or image.size == 0:
        raise ValueError("Expected a non-empty two-dimensional image")
    if not np.isfinite(image).all():
        raise ValueError("Image contains non-finite values")
    if not isinstance(closing_radius_px, (int, np.integer)) or closing_radius_px < 0:
        raise ValueError("Closing radius must be a non-negative integer")
    value = float(filters.threshold_otsu(image) if threshold is None else threshold)
    if not np.isfinite(value):
        raise ValueError("Threshold must be finite")
    mask = image > value
    if closing_radius_px:
        mask = morphology.binary_closing(mask, footprint=morphology.disk(closing_radius_px))
    return mask, value


def measure_regions(mask):
    """Measure 8-connected foreground regions without rejecting by size.

    Centroids are floating point pixel coordinates; area is foreground pixel count.
    Coordinates refer to the supplied image, not an implicit crop offset.
    """
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("Expected a two-dimensional boolean mask")
    labels = measure.label(mask, connectivity=2)
    rows = []
    h, w = mask.shape
    for region in measure.regionprops(labels):
        y, x = region.centroid
        y0, x0, y1, x1 = region.bbox
        rows.append({"label": int(region.label), "x_px": float(x), "y_px": float(y),
                     "area_px": int(region.area),
                     "touches_image_edge": y0 == 0 or x0 == 0 or y1 == h or x1 == w})
    return labels, rows
