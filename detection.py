"""Initial fluorescence detector: explicit settings, measurements and rejections.

No video access, Qt, linking, motion prediction, or physical calibration here.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from segmentation import threshold_mask, measure_regions


@dataclass(frozen=True)
class DetectionSettings:
    # Match the PI's default blur recipe; kernel width and sigma are independent.
    blur_kernel_px: int = 3
    blur_sigma_px: float = 3.0
    threshold_gray: float = 40.0
    closing_radius_px: int = 0
    min_area_px: int = 6
    max_area_px: int = 150

    def __post_init__(self):
        if not isinstance(self.blur_kernel_px, int) or self.blur_kernel_px < 1 or self.blur_kernel_px % 2 != 1:
            raise ValueError("Blur kernel must be a positive odd integer")
        if not np.isfinite(self.blur_sigma_px) or self.blur_sigma_px <= 0:
            raise ValueError("Blur sigma must be finite and positive")
        if not np.isfinite(self.threshold_gray) or not 0 <= self.threshold_gray <= 255:
            raise ValueError("Threshold must be between 0 and 255")
        if not isinstance(self.closing_radius_px, int) or self.closing_radius_px < 0:
            raise ValueError("Closing radius must be a non-negative integer")
        if not all(isinstance(value, int) for value in (self.min_area_px, self.max_area_px)):
            raise ValueError("Area limits must be integer pixel counts")
        if not 1 <= self.min_area_px <= self.max_area_px:
            raise ValueError("Area limits must satisfy 1 <= minimum <= maximum")


@dataclass
class DetectionResult:
    gray: np.ndarray
    smoothed: np.ndarray
    mask: np.ndarray
    labels: np.ndarray
    regions: list


def detect_frame(frame, settings=DetectionSettings()):
    """Accept uint8 grayscale or OpenCV BGR; return all regions with area status.

    x/y refer to the supplied image. All labels remain available for manual
    selection, including regions rejected by the inclusive AUTO area interval.
    Region labels identify objects within one frame, not across frames.
    """
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.size == 0:
        raise ValueError("Expected a non-empty uint8 image")
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Expected grayscale or three-channel BGR")
    k = settings.blur_kernel_px
    smoothed = cv2.GaussianBlur(gray, (k, k), settings.blur_sigma_px,
                              borderType=cv2.BORDER_REFLECT_101)
    mask, _ = threshold_mask(smoothed, settings.threshold_gray, settings.closing_radius_px)
    labels, regions = measure_regions(mask)
    for region in regions:
        area = region["area_px"]
        reason = ("below_min_area" if area < settings.min_area_px else
                  "above_max_area" if area > settings.max_area_px else "accepted")
        region["area_status"] = reason
        region["accepted_for_auto"] = reason == "accepted"
    return DetectionResult(gray, smoothed, mask, labels, regions)
