"""Detect and measure fluorescent objects in a single image.

Converts the image to grayscale, applies Gaussian smoothing and a threshold,
then measures each region's position and area. Marks regions as accepted
or rejected for AUTO tracking based on their area.

Does not load videos, control the GUI, link objects between frames,
or convert pixel measurements to physical units.
"""

from dataclasses import dataclass  # Classes for settings and results

import cv2                        
import numpy as np                
from skimage.filters import gaussian  
from segmentation import threshold_mask, measure_regions  

@dataclass(frozen=True)  # Prevent changing settings after creation
class DetectionSettings:
    blur_sigma_px: float = 3.0    # Gaussian smoothing scale in pixels
    threshold_gray: float = 40.0  # Foreground intensity cutoff, on a 0–255 scale
    closing_radius_px: int = 0    # Morphological closing radius; 0 disables it
    min_area_px: int = 6          # Minimum AUTO region area, in pixels²
    max_area_px: int = 150        # Maximum AUTO region area, in pixels²
    vmin_gray: float = 0.0        # Intensity mapped to black before thresholding
    vmax_gray: float = 255.0      # Intensity mapped to white before thresholding

            
    def __post_init__(self):
        """Check that the detection parameters have valid values."""
        if not np.isfinite(self.blur_sigma_px) or self.blur_sigma_px < 0:
            raise ValueError("Blur sigma must be finite and non-negative")

        # Keep the threshold within the image intensity range.
        if not np.isfinite(self.threshold_gray) or not 0 <= self.threshold_gray <= 255:
            raise ValueError("Threshold must be between 0 and 255")

        # Ordered intensity limits keep contrast stretching well-defined.
        if not (
            np.isfinite(self.vmin_gray)
            and np.isfinite(self.vmax_gray)
            and 0 <= self.vmin_gray < self.vmax_gray <= 255
        ):
            raise ValueError("Intensity limits must satisfy 0 <= vmin < vmax <= 255")

        # Closing requires a non-negative integer radius.
        if not isinstance(self.closing_radius_px, int) or self.closing_radius_px < 0:
            raise ValueError("Closing radius must be a non-negative integer")

        # Region areas are integer foreground pixel counts.
        if not all(
            isinstance(value, int)
            for value in (self.min_area_px, self.max_area_px)
        ):
            raise ValueError("Area limits must be integer pixel counts")

        # Require positive, ordered area limits.
        if not 1 <= self.min_area_px <= self.max_area_px:
            raise ValueError("Area limits must satisfy 1 <= minimum <= maximum")


@dataclass  # Create a data container for processed images and region measurements.
class DetectionResult:
    gray: np.ndarray       # Original grayscale image
    smoothed: np.ndarray   # Floating-point smoothed image
    mask: np.ndarray       # Binary foreground mask
    labels: np.ndarray     # Region labels; 0 represents background
    regions: list          # Measurements and AUTO eligibility for every region
    


def detect_frame(frame, settings=DetectionSettings()):
    """Detect regions in a uint8 grayscale or BGR image.

    Coordinates refer to the supplied image. Area-rejected regions remain
    available for manual selection. Labels are specific to this frame.
    """
    
    # Reject empty images, incorrect types and unsupported intensity formats.
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.size == 0
    ):
        raise ValueError("Expected a non-empty uint8 image")

    # Use grayscale input directly, or convert OpenCV BGR input.
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Expected grayscale or three-channel BGR")

    # Smooth before thresholding to reduce small intensity fluctuations.
    smoothed = gaussian(
        gray,                          # Input grayscale image
        sigma=settings.blur_sigma_px,  # Smoothing scale in pixels
        preserve_range=True,           # Preserve the 0–255 intensity scale
        mode="mirror",                 # Reflect at image boundaries
        truncate=3.0,                  # Extend to ±3 sigma; default is 4
    )
    # Clip and rescale to 0–255 for thresholding; preserve the smoothed image.
    clipped = np.clip(smoothed, settings.vmin_gray, settings.vmax_gray)
    stretched = (
        255.0 * (clipped - settings.vmin_gray)
        / (settings.vmax_gray - settings.vmin_gray)
    ).astype(np.uint8)

    # Create the foreground mask, with optional morphological closing.
    mask, _ = threshold_mask(
        stretched,
        settings.threshold_gray,
        settings.closing_radius_px, #This can fill small holes and connect narrow gaps.
    )

    # Measure each connected foreground object without splitting it.
    labels, regions = measure_regions(mask)

    # Mark AUTO eligibility without deleting any segmented region.
    for region in regions:
        area = region["area_px"]
        reason = (
            "below_min_area" if area < settings.min_area_px else
            "above_max_area" if area > settings.max_area_px else
            "accepted"
        )
        region["area_status"] = reason  # Store the area-filter result.
        region["accepted_for_auto"] = reason == "accepted"  # True if area is within limits.
        
    # Return intermediate images alongside all region measurements.
    return DetectionResult(gray, smoothed, mask, labels, regions)