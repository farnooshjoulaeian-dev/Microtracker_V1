"""Choose AUTO detections by distance.

Candidate matching uses area and distance. A helper predicts short gaps.
The older segmented-click helper is retained for existing tests.
"""

from dataclasses import dataclass
from math import hypot, isfinite

import numpy as np


@dataclass
class LinkResult:
    """Store the selected region, result reason and candidate measurements."""

    selected: dict | None  # Selected region, or None when no match exists
    reason: str            # Matching result or reason for a missed detection
    candidates: list       # All regions with their distances and radius status


def candidate_order(candidate):
    """Sort by distance, then by label when distances are equal."""

    distance = candidate["distance_px"]
    label = candidate["label"]

    return distance, label


def link_nearest(
    regions,
    previous_position_px,
    search_radius_px=20,
    previous_area_px=None,
    max_area_change=0.30,       #0.30 means the candidate’s area may be up to 30% smaller or larger than the previous measured area.
):
    
    """Choose by area similarity, then distance, inside the search radius."""

    x, y = previous_position_px
    # The previous area must be positive when it is available.
    if previous_area_px is not None:
        if not isfinite(previous_area_px) or previous_area_px <= 0:
            raise ValueError("Previous area must be positive")

    # The allowed area change must be zero or positive.
    if not isfinite(max_area_change) or max_area_change < 0:
        raise ValueError("Allowed area change must be zero or positive")

    # Check the previous position and search radius.
    if not isfinite(x):
        raise ValueError("Position must be finite and search radius positive")

    if not isfinite(y):
        raise ValueError("Position must be finite and search radius positive")

    if not isfinite(search_radius_px):
        raise ValueError("Position must be finite and search radius positive")

    if search_radius_px <= 0:
        raise ValueError("Position must be finite and search radius positive")

    candidates = []

    # Measure each region's distance from the previous position.
    for region in regions:
        dx = region["x_px"] - x
        dy = region["y_px"] - y

        # hypot calculates the straight-line distance.
        distance = hypot(dx, dy)

        # Copy the measurements without changing the original region.
        candidate = dict(region)
        candidate["distance_px"] = distance
        candidate["within_radius"] = distance <= search_radius_px

        candidates.append(candidate)

    # The nearest regions come first.
    # Equal distances are ordered by frame-local region label.
    candidates.sort(key=candidate_order)

# Keep objects that pass the area limits and are inside the radius.
    nearby = []

    for candidate in candidates:
        if candidate["accepted_for_auto"] and candidate["within_radius"]:
            nearby.append(candidate)

    if nearby:
        # Without a previous measured area, choose the nearest object.
        # The candidates are already sorted by distance.
        if previous_area_px is None:
            return LinkResult(nearby[0], "matched", candidates)

        similar = []

        for candidate in nearby:
            area = candidate["area_px"]

            # Compare this area with the previous measured area.
            change = abs(area - previous_area_px) / previous_area_px
            candidate["relative_area_change"] = change

            if change <= max_area_change:
                similar.append(candidate)

        # Nearby objects exist, but none has a similar enough area.
        if not similar:
            return LinkResult(None, "area_change_too_large", candidates)

        # Choose the closest area match.
        # If area differences are equal, choose the nearest position.
        similar.sort(
            key=lambda candidate: (
                candidate["relative_area_change"],
                candidate["distance_px"],
                candidate["label"],
            )
        )

        return LinkResult(similar[0], "matched", candidates)
    
    nearby_region_found = False

    for candidate in candidates:
        if candidate["within_radius"]:
            nearby_region_found = True
            break

    if nearby_region_found:
        reason = "nearby_regions_rejected_by_area"
    elif len(candidates) == 0:
        reason = "no_segmented_regions"
    else:
        reason = "no_accepted_candidate_within_radius"

    return LinkResult(None, reason, candidates)


def select_region_at_click(labels, regions, x, y, radius_px=5):
    """Find a segmented region at or near a click.

    This helper ignores AUTO area limits.
    The current GUI stores raw clicks instead of using this function.
    """

    height, width = labels.shape

    # Check the click coordinates and neighborhood radius.
    if not isfinite(x):
        raise ValueError("Click coordinates must be finite")

    if not isfinite(y):
        raise ValueError("Click coordinates must be finite")

    if not isfinite(radius_px):
        raise ValueError("Click radius must be finite")

    if radius_px < 0:
        raise ValueError("Click radius must be non-negative")

    # Reject clicks outside the image.
    if x < 0 or x >= width:
        return None

    if y < 0 or y >= height:
        return None

    # Read the label of the pixel under the click.
    xi = int(x)
    yi = int(y)

    label = int(labels[yi, xi])

    # Label zero means the click is on background.
    if label == 0:
        # Limit the neighborhood to the image boundaries.
        x0 = max(0, int(x - radius_px))
        x1 = min(width, int(x + radius_px) + 2)
        y0 = max(0, int(y - radius_px))
        y1 = min(height, int(y + radius_px) + 2)

        neighborhood = labels[y0:y1, x0:x1]

        # Find foreground pixels inside this neighborhood.
        yy, xx = np.nonzero(neighborhood)

        if len(xx) == 0:
            return None

        # Convert neighborhood positions to image coordinates.
        image_x = xx + x0
        image_y = yy + y0

        # Calculate distances from the click to those pixels.
        distances = np.hypot(image_x - x, image_y - y)

        nearest = int(np.argmin(distances))

        # The rectangular neighborhood also contains pixels
        # outside the circular search radius.
        if distances[nearest] > radius_px:
            return None

        nearest_x = int(image_x[nearest])
        nearest_y = int(image_y[nearest])

        label = int(labels[nearest_y, nearest_x])

    # Return a copy of the region with this label.
    for region in regions:
        if region["label"] == label:
            return dict(region)

    return None


def predict_position(first, last, frame, max_gap_frames=5):
    """Estimate a position from two successful detections."""

    # We need two positions to estimate movement.
    if first is None or last is None:
        return None

    # Count frames since the last successful detection.
    gap = frame - last["frame"]

    # Predict only a short distance into the future.
    if gap <= 0 or gap > max_gap_frames:
        return None

    frame_difference = last["frame"] - first["frame"]

    if frame_difference <= 0:
        return None

    # Movement in pixels per video frame.
    vx = (last["x_px"] - first["x_px"]) / frame_difference
    vy = (last["y_px"] - first["y_px"]) / frame_difference

    # Continue the same movement.
    x = last["x_px"] + vx * gap
    y = last["y_px"] + vy * gap

    return x, y