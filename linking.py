"""Distance-only linking and explicit diagnostics; no motion prediction."""
from dataclasses import dataclass
from math import hypot, isfinite


@dataclass
class LinkResult:
    selected: dict | None
    reason: str
    candidates: list


def link_nearest(regions, previous_position_px, search_radius_px=20):
    """Choose nearest area-accepted centroid within an inclusive radius.

    Multiple candidates are allowed: the user monitors identity. Exact ties are
    resolved by frame-local label order, recorded through candidate diagnostics.
    """
    x, y = previous_position_px
    if not all(isfinite(v) for v in (x, y, search_radius_px)) or search_radius_px <= 0:
        raise ValueError("Position must be finite and search radius positive")
    candidates = []
    for region in regions:
        distance = hypot(region["x_px"] - x, region["y_px"] - y)
        candidates.append({**region, "distance_px": distance,
                           "within_radius": distance <= search_radius_px})
    candidates.sort(key=lambda r: (r["distance_px"], r["label"]))
    accepted = [r for r in candidates if r["accepted_for_auto"]]
    if accepted and accepted[0]["within_radius"]:
        return LinkResult(accepted[0], "matched", candidates)
    if any(r["within_radius"] for r in candidates):
        reason = "nearby_regions_rejected_by_area"
    elif not candidates:
        reason = "no_segmented_regions"
    else:
        reason = "no_accepted_candidate_within_radius"
    return LinkResult(None, reason, candidates)


def select_region_at_click(labels, regions, x, y, radius_px=5):
    """Direct label hit or nearest foreground pixel in a circular neighborhood.

    Deliberately ignores AUTO area restrictions. A background click is not itself
    a measurement; return None when no segmented region is nearby.
    """
    import numpy as np
    h, w = labels.shape
    if not all(isfinite(v) for v in (x, y, radius_px)) or radius_px < 0:
        raise ValueError("Click coordinates and radius must be finite; radius non-negative")
    if not (0 <= x < w and 0 <= y < h):
        return None
    xi, yi = int(x), int(y)
    label = int(labels[yi, xi])
    if not label:
        x0, x1 = max(0, int(x-radius_px)), min(w, int(x+radius_px)+2)
        y0, y1 = max(0, int(y-radius_px)), min(h, int(y+radius_px)+2)
        yy, xx = np.nonzero(labels[y0:y1, x0:x1])
        if not len(xx):
            return None
        distances = np.hypot(xx+x0-x, yy+y0-y)
        nearest = int(np.argmin(distances))
        if distances[nearest] > radius_px:
            return None
        label = int(labels[yy[nearest]+y0, xx[nearest]+x0])
    return next((dict(r) for r in regions if r["label"] == label), None)
