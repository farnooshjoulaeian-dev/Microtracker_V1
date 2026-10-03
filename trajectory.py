"""One frame-indexed trajectory with reversible edits and reproducible export."""
import copy
import csv
from dataclasses import asdict
import json
from math import isfinite
from pathlib import Path


class Trajectory:
    def __init__(self):
        self._points = {}
        self._undo = []
        self.events = []

    @property
    def points(self):
        return [copy.deepcopy(self._points[f]) for f in sorted(self._points)]

    def get(self, frame):
        return copy.deepcopy(self._points.get(frame))

    def record(self, frame, region, source, settings, search_radius_px, time_s=None,
               truncate_future=False):
        if source not in ("manual", "auto"):
            raise ValueError("Source must be manual or auto")
        if not isinstance(frame, int) or frame < 0:
            raise ValueError("Frame must be a non-negative integer")
        if not all(isfinite(region[k]) for k in ("x_px", "y_px", "area_px")):
            raise ValueError("Measurements must be finite")
        if source == "manual":
            self._undo.append(copy.deepcopy(self._points))
        removed = []
        if truncate_future:
            removed = [f for f in self._points if f > frame]
            for f in removed:
                del self._points[f]
        point = {"frame": frame, "time_s": time_s,
                 "x_px": float(region["x_px"]), "y_px": float(region["y_px"]),
                 "area_px": int(region["area_px"]), "label": int(region["label"]),
                 "source": source, "area_status": region["area_status"],
                 "distance_from_previous_px": region.get("distance_px"),
                 **asdict(settings), "search_radius_px": search_radius_px}
        self._points[frame] = point
        if source == "manual":
            self.events.append({"event": "manual_selection", "frame": frame,
                                "removed_later_frames": sorted(removed)})
        return copy.deepcopy(point)

    def clear(self):
        if self._points:
            self._undo.append(copy.deepcopy(self._points))
            self._points.clear()
            self.events.append({"event": "clear"})

    def undo(self):
        if not self._undo:
            return False
        self._points = self._undo.pop()
        self.events.append({"event": "undo"})
        return True

    def save(self, csv_path, metadata):
        points = self.points
        if not points:
            raise ValueError("Cannot save an empty trajectory")
        csv_path = Path(csv_path)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=points[0].keys())
            writer.writeheader()
            writer.writerows(points)
        companion = csv_path.with_suffix(".session.json")
        companion.write_text(json.dumps({**metadata, "events": self.events}, indent=2, allow_nan=False))
        return companion
