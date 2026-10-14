"""Store one trajectory, support Undo, and export measurements.

A trajectory contains measured positions from different video frames.
Each frame can have only one stored point.

This file stores data. It does not detect cells or control the GUI.
"""

import copy
import csv
from dataclasses import asdict
import json

from math import isfinite   
from pathlib import Path



class Trajectory:
     
     """Keep the points and editing history of one trajectory."""
     def __init__(self):
        
        self._points = {}   # Each frame number identifies one stored point.
        self._undo = []     # Previous trajectory states, used by Undo.
        self.events = []     # Record actions such as corrections and clearing.


     @property
     def points(self):
         
         ordered_points = []
         frame_numbers = sorted(self._points)

         for frame in frame_numbers:
             stored_point = self._points[frame]
             point_copy = copy.deepcopy(stored_point)
             ordered_points.append(point_copy)
         return ordered_points
     
     def get(self, frame):
         stored_point = self._points.get(frame)
         point_copy = copy.deepcopy(stored_point)
         return point_copy

     def record(
             self,
             frame,
             region,
             source, 
             settings,
             search_radius_px,
             time_s = None,
             truncate_future = False,

     ) :
         
        """Store a point, replacing any existing point for the same frame.

        frame: frame number to store.
        region: dictionary containing the measured position, area and label.
        source: "manual" or "auto".
        settings: detection settings used for this point.
        search_radius_px: tracking search radius, in pixels.
        time_s: measurement time in seconds, or None if unavailable.
        truncate_future: whether to remove points after this frame.
        """


        if source not in ("manual" , "auto"):
            raise ValueError("source must be manual or auto")
        if not isinstance(frame, int):
            raise ValueError("Frame must be a non-negative integer")
        if frame < 0:
            raise ValueError("Frame must be a non-negative integer")
        
        x = region["x_px"]
        y = region["y_px"]
        area = region["area_px"]
        label = region["label"]

        # Reject infinity and NaN in the horizontal coordinate.
        if not isfinite(x):     
            raise ValueError("Value must ne finit")
        if not isfinite(y):
            raise ValueError("Value must be finite")
        

         # AUTO points must come from a measured region.
        if source == "auto":
            if area is None:
                raise ValueError("AUTO points need a measured area")

            if label is None:
                raise ValueError("AUTO points need a region label")

        # Manual points may have no area or label.
        # Convert these values only when they exist.
        if area is not None:
            if not isfinite(area):
                raise ValueError("Area must be finite")

            area = int(area)

        if label is not None:
            label = int(label)

        if source == "manual":
            previous_points = copy.deepcopy(self._points)
            self._undo.append(previous_points)
        
        removed_frames = []
        if truncate_future:
            for stored_frame in self._points:
                if stored_frame > frame:
                    removed_frames.append(stored_frame)



            for stored_frame in removed_frames:
                del self._points[stored_frame]

        point = {}
        point["frame"] = frame
        point["time_s"] = time_s
        point["x_px"] = float(x)
        point["y_px"] = float(y)
        point["area_px"] = area
        point["label"] = label
        point["source"] = source
        if source == "manual":
            point["position_method"] = "manual_click"
        else:
            point["position_method"] = region.get("position_method")
            
        point["area_status"] = region["area_status"]
        point["distance_from_previous_px"] = region.get("distance_px")
        settings_dictionary = asdict(settings)

        for name, value in settings_dictionary.items():
            point[name] = value

        point["search_radius_px"] = search_radius_px

        self._points[frame] = point

        # Add a history event for manual selection.
        if source == "manual":
            event = {}
            # Identify the type of action.
            event["event"] = "manual_selection"
            event["frame"] = frame
            event["removed_later_frames"] = sorted(removed_frames)
            self.events.append(event)

        return copy.deepcopy(point)

     def clear(self):
        """Remove all points, while allowing Undo to restore them."""

        if len(self._points) == 0:
            return

        # Save the current trajectory before removing its points.
        previous_points = copy.deepcopy(self._points)
        self._undo.append(previous_points)

        # Empty the point dictionary.
        self._points.clear()

        self.events.append({"event": "clear"})

     def undo(self):
        """Restore the most recently saved trajectory state.

        Return True if restoration succeeds, or False if nothing is available.
        """
        if len(self._undo) == 0:
            return False

        # pop() removes and returns the last item in the list.
        previous_points = self._undo.pop()
        self._points = previous_points
        self.events.append({"event": "undo"})

        return True
     
     def save(self, csv_path,metadata):
         points = self.points
         if len(points)==0:
             raise ValueError("empty trajectory")
         csv_path = Path(csv_path)

         column_names = list(points[0].keys())

         with csv_path.open("w", newline="") as handle:     #"w" creates the file or overwrites an existing file.
            writer = csv.DictWriter(
                handle,
                fieldnames=column_names,
            )
            writer.writeheader()

            for point in points:
                writer.writerow(point)

        # Build a new dictionary for session information.
        # This avoids adding "events" to the caller's metadata dictionary.
        
         session = {}

        
         for name, value in metadata.items():
            session[name] = value

         session["events"] = self.events

         companion = csv_path.with_suffix(".session.json")

         session_text = json.dumps(
            session,
            indent=2,        
            allow_nan=False, 
         )

         companion.write_text(session_text)

         return companion