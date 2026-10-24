"""Test GUI behavior using a small artificial video.

Run from the project root:

QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
"""

import unittest                    # Test cases and assertions
from unittest.mock import patch    # Temporarily replace the real Video class
from pathlib import Path           # Represent the fake video's filename

import numpy as np                 # Create artificial image frames
from PyQt5 import QtCore, QtWidgets, QtTest

from main_window import MainWindow


class FakeVideo:
    """Provide predictable frames without opening a video file."""

    def __init__(self, path):
        # The GUI expects a video path, frame count and reported FPS.
        # The supplied path is not opened in this fake implementation.
        self.path = Path("fixture.mp4")
        self.n_frames = 5
        self.fps = 25

        self.frames = []

        # These are the left edges of the artificial cell.
        # None creates a blank frame to simulate a missed detection.
        cell_positions = [12, 14, None, 18, 20]

        for x in cell_positions:
            # Create a black BGR image: height 40, width 60, channels 3.
            frame = np.zeros((40, 60, 3), dtype=np.uint8)

            if x is not None:
                # Add a bright 4 × 4 square, with area 16 pixels².
                frame[10:14, x:x + 4] = 120

            self.frames.append(frame)

    def read(self, index):
        # Return a copy so the GUI cannot modify the original test frame.
        return self.frames[index].copy()

    def close(self):
        # No decoder or file needs closing in this fake implementation.
        pass


class GuiTests(unittest.TestCase):
    """Check tracking interaction, correction and display behavior."""

    @classmethod
    def setUpClass(cls):
        # Qt requires a QApplication before any widgets are created.
        # Reuse an existing application if one is already running.
        cls.app = QtWidgets.QApplication.instance()

        if cls.app is None:
            cls.app = QtWidgets.QApplication([])

    def setUp(self):
        """Create a fresh window before each test."""

        self.window = MainWindow()

        # Disable smoothing to preserve the artificial square's geometry.
        self.window.sigma.setValue(0)

        # Replace Video only while loading the artificial video.
        # The GUI receives a FakeVideo object instead of opening a file.
        with patch("main_window.Video", FakeVideo):
            loaded = self.window.load_video("fixture.mp4")
            self.assertTrue(loaded)

        # Begin every test with a raw manual point on frame zero.
        self.window.manual_click(13, 11)

    def tearDown(self):
        """Clean up the window after each test."""

        self.window.pause()

        # Avoid the unsaved-changes dialog during automated cleanup.
        self.window.unsaved_changes = False
        self.window.close()

    def stored_frames(self):
        """Return stored frame numbers using an explicit loop."""

        frames = []

        for point in self.window.trajectory.points:
            frames.append(point["frame"])

        return frames

    def test_auto_miss_continues_and_manual_correction_keeps_same_track(self):
        w = self.window

        # Start AUTO, but stop its timer so the test controls frame steps.
        # Stopping the timer directly leaves w.playing set to True.
        w.start_auto()
        w.timer.stop()

        # Frame 1 contains a detected square centered at x = 15.5.
        w.advance_frame()

        self.assertEqual(w.frame_index, 1)
        self.assertEqual(w.anchor["x_px"], 15.5)

        # Frame 2 is blank: continue the track with a predicted position.
        w.advance_frame()

        self.assertEqual(w.frame_index, 2)
        self.assertEqual(w.mode, "AUTO")
        self.assertTrue(w.playing)
        self.assertEqual(w.anchor["frame"], 2)
        self.assertEqual(self.stored_frames(), [0, 1, 2])
        self.assertIsNone(w.anchor["area_px"])
        self.assertEqual(w.anchor["source"], "auto")
        self.assertAlmostEqual(w.anchor["x_px"], 18.0)
        self.assertEqual(w.anchor["time_s"], 2 / 25)
        self.assertIn("gap 1/3", w.last_note)

        # Test the playback function used by Space: pause, then resume.
        w.toggle_play()
        self.assertFalse(w.playing)

        w.toggle_play()
        w.timer.stop()
        self.assertTrue(w.playing)

        # Frame 3 contains the cell again, within the retained search radius.
        w.advance_frame()

        self.assertEqual(w.frame_index, 3)
        self.assertEqual(w.anchor["frame"], 3)
        self.assertEqual(w.anchor["x_px"], 19.5)
        self.assertEqual(self.stored_frames(), [0, 1, 2, 3])

        # Enter MANUAL and replace frame 3 with a raw click.
        w.toggle_mode()

        self.assertEqual(w.mode, "MANUAL")
        self.assertFalse(w.playing)

        w.manual_click(19, 11)

        self.assertEqual(self.stored_frames(), [0, 1, 2, 3])

        corrected_point = w.trajectory.get(3)
        self.assertEqual(corrected_point["source"], "manual")
        self.assertEqual(corrected_point["x_px"], 19.0)
        self.assertEqual(corrected_point["y_px"], 11.0)

        # Continue the same trajectory into the final frame.
        w.toggle_mode()
        w.timer.stop()
        w.advance_frame()

        self.assertEqual(self.stored_frames(), [0, 1, 2, 3, 4])
        self.assertEqual(w.mode, "AUTO")

        # Playback stops at the end of the video.
        self.assertFalse(w.playing)

    def test_earlier_correction_becomes_anchor_and_removes_later_points(self):
        w = self.window

        # Record an AUTO point on frame 1.
        w.start_auto()
        w.timer.stop()
        w.advance_frame()

        # Correct frame 0 using a different manual position.
        # The default policy removes later points.
        w.toggle_mode()
        w.navigate(0)
        w.manual_click(12, 10)

        self.assertEqual(self.stored_frames(), [0])
        self.assertEqual(w.anchor["frame"], 0)
        self.assertEqual(w.anchor["x_px"], 12.0)
        self.assertEqual(w.anchor["y_px"], 10.0)

        # Continue AUTO tracking from the corrected anchor.
        w.start_auto()
        w.timer.stop()
        w.advance_frame()

        self.assertEqual(self.stored_frames(), [0, 1])

    def test_space_and_m_work_with_numeric_and_combo_focus(self):
        w = self.window

        # Show and activate the window for keyboard-event testing.
        w.show()
        w.activateWindow()
        self.app.processEvents()

        # Check shortcuts while each control has keyboard focus.
        widgets = [
            w.threshold,
            w.min_area,
            w.max_area,
            w.radius,
            w.sigma,
            w.playback_fps,
            w.frame_spin,
            w.vmin,
            w.vmax,
            w.display_mode,
            w.view,
        ]

        for widget in widgets:
            # Reset the state before testing the next control.
            w.mode = "MANUAL"
            w.pause()

            widget.setFocus()
            self.app.processEvents()

            # M starts AUTO from the selected point.
            QtTest.QTest.keyClick(widget, QtCore.Qt.Key_M)
            self.app.processEvents()
            w.timer.stop()

            self.assertEqual(w.mode, "AUTO", type(widget).__name__)

            # Space pauses without switching modes.
            QtTest.QTest.keyClick(widget, QtCore.Qt.Key_Space)

            self.assertFalse(w.playing)
            self.assertEqual(w.mode, "AUTO")

            # M enters MANUAL.
            QtTest.QTest.keyClick(widget, QtCore.Qt.Key_M)

            self.assertEqual(w.mode, "MANUAL")

    def test_manual_click_is_raw_and_auto_still_checks_area(self):
        w = self.window

        # The artificial square has area 16, above this AUTO maximum.
        w.max_area.setValue(10)

        # Manual recording stores the click, not the detected centroid.
        w.manual_click(13, 11)

        self.assertEqual(w.anchor["x_px"], 13.0)
        self.assertEqual(w.anchor["y_px"], 11.0)
        self.assertEqual(w.anchor["source"], "manual")

        # A raw click does not measure segmentation area or label.
        self.assertIsNone(w.anchor["area_px"])
        self.assertIsNone(w.anchor["label"])
        self.assertEqual(w.anchor["area_status"], "not_measured")

        # AUTO still rejects the next region because of its area.
        w.start_auto()
        w.timer.stop()
        w.advance_frame()

        self.assertEqual(w.mode, "AUTO")
        self.assertTrue(w.playing)
        self.assertIn("nearby_regions_rejected_by_area", w.last_note)

        # Keep the manual anchor without adding an AUTO point.
        self.assertEqual(len(w.trajectory.points), 1)
        self.assertEqual(w.anchor["frame"], 0)

    def test_display_changes_and_navigation_do_not_change_measurements(self):
        w = self.window

        original_result = w.ensure_detection()
        original_points = w.trajectory.points

        # Switching the display view should reuse the detection result.
        w.display_mode.setCurrentText("Binary mask")

        current_result = w.ensure_detection()

        # assertIs checks that these are the same cached object.
        self.assertIs(current_result, original_result)

        # Browsing should not add or modify measurements.
        w.navigate(1)
        w.navigate(0)

        self.assertEqual(w.trajectory.points, original_points)

    def test_prediction_stops_after_three_frames_and_uses_actual_positions(self):
        w = self.window
        blank = np.zeros((40, 60, 3), dtype=np.uint8)
        w.video.frames[2:] = [blank.copy() for _ in range(5)]
        w.video.n_frames = 7
        w.start_auto()
        w.timer.stop()
        w.advance_frame()
        first = w.trajectory.get(0)
        last = w.trajectory.get(1)
        from linking import predict_position
        with patch("main_window.predict_position", wraps=predict_position) as predict:
            for frame in range(2, 6):
                w.advance_frame()
                self.assertEqual(predict.call_args.args[:2], (first, last))
        self.assertEqual(self.stored_frames(), [0, 1, 2, 3, 4])
        self.assertIsNone(w.trajectory.get(5))
        self.assertTrue(w.playing)
        self.assertEqual(w.anchor["frame"], 4)

    def test_prediction_keeps_last_measured_area_for_matching(self):
        w = self.window
        # Frame 3 contains a larger object that must fail area comparison.
        w.video.frames[3][:] = 0
        w.video.frames[3][9:15, 16:22] = 120
        w.start_auto()
        w.timer.stop()
        w.advance_frame()  # Actual area: 16.
        w.advance_frame()  # Prediction: no measured area.
        w.advance_frame()  # The area-36 object must not be accepted.
        self.assertIsNone(w.trajectory.get(3)["area_px"])
        misses = [event for event in w.trajectory.events if event["event"] == "auto_miss"]
        self.assertEqual(misses[-1]["reason"], "area_change_too_large")

    def test_prediction_requires_two_actual_positions_and_stays_inside_image(self):
        w = self.window
        w.video.frames[1][:] = 0
        w.start_auto()
        w.timer.stop()
        w.advance_frame()
        self.assertEqual(self.stored_frames(), [0])

        # Restore frame 1, obtain an actual position, then reject an outside estimate.
        w.video.frames[1][10:14, 14:18] = 120
        w.navigate(0)
        w.start_auto()
        w.timer.stop()
        w.advance_frame()
        with patch("main_window.predict_position", return_value=(-1, 10)):
            w.advance_frame()
        self.assertEqual(self.stored_frames(), [0, 1])

    def test_clear_starts_a_fresh_session(self):
        w = self.window
        w.vmin.setValue(10)
        settings = w.settings()
        w.start_auto()
        w.timer.stop()
        w.advance_frame()
        with patch("main_window.Video", FakeVideo):
            w.clear_track()
        self.assertEqual(w.frame_index, 0)
        self.assertEqual(w.mode, "MANUAL")
        self.assertFalse(w.playing)
        self.assertFalse(w.timer.isActive())
        self.assertEqual(w.trajectory.points, [])
        self.assertEqual(w.trajectory.events, [])
        self.assertIsNone(w.anchor)
        self.assertFalse(w.unsaved_changes)
        self.assertEqual(w.settings(), settings)
        self.assertFalse(hasattr(w, "undo"))

# This runs the tests when the file is executed directly.
# Test discovery also finds the methods inside GuiTests.
if __name__ == "__main__":
    unittest.main()