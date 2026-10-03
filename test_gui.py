"""Run headless: QT_QPA_PLATFORM=offscreen python -m unittest test_gui."""
import unittest
from unittest.mock import patch
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtTest

from main_window import MainWindow


class FakeVideo:
    def __init__(self, _):
        from pathlib import Path
        self.path = Path("fixture.mp4")
        self.n_frames, self.fps = 5, 25
        self.frames = []
        for x in (12, 14, None, 18, 20):
            frame = np.zeros((40, 60, 3), dtype=np.uint8)
            if x is not None:
                frame[10:14, x:x+4] = 120
            self.frames.append(frame)

    def read(self, index):
        return self.frames[index].copy()

    def close(self):
        pass


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.window.kernel.setCurrentIndex(0)
        with patch("main_window.Video", FakeVideo):
            self.assertTrue(self.window.load_video("fixture.mp4"))
        self.window.manual_click(13, 11)

    def tearDown(self):
        self.window.pause()
        self.window.unsaved_changes = False
        self.window.close()

    def test_auto_failure_keeps_mode_manual_recovery_continues_same_track(self):
        w = self.window
        w.start_auto(); w.timer.stop()
        w.advance_frame()
        self.assertEqual(w.frame_index, 1)
        self.assertEqual(w.anchor["x_px"], 15.5)
        w.advance_frame()
        self.assertEqual(w.mode, "AUTO")
        self.assertFalse(w.playing)
        self.assertEqual(len(w.trajectory.points), 2)
        self.assertIn("no_segmented_regions", w.last_note)
        w.toggle_mode()
        self.assertEqual(w.mode, "MANUAL")
        w.navigate(3); w.manual_click(19, 11)
        w.toggle_mode(); w.timer.stop(); w.advance_frame()
        self.assertEqual([p["frame"] for p in w.trajectory.points], [0, 1, 3, 4])
        self.assertEqual(w.mode, "AUTO")

    def test_earlier_correction_becomes_anchor_and_can_be_undone(self):
        w = self.window
        w.start_auto(); w.timer.stop(); w.advance_frame()
        w.toggle_mode(); w.navigate(0); w.manual_click(13, 11)
        self.assertEqual([p["frame"] for p in w.trajectory.points], [0])
        w.undo()
        self.assertEqual([p["frame"] for p in w.trajectory.points], [0, 1])
        w.manual_click(13, 11); w.start_auto(); w.timer.stop(); w.advance_frame()
        self.assertEqual([p["frame"] for p in w.trajectory.points], [0, 1])

    def test_space_and_m_work_with_numeric_and_combo_focus(self):
        w = self.window
        w.show(); w.activateWindow(); self.app.processEvents()
        for widget in (w.threshold, w.min_area, w.max_area, w.radius, w.sigma,
                       w.playback_fps, w.frame_spin, w.display_low, w.display_high,
                       w.kernel, w.display_mode, w.correction_choice, w.view):
            w.mode = "MANUAL"; w.pause()
            widget.setFocus(); self.app.processEvents()
            QtTest.QTest.keyClick(widget, QtCore.Qt.Key_M)
            self.app.processEvents(); w.timer.stop()
            self.assertEqual(w.mode, "AUTO", type(widget).__name__)
            QtTest.QTest.keyClick(widget, QtCore.Qt.Key_Space)
            self.assertFalse(w.playing)
            self.assertEqual(w.mode, "AUTO")
            QtTest.QTest.keyClick(widget, QtCore.Qt.Key_M)
            self.assertEqual(w.mode, "MANUAL")

    def test_manual_override_and_area_diagnostics(self):
        w = self.window
        w.max_area.setValue(10)
        w.manual_click(13, 11)
        self.assertEqual(w.anchor["area_status"], "above_max_area")
        w.start_auto(); w.timer.stop(); w.advance_frame()
        self.assertEqual(w.mode, "AUTO")
        self.assertIn("nearby_regions_rejected_by_area", w.last_note)

    def test_display_contrast_does_not_change_detection_and_navigation_does_not_append(self):
        w = self.window
        result = w.ensure_detection()
        w.display_low.setValue(30); w.display_high.setValue(180)
        self.assertIs(w.ensure_detection(), result)
        w.navigate(1); w.navigate(0)
        self.assertEqual(len(w.trajectory.points), 1)

    def test_keep_future_policy_replaces_instead_of_duplicating(self):
        w = self.window
        w.start_auto(); w.timer.stop(); w.advance_frame()
        w.toggle_mode(); w.correction_choice.setCurrentIndex(1)
        w.navigate(0); w.manual_click(13, 11)
        self.assertEqual(len(w.trajectory.points), 2)
        w.start_auto(); w.timer.stop(); w.advance_frame()
        self.assertEqual(len(w.trajectory.points), 2)


if __name__ == "__main__":
    unittest.main()
