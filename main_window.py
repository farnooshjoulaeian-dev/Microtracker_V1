"""Interactive single-cell tracker. Scientific processing lives in other modules."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
import skimage
from PyQt5 import QtCore, QtGui, QtWidgets
from detection import DetectionSettings, detect_frame
from linking import link_nearest, predict_position
from trajectory import Trajectory
from video import Video

class FrameView(QtWidgets.QGraphicsView):
    clicked = QtCore.pyqtSignal(float, float)

    def __init__(self):
        """Create the widget and its initial state."""
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.item = QtWidgets.QGraphicsPixmapItem()
        self.scene().addItem(self.item)
        self.setBackgroundBrush(QtGui.QColor('#c7d3c7'))
        self.setMinimumSize(600, 400)
        self.fit_enabled = True

    def display(self, rgb):
        """Show an RGB image. Copy its data so Qt owns the displayed image."""
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        image = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
        self.item.setPixmap(QtGui.QPixmap.fromImage(image))
        self.scene().setSceneRect(0, 0, w, h)
        if self.fit_enabled:
            self.fitInView(self.item, QtCore.Qt.KeepAspectRatio)

    def fit_image(self):
        """Fit the complete image inside the view."""
        self.fit_enabled = True
        self.fitInView(self.item, QtCore.Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        """Refit the image when the window size changes, if automatic fitting is enabled."""
        super().resizeEvent(event)
        if self.fit_enabled and (not self.item.pixmap().isNull()):
            self.fitInView(self.item, QtCore.Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        """Zoom in or out around the mouse position."""
        self.fit_enabled = False
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        if event.angleDelta().y() > 0:
            factor = 1.2
        else:
            factor = 1 / 1.2
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        """Convert a left click to image coordinates and send those coordinates to the window."""
        if event.button() == QtCore.Qt.LeftButton:
            point = self.mapToScene(event.pos())
            self.clicked.emit(point.x(), point.y())
        else:
            super().mousePressEvent(event)

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        """Create the widget and its initial state."""
        super().__init__()
        self.setWindowTitle('FluorescenceTracker V0')
        self.resize(1350, 850)
        self.video = None
        self.frame_index = 0
        self.frame_bgr = None
        self.result = None
        self.cache_settings = None
        self.mode = 'MANUAL'
        self.playing = False
        self.anchor = None
        self.trajectory = Trajectory()
        self.last_note = 'Load a video, then click a bacterium in MANUAL.'
        self.session_started = datetime.now(timezone.utc).isoformat()
        self.correction_policy = 'remove_future'
        self.max_area_change = 0.15  # Allow up to 15% area change.
        self.max_prediction_gap_frames = 3  # Stop predicting after three missing frames.
        self.unsaved_changes = False
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.timer_tick)
        self.build_ui()
        self.update_status()

    def build_ui(self):
        """Create controls, image view and frame navigation. Connect user actions to functions."""
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        middle = QtWidgets.QHBoxLayout()
        layout.addLayout(middle, 1)
        controls = QtWidgets.QWidget()
        controls.setFixedWidth(300)
        controls_layout = QtWidgets.QVBoxLayout(controls)
        form = QtWidgets.QFormLayout()
        controls_layout.addLayout(form)
        self.threshold = self.spin(0, 255, 40)
        self.sigma = QtWidgets.QDoubleSpinBox()
        self.sigma.setRange(0, 7)
        self.sigma.setSingleStep(0.1)
        self.sigma.setValue(3)

        self.vmin = self.spin(0, 254, 0)
        self.vmax = self.spin(1, 255, 255)

        # Keep vmin strictly below vmax.
        self.vmin.valueChanged.connect(
            lambda value: self.vmax.setMinimum(value + 1)
        )
        self.vmax.valueChanged.connect(
            lambda value: self.vmin.setMaximum(value - 1)
        )
        self.min_area = self.spin(1, 1000000, 6)
        self.max_area = self.spin(6, 1000000, 150)
        self.min_area.setMaximum(150)
        self.min_area.valueChanged.connect(self.max_area.setMinimum)
        self.max_area.valueChanged.connect(self.min_area.setMaximum)
        self.radius = self.spin(1, 500, 20)
        self.playback_fps = QtWidgets.QDoubleSpinBox()
        self.playback_fps.setRange(0.1, 240)
        self.playback_fps.setValue(25)
        for label, widget in [
            ('Threshold (0–255)', self.threshold),
            ('Blur sigma (px)', self.sigma),
            ('vmin (0–255)', self.vmin),
            ('vmax (0–255)', self.vmax),
            ('Min area (pixels²)', self.min_area),
            ('Max area (pixels²)', self.max_area),
            ('Search radius (px)', self.radius),
            ('Playback FPS', self.playback_fps),
        ]:
            form.addRow(label, widget)
        separator = QtWidgets.QLabel('Display settings — do not affect detection')
        separator.setWordWrap(True)
        form.addRow(separator)
        self.display_mode = QtWidgets.QComboBox()
        self.display_mode.addItems(['Overlay', 'Grayscale', 'Binary mask'])
        form.addRow('View', self.display_mode)
        hint = QtWidgets.QLabel(
            "M switches mode. Space plays/pauses.\n"
            "Left/Right navigate. Mouse wheel zooms.\n"
            "Manual clicks record your selected position."
        )
        hint.setWordWrap(True)
        form.addRow(hint)
        button_grid = QtWidgets.QGridLayout()
        actions = [('Load', self.choose_video, 'Load video'), ('Track', self.start_auto, 'Start AUTO tracking'), ('Play', self.toggle_play, 'Play/pause (Space)'), ('Save', self.save, 'Save CSV and session settings'), ('Clear', self.clear_track, 'Restart the current video from frame 0'), ('Fit', self.fit_image, 'Fit image')]
        for index, (text, callback, tooltip) in enumerate(actions):
            button = QtWidgets.QPushButton(text)
            button.setToolTip(tooltip)
            button.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
            button.clicked.connect(callback)
            button_grid.addWidget(button, index // 4, index % 4)
        for column in range(4):
            button_grid.setColumnStretch(column, 1)
        controls_layout.addLayout(button_grid)
        controls_layout.addStretch()
        middle.addWidget(controls)
        self.view = FrameView()
        middle.addWidget(self.view, 1)
        bottom = QtWidgets.QHBoxLayout()
        self.frame_label = QtWidgets.QLabel('Frame —')
        self.frame_label.setMinimumWidth(140)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.frame_spin = self.spin(0, 0, 0)
        bottom.addWidget(self.frame_label)
        bottom.addWidget(self.slider, 1)
        bottom.addWidget(self.frame_spin)
        layout.addLayout(bottom)
        self.slider.valueChanged.connect(self.navigate)
        self.frame_spin.valueChanged.connect(self.navigate)
        self.view.clicked.connect(self.manual_click)
        for widget in (self.threshold, self.sigma, self.vmin, self.vmax,self.min_area, 
                       self.max_area, self.radius,):
            widget.valueChanged.connect(self.processing_changed)

        self.display_mode.currentIndexChanged.connect(self.render)
        self.shortcuts = []
        for key, callback in [('M', self.toggle_mode), ('Space', self.toggle_play), ('Right', self.step_forward), ('Left', self.step_back)]:
            self.shortcuts.append(QtWidgets.QShortcut(QtGui.QKeySequence(key), self, activated=callback))
            self.shortcuts[-1].setAutoRepeat(False)
        for widget in self.findChildren(QtWidgets.QAbstractSpinBox):
            widget.installEventFilter(self)
            widget.lineEdit().installEventFilter(self)
        for widget in self.findChildren(QtWidgets.QComboBox):
            widget.installEventFilter(self)

    @staticmethod
    def spin(low, high, value):
        """Create an integer input with a minimum, maximum and initial value."""
        widget = QtWidgets.QSpinBox()
        widget.setMinimumWidth(85)
        widget.setRange(low, high)
        widget.setValue(value)
        return widget

    def eventFilter(self, watched, event):
        """Allow M and Space shortcuts while an input control has keyboard focus."""
        if event.type() == QtCore.QEvent.ShortcutOverride and event.key() in (QtCore.Qt.Key_M, QtCore.Qt.Key_Space) and (event.modifiers() in (QtCore.Qt.NoModifier, QtCore.Qt.ShiftModifier)):
            event.ignore()
            return True
        return super().eventFilter(watched, event)

    def settings(self):
        """Read the controls and create the current detection settings."""
        return DetectionSettings(
            blur_sigma_px=self.sigma.value(),
            vmin_gray=self.vmin.value(),
            vmax_gray=self.vmax.value(),
            threshold_gray=self.threshold.value(),
            min_area_px=self.min_area.value(),
            max_area_px=self.max_area.value(),
            
        )
    def note(self, text, event=None):
        """Store the latest message and optional event, then update the status bar."""
        self.last_note = text
        if event:
            self.trajectory.events.append({'frame': self.frame_index, **event})
        self.update_status()

    def update_status(self):
        """Show the current mode, playback state and latest message."""
        self.statusBar().showMessage(f"{self.mode} | {('playing' if self.playing else 'paused')} | {self.last_note}")

    def pause(self):
        """Stop the timer and mark playback as paused."""
        self.timer.stop()
        self.playing = False
        self.update_status()

    def choose_video(self):
        """Ask for a video path and confirm before discarding unsaved work."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open video', '', 'Videos (*.mp4 *.avi *.mov *.mkv);;All files (*)')
        if not path:
            return
        if self.unsaved_changes and QtWidgets.QMessageBox.question(self, 'Load another video', 'Discard the current unsaved trajectory and open another video?') != QtWidgets.QMessageBox.Yes:
            return
        self.load_video(path)

    def load_video(self, path):
        """Open a video, read its first frame and reset the tracking session."""
        try:
            video = Video(path)
            first = video.read(0)
        except (ValueError, IndexError) as error:
            if 'video' in locals():
                video.close()
            self.note(str(error))
            return False
        self.pause()
        if self.video:
            self.video.close()
        self.video = video
        self.mode = 'MANUAL'
        self.anchor = None
        self.trajectory = Trajectory()
        self.unsaved_changes = False
        self.session_started = datetime.now(timezone.utc).isoformat()
        self.frame_index = 0
        self.frame_bgr = first
        self.result = None
        for widget in (self.slider, self.frame_spin):
            widget.blockSignals(True)
            widget.setRange(0, video.n_frames - 1)
            widget.setValue(0)
            widget.blockSignals(False)
        self.playback_fps.setValue(video.fps or 25)
        self.view.fit_enabled = True
        self.note(f'Loaded {video.path.name}; {video.n_frames} frames. Click a bacterium in MANUAL.')
        self.refresh_display()
        return True

    def ensure_detection(self):
        """Reuse the current detection unless the image or settings changed."""
        if self.frame_bgr is None:
            return None
        settings = self.settings()
        if self.result is None or self.cache_settings != settings:
            self.result = detect_frame(self.frame_bgr, settings)
            self.cache_settings = settings
        return self.result

    def load_frame(self, index):
        """Read a frame and update navigation controls without triggering their signals."""
        try:
            self.frame_bgr = self.video.read(index)
        except (ValueError, IndexError) as error:
            self.pause()
            self.note(str(error), {'event': 'decode_failure'})
            return False
        self.frame_index = index
        self.result = None
        for widget in (self.slider, self.frame_spin):
            widget.blockSignals(True)
            widget.setValue(index)
            widget.blockSignals(False)
        return True

    def navigate(self, index):
        """Pause and browse a frame. Use its existing trajectory point as the anchor."""
        if not self.video:
            return
        self.pause()
        if index == self.frame_index or self.load_frame(index):
            self.anchor = self.trajectory.get(self.frame_index)
            self.refresh_display()
            self.note('Frame navigation paused playback; select a continuation point if needed.')

    def step_back(self):
        """Browse the previous frame without going below frame zero."""
        if self.video:
            self.navigate(max(0, self.frame_index - 1))

    def step_forward(self):
        """Pause playback and advance one frame."""
        self.pause()
        self.advance_frame()

    def valid_anchor(self):
        """Check whether the continuation point belongs to the displayed frame."""
        return self.anchor is not None and self.anchor['frame'] == self.frame_index

    def start_auto(self):
        """Start AUTO. Resuming AUTO may use the last successful point after a miss."""
        if not self.video:
            self.note('Load a video first.')
            return
        if self.anchor is None or (self.mode != 'AUTO' and (not self.valid_anchor())):
            self.note('Select the cell on this frame in MANUAL, or return to a stored trajectory point.')
            return
        if self.frame_index == self.video.n_frames - 1:
            self.note('End of video; navigate to an earlier frame to continue.')
            return
        self.mode = 'AUTO'
        self.playing = True
        self.note('AUTO started from the selected point.', {'event': 'start_auto'})
        self.timer.start(max(1, round(1000 / self.playback_fps.value())))

    def toggle_mode(self):
        """Enter MANUAL and pause, or attempt to start AUTO."""
        if not self.video:
            self.note('Load a video first.')
            return
        if self.mode == 'AUTO':
            self.pause()
            self.mode = 'MANUAL'
            self.note('MANUAL: navigate and click the correct bacterium.', {'event': 'enter_manual'})
        else:
            self.start_auto()
        self.render()

    def toggle_play(self):
        """Pause active playback, or resume according to the current mode."""
        if not self.video:
            return
        if self.playing:
            self.pause()
            self.note('Playback paused.')
        elif self.mode == 'AUTO':
            self.start_auto()
        elif self.frame_index < self.video.n_frames - 1:
            self.playing = True
            self.note('MANUAL playback: no trajectory points are added.')
            self.timer.start(max(1, round(1000 / self.playback_fps.value())))

    def timer_tick(self):
        """Advance one frame and schedule the next, allowing for processing time."""
        start = perf_counter()
        if not self.playing:
            return
        self.advance_frame()
        if self.playing:
            delay = 1000 / self.playback_fps.value() - (perf_counter() - start) * 1000
            self.timer.start(max(1, round(delay)))

    def advance_frame(self):
        """Advance playback and predict positions across short AUTO gaps."""
        if not self.video:
            return
        if self.frame_index >= self.video.n_frames - 1:
            self.pause()
            self.note('End of video.')
            return
        if self.mode == 'AUTO' and self.anchor is None:
            self.pause()
            self.note('AUTO paused: current frame has no continuation point. M enters MANUAL.')
            return
        previous = self.anchor
        if not self.load_frame(self.frame_index + 1):
            return
        if self.mode == 'AUTO':
            result = self.ensure_detection()
            # Keep actual positions only; predictions must not set the velocity or area.
            measured = []
            for point in self.trajectory.points:
                if point["frame"] >= self.frame_index:
                    continue
                if point["source"] == "manual" or point["area_px"] is not None:
                    measured.append(point)

            last_measured = measured[-1] if measured else None
            previous_area = None
            if last_measured is not None:
                previous_area = last_measured.get("area_px")

                # Re-establish the area after detection settings change.
                for name, value in asdict(self.settings()).items():
                    if last_measured.get(name) != value:
                        previous_area = None
                        break

            linked = link_nearest(
                result.regions,
                (previous["x_px"], previous["y_px"]),
                search_radius_px=self.radius.value(),
                previous_area_px=previous_area,
                max_area_change=self.max_area_change,
            )
            if linked.selected is None:
                self.note(f'AUTO: no match at frame {self.frame_index}: {linked.reason}. Playback continues; M enters MANUAL.', {'event': 'auto_miss', 'reason': linked.reason, 'settings': asdict(self.settings()), 'search_radius_px': self.radius.value(), 'candidates': linked.candidates})

                prediction = None
                if len(measured) >= 2:
                    prediction = predict_position(
                        measured[-2], measured[-1], self.frame_index,
                        max_gap_frames=self.max_prediction_gap_frames,
                    )

                if prediction is not None:
                    x, y = prediction
                    height, width = self.frame_bgr.shape[:2]

                    # Do not add a predicted position outside the image.
                    if 0 <= x < width and 0 <= y < height:
                        point = {
                            "x_px": x,
                            "y_px": y,
                            "area_px": None,
                            "label": None,
                            "area_status": "not_measured",
                            "distance_px": float(np.hypot(
                                x - previous["x_px"], y - previous["y_px"],
                            )),
                        }
                        self.anchor = self.record_point(point, "auto", predicted=True)
                        gap = self.frame_index - last_measured["frame"]
                        self.last_note = (
                            f'Frame {self.frame_index}: continuing across gap '
                            f'{gap}/{self.max_prediction_gap_frames}.'
                        )
            else:
                self.anchor = self.record_point(linked.selected, 'auto')
                self.last_note = f"Frame {self.frame_index}: area {linked.selected['area_px']} px²; step {linked.selected['distance_px']:.2f} px."
        else:
            self.anchor = self.trajectory.get(self.frame_index)
        self.refresh_display()
        if self.frame_index == self.video.n_frames - 1:
            self.pause()
            self.note('End of video.')

    def record_point(self, region, source, predicted=False):
        """Store a point with its time, settings and correction policy."""
        if self.video.fps:
            time_s = self.frame_index / self.video.fps
        else:
            time_s = None
        self.unsaved_changes = True
        return self.trajectory.record(self.frame_index, region, source, self.settings(), self.radius.value(), time_s=time_s, truncate_future=source == 'manual' and self.correction_policy == 'remove_future', predicted=predicted)

    def manual_click(self, x, y):
        """Store a valid raw click as a manual position, without measuring region area or label."""
        if self.video is None:
            return
        if self.mode != 'MANUAL':
            self.note('Press M to enter MANUAL before selecting a cell.')
            return
        if not np.isfinite(x):
            return
        if not np.isfinite(y):
            return
        height, width = self.frame_bgr.shape[:2]
        if x < 0 or x >= width:
            return
        if y < 0 or y >= height:
            return
        self.pause()
        point = {}
        point['x_px'] = float(x)
        point['y_px'] = float(y)
        point['area_px'] = None
        point['label'] = None
        point['area_status'] = 'not_measured'
        later_count = 0
        for stored_point in self.trajectory.points:
            if stored_point['frame'] > self.frame_index:
                later_count += 1
        self.anchor = self.record_point(point, 'manual')
        event = {}
        event['event'] = 'manual_click'
        event['x_click_px'] = float(x)
        event['y_click_px'] = float(y)
        message = f'Manual frame {self.frame_index}: click ({x:.2f}, {y:.2f}). M resumes AUTO.'
        self.note(message, event)
        if later_count > 0:
            if self.correction_policy == 'remove_future':
                message = f'Removed {later_count} later points. M resumes AUTO.'
                self.note(message)
        self.refresh_display()

    def processing_changed(self):
        """Invalidate cached detection after a parameter change. Existing points retain their settings."""
        if not self.video:
            return
        #self.pause()
        self.result = None
        self.note('Detection settings changed; existing points retain their original settings.', {'event': 'settings_changed', 'settings': asdict(self.settings()), 'search_radius_px': self.radius.value()})
        self.refresh_display()

    def clear_track(self):
        """Reload the current video at frame 0 with a fresh tracking session."""
        if not self.video:
            return
        self.load_video(self.video.path)

    def fit_image(self):
        """Fit the complete image inside the view."""
        self.view.fit_image()

    def refresh_display(self):
        """Ensure detection is available, redraw the image and update status."""
        result = self.ensure_detection()
        if result is None:
            return
        self.render()
        self.update_status()

    def render(self):
        """Draw the selected image view, region boundaries, trajectory and search radius."""
        if self.frame_bgr is None:
            return
        result = self.ensure_detection()
        # Display original grayscale; contrast stretching is used only for thresholding.
        display = result.gray
        if self.display_mode.currentText() == 'Binary mask':
            display = result.mask.astype(np.uint8) * 255
        rgb = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)
        
        if self.display_mode.currentText() == 'Overlay':
            for region in result.regions:
                # Select only this connected region.
                region_mask = result.labels == region['label']

                color = (0, 220, 220)  # Cyan for every region

                contours, _ = cv2.findContours(
                    region_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                cv2.drawContours(rgb, contours, -1, color, 1)
            
        
        
        points = []
        for p in self.trajectory.points:
            if p['frame'] <= self.frame_index:
                points.append(p)
        for first, second in zip(points, points[1:]):
            p1, p2 = ((round(first['x_px']), round(first['y_px'])), (round(second['x_px']), round(second['y_px'])))
            if second['frame'] == first['frame'] + 1:
                cv2.line(rgb, p1, p2, (0, 255, 0), 1)
        for point in points:
            if point['source'] == 'manual':
                color = (255, 255, 0)
            else:
                color = (0, 255, 0)
            cv2.circle(rgb, (round(point['x_px']), round(point['y_px'])), 1, color, -1)
        if self.anchor:
            xy = (round(self.anchor['x_px']), round(self.anchor['y_px']))
            cv2.circle(rgb, xy, self.radius.value(), (255, 255, 255), 1)
        self.view.display(rgb)
        self.frame_label.setText(f'Frame {self.frame_index} / {self.video.n_frames - 1}')

    def metadata(self):
        """Collect session, video, settings and dependency information for export."""
        information = {}
        information['software'] = 'FluorescenceTracker'
        information['version'] = '0.1.0'
        information['session_started_utc'] = self.session_started
        information['saved_utc'] = datetime.now(timezone.utc).isoformat()
        information['video'] = self.video.metadata()
        information['frame_indexing'] = 'zero-based'
        information["manual_position_rule"] = "Raw image click coordinates"
        information["auto_position_rule"] = ("Outer-contour centroid; pixel-centroid fallback " "when contour area is zero")
        information['prediction'] = 'constant velocity from the last two actual positions'
        information['max_prediction_gap_frames'] = self.max_prediction_gap_frames
        information['point_units'] = 'pixels; area is foreground pixel count'
        information['current_detection_settings'] = asdict(self.settings())
        information['current_search_radius_px'] = self.radius.value()
        information['max_relative_area_change'] = self.max_area_change
        information['candidate_selection'] = 'closest area, then distance; nearest distance when no comparable area exists'
        information['correction_policy'] = self.correction_policy
        information['display'] = {'view': self.display_mode.currentText()}
        information['playback_fps'] = self.playback_fps.value()
        information['dependencies'] = {'opencv': cv2.__version__, 'numpy': np.__version__, 'skimage': skimage.__version__, 'qt': QtCore.QT_VERSION_STR, 'pyqt': QtCore.PYQT_VERSION_STR}
        return information

    def save(self):
        """Ask for an output path and save CSV measurements plus JSON metadata."""
        self.pause()
        if not self.video or not self.trajectory.points:
            self.note('No trajectory to save.')
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save trajectory', 'trajectory.csv', 'CSV (*.csv)')
        if not path:
            return
        path = str(Path(path).with_suffix('.csv'))
        companion = Path(path).with_suffix('.session.json')
        if companion.exists() and QtWidgets.QMessageBox.question(self, 'Replace settings', f'Replace existing settings file {companion.name}?') != QtWidgets.QMessageBox.Yes:
            return
        try:
            companion = self.trajectory.save(path, self.metadata())
        except (OSError, ValueError) as error:
            self.note(f'Save failed: {error}')
            return
        self.note(f'Saved {Path(path).name} and {companion.name}.')
        self.unsaved_changes = False

    def closeEvent(self, event):
        """Confirm unsaved changes before closing and release the video."""
        self.pause()
        if self.unsaved_changes and QtWidgets.QMessageBox.question(self, 'Unsaved trajectory', 'Close without saving trajectory changes?') != QtWidgets.QMessageBox.Yes:
            event.ignore()
            return
        if self.video:
            self.video.close()
        event.accept()
