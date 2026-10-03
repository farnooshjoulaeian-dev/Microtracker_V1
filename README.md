# FluorescenceTracker V0

A semi-automatic single-bacterium tracker for fluorescence videos. This is a
working first version, informed by MicroTracker V0/V1 and the PI's tracker.
The user reviews identity and corrects the same trajectory; AUTO uses distance
alone and never inserts predicted positions.

## Run

From this folder, using the existing Python environment:

```sh
python run.py ../flour_bacteria_test1.mp4
```

Or run `python run.py` and use Load video. Python 3.10 or newer is required;
validation used Python 3.11. For a separate environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py
```

## Workflow

1. Load a video. The initial mode is MANUAL, paused.
2. Click the bacterium. The click identifies a segmented object; its measured
   centroid is stored, not the mouse position. Orange area-rejected objects can
   still be selected manually. If the object is not segmented, adjust threshold.
3. Press Track or M to start AUTO from the selected point into the next frame.
4. Space pauses/resumes playback without changing mode.
5. If the identity is wrong, press M to enter MANUAL and pause. Navigate to the
   appropriate frame, select the correct bacterium, and press M to resume AUTO.
6. Save writes a CSV and a companion `.session.json` with metadata and diagnostics.

AUTO pauses on failure but remains AUTO. Only the user switches modes. The status
bar and log report why it stopped; the table lists area and distance diagnostics.
After a failure, M enters MANUAL so the user can select a recovery point.
MANUAL playback advances frames without adding artificial trajectory points.

Left/Right navigate (arrows edit numbers when a numeric input has focus). The
slider and frame input allow seeking. M and Space work while numeric controls
have focus. Wheel zooms; Fit image restores the full-frame view.

## Correction and gaps

The visible Correction behavior control defaults to removing points after an
earlier manual correction. Those later points may belong to the wrong bacterium.
Undo correction restores the previous trajectory, including removed points.
Alternatively choose Keep later until replaced; points then remain active until
AUTO overwrites them, and should be reviewed before export.

Only one point is stored per frame. Navigation does not add, delete or remeasure
points. Retracking overwrites that frame's point. Missing frames remain missing;
the display does not draw a continuous line across gaps. Clear track is undoable.
Loading another video resets the session and asks before discarding unsaved work.

## Algorithms and units

Full-frame BGR to grayscale → Gaussian blur → fixed threshold → 8-connected
components → floating centroids and foreground pixel counts → explicit area
acceptance → nearest accepted centroid within the search radius.

Starting settings: blur kernel 3x3, sigma 3 pixels, threshold 40 on blurred
grayscale intensities (0..255), area 6..150 pixels², radius 20 pixels. All these
are adjustable and recorded. No background subtraction, NLM, morphology closing,
VAC, motion prediction, automatic gap filling or hidden intensity normalization
is used in the runnable tracker.

The threshold-40 setting preserves a broader footprint than threshold 60 in the
reviewed dim target. It is a prototype setting, not a validated biological size
measurement. Area limits include their endpoints and do not delete rejected labels.
Labels identify regions within one frame, not persistent bacterial identities.
Multiple candidates are resolved by nearest centroid; the user monitors identity.

Grayscale/Overlay/Binary mask and Display low/high affect visualization only.
Cyan boundaries identify area-accepted regions; orange boundaries identify rejected
regions. Green dots/segments are AUTO points, yellow dots are manual points, and
the pink circle is the search radius about the continuation position.

Coordinates and areas are in pixels/pixels². Spatial calibration is intentionally
unset; no guessed micrometre conversion is made. `time_s` uses frame/reported FPS,
assuming constant frame rate. If FPS is unknown, measurement times are blank.
Playback FPS controls presentation speed, not recorded physical time.

## Structure

| File | Responsibility |
|---|---|
| detection.py | Settings, Gaussian smoothing, measured regions and rejection status |
| segmentation.py | Threshold masks and connected-component measurements |
| linking.py | Distance-based selection and area-independent manual label selection |
| trajectory.py | Unique frame-indexed points, correction, Undo and export |
| video.py | Decoding, seeking and video metadata |
| main_window.py | Qt interaction, playback state, previews and diagnostics |
| run.py | Application entry point |

Scientific functions do not depend on Qt. The project uses small flat modules
for now rather than introducing a framework or package hierarchy prematurely.

## Saving and reproducibility

Each CSV row includes zero-based frame, nominal time, x/y, foreground area,
frame-local label, manual/AUTO source, area status, displacement (AUTO only),
detection parameters and search radius used for that point. Changing settings
does not silently recalculate old points.

The companion JSON records video metadata, dependency versions, current settings,
display settings, correction policy, manual clicks, setting changes and failed
link candidate diagnostics. A filename/path identifies the video; the export
does not copy or embed it. Preserve the original video alongside exported results.

## Checks and observed behavior

```sh
QT_QPA_PLATFORM=offscreen python -m unittest discover -p 'test_*.py'
```

Tests cover segmentation geometry, area bounds and retained labels, manual override,
link stop reasons, unique frame storage, correction/Undo, export metadata, video
seek behavior, GUI state transitions, shortcut focus and display/detection separation.

On the reference video, actual distance-only linking from seed (615,502) at frame
0 continued through frame 194 and paused at frame 195. A nearby region of area
183 pixels² exceeded maximum 150. The inspected mask appears to join fluorescent
objects; increasing the limit automatically could accept a merged region.
Uninterrupted linking is not proof of correct identity; the user must review it.

Detection plus linking took about 6 ms per full frame in an exploratory run.
After avoiding content-driven table resizing, a headless GUI run over 74 advances
including decoding, diagnostics, rendering and event processing took median
15.5 ms and 95th percentile 23.1 ms. This fits the nominal 40 ms budget of a 25 FPS
video in that run, but does not guarantee native desktop playback speed.

The single-shot playback timer schedules the next frame after processing, subtracting
elapsed processing time from the desired interval. Sequential playback avoids seeking
on every frame and reuses detection results for click/preview until settings change.
Processing remains synchronous in Qt because it was within the measured budget;
larger videos may require profiling and a worker thread in a later version.

## Scientific comparison artifacts

The three earlier review documents and scripts remain in the project. Saved
comparison images, arrays, CSVs and settings are in outputs/. Historical reviews
describe settings used at that stage; the runnable default is now threshold 40.

```sh
python compare_preprocessing.py --video ../flour_bacteria_test1.mp4
python compare_segmentation.py
python review_detection_sequence.py
python review_linking.py
```

Next research checks: review other cells and the remainder of the video, verify
spatial and temporal calibration, and compare candidate masks at merges/focus
changes. The first version is ready for supervised use, not unattended identity
tracking or validated biological size estimation.
