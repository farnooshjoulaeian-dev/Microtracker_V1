# Microtracker_V1

A semi-automatic single-bacterium tracker for fluorescence microscopy videos,
with a PyQt interface, distance-based automatic linking and manual trajectory
correction.

The user monitors cell identity and corrects the trajectory when necessary.
The software records measured region centroids and does not insert predicted
positions or interpolate missing frames.

## Installation

Requires Python 3.10 or newer. Development testing used Python 3.11.

Clone the repository and enter its folder:

```sh
git clone https://github.com/farnooshjoulaeian-dev/Microtracker_V1.git
cd Microtracker_V1
```

Create and activate a virtual environment:

```sh
python -m venv .venv
```

On macOS or Linux:

```sh
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```sh
python -m pip install -r requirements.txt
```

## Run

```sh
python run.py
```

Use Load to select a video, or supply its path:

```sh
python run.py /path/to/video.mp4
```

Video files are not included in this repository.

## Tracking workflow

1. Load a video. The application starts in MANUAL mode, paused.
2. Click a segmented bacterium to record its measured centroid.
3. Press M or Track to start AUTO from the selected point.
4. Monitor the trajectory. Space pauses or resumes playback.
5. To correct identity, press M to enter MANUAL, navigate to the appropriate
   frame and click the correct region.
6. Press M to resume AUTO as part of the same trajectory.
7. Use Save to export measurements and session metadata.

AUTO pauses when no eligible candidate is within the search radius. It remains
in AUTO mode until the user changes modes.

Manual selection bypasses AUTO area limits, but still requires a segmented
region. A mouse click itself is not stored as a measured cell position.

### Controls

| Control | Action |
|---|---|
| M | Switch between MANUAL and AUTO |
| Space | Pause or resume playback |
| Left / Right | Navigate frames |
| Slider or frame input | Seek to a frame |
| Mouse wheel | Zoom |
| Fit | Restore the full-frame view |
| Undo | Restore the previous correction or cleared trajectory |

Arrow keys edit values when a numeric input has focus.

## Detection and linking

The current processing pipeline is:

1. Convert the full frame to grayscale.
2. Apply Gaussian smoothing.
3. Apply a fixed intensity threshold.
4. Measure 8-connected foreground regions.
5. Mark regions accepted or rejected by the area limits.
6. Link the nearest accepted centroid within the search radius.

All segmented regions are retained for diagnostics and manual selection.

### Starting parameters

| Parameter | Default |
|---|---|
| Gaussian kernel | 3 × 3 pixels |
| Gaussian sigma | 3 pixels |
| Intensity threshold | 40 on a 0–255 grayscale scale |
| Accepted area | 6–150 pixels², inclusive |
| Search radius | 20 pixels |

These parameters are adjustable and recorded with measurements. Defaults are
starting settings, not values validated for every video.

AUTO uses Euclidean distance only. It does not use directional persistence,
velocity autocorrelation, motion prediction or automatic gap filling.

Frame-local region labels are segmentation identifiers, not persistent cell
identities. The nearest candidate may be a different bacterium, so visual
supervision is required.

## Correction and missing frames

Only one point is stored per frame. A correction replaces that frame's point.

By default, correcting an earlier frame removes later trajectory points.
Undo restores the previous trajectory. The alternative correction policy
retains later points until replaced; these should be reviewed before export.

Navigation alone does not change measurements. Missing frames remain missing,
and trajectory lines are not drawn across gaps.

## Measurements and units

- Positions are segmented-region centroids in pixels.
- Area is the foreground pixel count, expressed as pixels².
- Spatial calibration is not supplied, so no micrometre conversion is made.
- Nominal time is the zero-based frame index divided by reported video FPS.
  This assumes constant frame rate.
- If video FPS is unknown, measurement times are left blank.
- Playback FPS changes display speed, not measurement time.

The fluorescent footprint depends on threshold, focus, intensity and overlap.
Its area is not automatically a physical cell-size measurement.

## Display

Grayscale, overlay, binary-mask and display-contrast settings affect
visualization only.

| Appearance | Meaning |
|---|---|
| Cyan boundaries | Area-accepted regions |
| Orange boundaries | Area-rejected regions |
| Green trajectory points and lines | AUTO measurements |
| Yellow points | Manual measurements |
| Pink circle | Search radius around the continuation position |

## Exports and reproducibility

Save produces a CSV and a companion `.session.json`.

The CSV records frame, nominal time, position, region area, frame-local label,
manual/AUTO source, area status, AUTO displacement and processing parameters
used for each point.

The JSON records video metadata, dependency versions, session settings,
correction policy and diagnostic events.

Changing settings does not silently recalculate existing measurements.
Exports identify the source video but do not embed it. Preserve the original
video with the exported results.

## Code structure

| File or folder | Responsibility |
|---|---|
| detection.py | Detection settings and preprocessing |
| segmentation.py | Thresholding and region measurements |
| linking.py | Distance linking and manual region selection |
| trajectory.py | Frame-indexed measurements, correction and export |
| video.py | Video decoding, seeking and metadata |
| main_window.py | GUI interaction, playback and visualization |
| run.py | Application entry point |
| tests/ | Automated software checks |

Scientific processing functions are independent of the Qt interface.

## Testing

Run from the repository root:

```sh
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
```

The 27 tests cover segmentation measurements, area limits, manual override,
distance linking, correction, export, video access and GUI state behavior.

Passing these tests verifies software behavior. It does not establish
biological accuracy or correct identity throughout a recorded trajectory.

## Current limitations

- Cell identity requires human supervision.
- Overlapping fluorescent objects can form a merged region.
- Fixed-threshold segmentation may need adjustment between videos.
- Physical calibration and independently annotated identity validation
  remain necessary for quantitative biological conclusions.
- Processing runs synchronously in the GUI thread. Performance depends on
  video size and the computer.
- This is an interactive selected-cell tracker, not a population tracker.
## License

MIT License. See [LICENSE](LICENSE).

Copyright (c) 2026 Farnoush Joulaeian.
