# MicroTracker

MicroTracker is a semi-automatic single-cell tracking application for a microscopy videos. It combines simple distance-based automatic linking with manual supervision and trajectory correction through a PyQt graphical interface.

The software records the user's clicked coordinates for manual points and contour centroids for AUTO points. When contour area is zero, AUTO uses the foreground-pixel centroid instead. It does not insert predicted positions or interpolate missing frames.

## Features

- Interactive tracking of a selected microorganism
- Manual selection and correction of cell identity
- Automatic nearest-neighbour linking within a configurable search radius
- Configurable segmentation and area filtering
- Frame-by-frame trajectory inspection
- Export of measurements and session metadata
- Automated software tests for core tracking behavior

## Installation

Requires Python 3.10 or newer. Development and testing were performed with Python 3.11.

Clone the repository:

```bash
git clone https://github.com/farnooshjoulaeian-dev/Microtracker_V1.git
cd Microtracker_V1
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

macOS or Linux:

```bash  
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

Start the application with:

```bash
python run.py
```

A video can also be supplied directly:

```bash
python run.py /path/to/video.mp4
```

Video files are not included in the repository.

## Basic workflow

1. Load a microscopy video.
2. Click the bacterium to record its position.
3. Start automatic tracking from the selected object.
4. Monitor the trajectory and correct the identity manually when necessary.
5. Export the trajectory and session metadata.

The application starts in **MANUAL** mode.

Press **M** to switch between MANUAL and AUTO tracking.

AUTO selects the nearest eligible object within the search radius. If none is found, playback continues without adding a measurement, and subsequent frames are searched from the last successful position. Missing frames remain missing.

The user can pause with **Space** or switch to **MANUAL** mode with **M** for correction.
## Controls

| Control | Action |
|---|---|
| `M` | Switch between MANUAL and AUTO |
| `Space` | Pause or resume playback |
| `Left / Right` | Navigate frames |
| Frame slider/input | Seek to a frame |
| Mouse wheel | Zoom |
| `Fit` | Restore full-frame view |
| `Undo` | Restore the previous trajectory state |

## Detection and tracking

The current processing pipeline is:

```text
Video frame
    ↓
Grayscale conversion
    ↓
Gaussian smoothing
    ↓
Clip intensities to [vmin, vmax] and rescale to 0–255
    ↓
Intensity thresholding
    ↓
Connected-component segmentation
    ↓
Area filtering
    ↓
Nearest-neighbour linking
    ↓
Trajectory
```

Thresholding uses the contrast-stretched image. After smoothing, intensities
at or below `vmin` become 0, and intensities at or above `vmax` become 255.
Values between them are linearly rescaled to 0–255. The limits must satisfy
`0 ≤ vmin < vmax ≤ 255`. The grayscale display keeps the original intensities.

Parameters can be changed while the video plays. Detection updates immediately,
and subsequent AUTO tracking uses the new settings. Existing trajectory points
keep their original measurements and settings.

Playback FPS controls viewing speed only. Measurement timestamps use the source
video's reported FPS, assuming a constant frame rate. If source FPS is unavailable,
timestamps are left blank.

Automatic tracking currently uses Euclidean distance only. It does not use motion prediction, directional persistence or automatic gap filling.

All segmented regions are retained so that rejected objects can still be inspected or selected manually.

### Default parameters

| Parameter | Default |
|---|---:|
| Gaussian sigma | 3 px (GUI range 0–7) |
| Gaussian truncation | 3 sigma |
| vmin | 0 |
| vmax | 255 |
| Intensity threshold | 40 / 255 |
| Accepted area | 6–150 px² |
| Search radius | 20 px |

These values are starting parameters and should be adjusted according to image quality and acquisition conditions. The default vmin/vmax limits preserve the intensity range.

Smoothing uses `skimage.filters.gaussian` with preserved 0–255 intensity range,
mirror boundaries and a fixed truncation of 3 sigma. Sigma 0 disables smoothing.
Kernel extent is determined automatically: sigma 3 uses approximately a 19 × 19
neighborhood, rather than the previous 3 × 3 neighborhood. Review segmentation
after this change because the smoothing strength and measured footprints differ.

## Position measurements

### Manual points

Manual mode records the user's clicked image coordinates directly.
A segmented object is not required.

Area and region label are unmeasured and exported as blank fields.
The user is responsible for placing the point on the correct cell.

### AUTO points

AUTO uses the geometric centroid of a region's outer contours.
Internal holes are treated as filled for this position calculation.

When contours enclose zero area, such as for a single pixel or thin line,
AUTO uses the foreground-pixel centroid instead.

Coordinates retain fractional pixel precision. Region area remains the
foreground pixel count, rather than contour-enclosed area.

The `position_method` column identifies manual clicks, contour centroids
and pixel-centroid fallbacks.

### Missed detections

AUTO continues playback when no eligible candidate is found, without
adding a measurement for that frame. It keeps searching from the last
successful position using the unchanged search radius.

Space pauses or resumes playback. M enters manual correction.
Missing frames remain missing; no predicted positions are inserted.

## Output and reproducibility

Saving a trajectory produces:

```text
trajectory.csv
trajectory.session.json
```

The CSV contains trajectory measurements and the processing parameters associated with each point.

The session JSON stores information required to interpret and reproduce the analysis, including:

- source-video metadata
- software dependencies
- tracking settings
- correction policy
- diagnostic events

Changing parameters during a session does not silently modify measurements that were already recorded.

The original microscopy video should be preserved together with the exported analysis files.

## Project structure

```text
Microtracker_V1/
│
├── detection.py
├── segmentation.py
├── linking.py
├── trajectory.py
├── video.py
├── main_window.py
├── run.py
├── requirements.txt
│
└── tests/
```

| File | Responsibility |
|---|---|
| `detection.py` | Detection settings and preprocessing |
| `segmentation.py` | Segmentation and region measurements |
| `linking.py` | Automatic linking and manual region selection |
| `trajectory.py` | Trajectory storage, correction and export |
| `video.py` | Video decoding and metadata |
| `main_window.py` | GUI, playback and visualization |
| `run.py` | Application entry point |
| `tests/` | Automated software tests |

The scientific processing modules are kept separate from the Qt interface.

## Testing

Run the test suite from the repository root:

```bash
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
```

The tests cover core software behavior including segmentation, area filtering, manual selection, distance-based linking, trajectory correction, export, video access and GUI state handling.

Passing the software tests verifies that the implemented functions behave according to their specifications. It does **not** by itself establish biological tracking accuracy or guarantee correct cell identity throughout a trajectory.

## Current limitations



## License

MIT License.

Copyright © 2026 Farnoush Joulaeian.