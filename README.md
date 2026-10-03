# MicroTracker

MicroTracker is a semi-automatic single-cell tracking application for fluorescence microscopy videos. It combines simple distance-based automatic linking with manual supervision and trajectory correction through a PyQt graphical interface.

The software records measured centroids of segmented objects. It does not insert predicted positions or interpolate missing frames.

## Features

- Interactive tracking of a selected fluorescent microorganism
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
2. Select a segmented bacterium manually.
3. Start automatic tracking from the selected object.
4. Monitor the trajectory and correct the identity manually when necessary.
5. Export the trajectory and session metadata.

The application starts in **MANUAL** mode.

Press **M** to switch between MANUAL and AUTO tracking.

AUTO selects the nearest eligible segmented object within the search radius. If no suitable object is found, tracking pauses until the user intervenes.

Manual selection identifies a segmented region; the measured centroid of that region is stored rather than the raw mouse-click coordinate.

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

Automatic tracking currently uses Euclidean distance only. It does not use motion prediction, directional persistence or automatic gap filling.

All segmented regions are retained so that rejected objects can still be inspected or selected manually.

### Default parameters

| Parameter | Default |
|---|---:|
| Gaussian kernel | 3 × 3 px |
| Gaussian sigma | 3 px |
| Intensity threshold | 40 / 255 |
| Accepted area | 6–150 px² |
| Search radius | 20 px |

These values are starting parameters and should be adjusted according to image quality and acquisition conditions.

## Measurements

Trajectory positions correspond to centroids of segmented fluorescent regions.

- Position is currently stored in pixels.
- Region area is measured in pixels².
- Physical spatial calibration is not yet applied.
- Nominal time is calculated from frame number and video FPS.
- Missing measurements remain missing and are not interpolated.

The measured fluorescent footprint depends on imaging conditions, thresholding, focus and object overlap. It should therefore not automatically be interpreted as the physical size of the bacterium.

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

- Cell identity still requires human supervision.
- Closely overlapping fluorescent objects may be segmented as a single region.
- Fixed-threshold segmentation may require adjustment between datasets.
- Spatial calibration is not yet included in exported measurements.
- Biological tracking accuracy has not yet been validated against independently annotated trajectories.
- Image processing currently runs synchronously with the GUI and may become slow for large videos.
- MicroTracker currently tracks a selected individual object rather than an entire population.

## License

MIT License.

Copyright © 2026 Farnoush Joulaeian.
