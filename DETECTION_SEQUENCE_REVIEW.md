# Step 3: a simple detector over frames 0..74

## Detector contract

detection.py composes Gaussian smoothing, thresholding and connected components.
It accepts uint8 grayscale or BGR images; it does not read videos or control Qt.
DetectionSettings makes every processing parameter explicit. Values are prototype
starting points, not final defaults established for all cells or experiments.

The smoothing reproduces the PI default recipe: a 3x3 kernel with sigma 3 pixels.
Kernel size limits the support; this is not equivalent to a wide sigma-3 kernel.
No clipping, intensity scaling, background subtraction or closing is applied.
The threshold acts directly on blurred grayscale values in the range 0..255.

Every connected region is retained with a frame-local label, floating centroid,
foreground pixel area, image-edge flag and explicit area status. The inclusive
area interval is 6..150 pixels squared in this experiment. Rejected objects are
still present in the label image, allowing later manual selection to override
automatic restrictions. No physical units are reported without calibration.

## Sequence review and limits

The experiment decodes frames 0..74 sequentially and applies thresholds 40, 60
and 80 to every full frame. The user's previous manual positions define a
piecewise linear inspection guide. It is not tracking, an automatically predicted
trajectory, or independent ground truth. A centroid within 10 pixels of that
guide is reported as a nearby detection candidate.

The guide can deviate from the cell path (e.g. frame 20). It therefore cannot
be used to measure localization accuracy. No false-positive or identity accuracy
is claimed. All full-frame measured regions are saved for later inspection.

| Threshold | Frames with nearest region area-accepted | Nearby measured area range | Median detector time |
|---|---:|---:|---:|
| 40 | 75/75 | 72..124 pixels² | 6.27 ms |
| 60 | 75/75 | 16..78 pixels² | 5.70 ms |
| 80 | 61/75 | 1..56 pixels² | 5.33 ms |

At T=80, seven frames have a region below minimum area and seven have no region
whose centroid lies within the inspection radius. No frame has multiple accepted
regions within that radius. This count does not establish uniqueness within a
later tracker's larger search radius or throughout the full image.

Timing includes BGR conversion, blur, mask generation, labeling, measurements and
area diagnostics; excludes decoding, plotting and GUI/display. Repeated threshold
calls use a fixed order and shared frames, so these are exploratory timings.
They do not establish end-to-end playback speed or responsiveness.

## Interpretation

Simple segmentation retains nearby candidates across this initial interval without
NLM or background subtraction. T=60 increasingly captures only the brightest core
as the target becomes dimmer; at frame 74 its region has 16 pixels, versus 75 at
T=40. T=40 appears to preserve a broader fluorescent footprint in the inspected
snapshots and deserves consideration for an adjustable prototype. Neither mask
is an established physical cell boundary. Do not infer cell growth or shrinking
from these threshold-dependent areas.

Retain both settings as explicit options rather than silently select a final
threshold. Test brighter and dimmer cells and a longer interval before claiming
general robustness. The current area interval is not a biological size prior.

## Verification and reproduction

```sh
python review_detection_sequence.py
python -m unittest discover -p 'test_*.py'
```

Run from the new project folder. The script defaults to the reference video next
to the project and stores results in outputs/detection_sequence. --video,
--output, --min-area and --max-area make alternate runs explicit.

Ten unit tests passed across the detector and segmentation modules. Tests check
known geometry, area rejection without label deletion, inclusive bounds, blank
frames, input validation, grayscale/BGR agreement and connectivity. Tests verify
software definitions, not biological identity.

Outputs include sequence.png, snapshots.png, availability.csv, all measured
regions.csv, timings.csv and settings.json with parameters, anchors and versions.

## Next component

Implement distance-only linking using the previous manually/automatically measured
position, without the inspection guide. Compare its actual trajectory and stop
reasons. Never substitute predicted positions for measurements. The user retains
responsibility for identity review and changing AUTO/MANUAL mode.
