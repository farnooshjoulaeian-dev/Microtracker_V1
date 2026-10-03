# Runnable V0: algorithms, design and validation

## Choices carried forward from the two reference projects

From the PI tracker: simple Gaussian smoothing, an explicit fixed threshold,
interactive parameter control, segmentation preview and zoom. Unlike that tracker,
no fitted motion model or predicted points are used; smoothness must not conceal
missing measurements.

From MicroTracker V1: connected-component label selection independent of area
filtering, floating centroids, one trajectory with frame-indexed manual/AUTO points,
and distance-only bacterial linking. A fixed full-frame threshold avoids the
crop-dependent Otsu changes observed in V0/V1 and measures full regions rather
than clipping them at a tracking crop boundary.

## Scientific algorithm

Full-frame Gaussian smoothing reduces local noise. A fixed grayscale threshold
defines foreground; 8-connected components define candidate objects. Each region
is measured before area rejection. AUTO selects the nearest accepted centroid
within the radius of the previous measured position. The radius is in pixels,
not micrometres or pixels squared. Size limits are foreground pixel counts.

Initial settings are threshold 40, 3x3 blur kernel, sigma 3 pixels, area 6..150
pixels squared, and distance radius 20 pixels. The GUI exposes these explicitly.
Display clipping/rescaling is separate and never changes segmentation.

Advantages: readable, fast, reproducible, no hidden frame-varying threshold, and
manual area override. Limitations: a fixed threshold can miss dim objects as
fluorescence changes; blur/thresholding can merge neighbors; area cutoffs remain
hard constraints; nearest distance does not guarantee identity. The user monitors
identity and decides when to correct. No automatic ambiguity-stop policy is added.

## Interaction and trajectory contract

- Start in MANUAL, paused. Click identifies a label and stores its centroid.
- Track or M starts AUTO from the current selected/stored frame into the next.
- Space pauses/resumes without changing mode.
- M in AUTO pauses and enters MANUAL, even if AUTO already paused itself.
- No valid AUTO candidate pauses with a diagnostic, retaining AUTO mode.
- Navigation alone never creates trajectory points; manual playback adds none.
- Correcting a frame replaces it. The visible correction policy defaults to
  removing later points, with Undo; keeping later points is selectable explicitly.
- Retracking stores one point per frame and overwrites existing points there.
- Gaps remain missing and are not drawn as measured continuous segments.
- Measurement and settings history is exported with point provenance.

## Architecture and speed

Scientific algorithms are independent of Qt. Video handles decoding; Trajectory
handles points/export; MainWindow handles user state, scheduling and visualization.
Sequential playback avoids seeking and rereading the same frame for overlays.
Detection is cached for the current frame/settings. A single-shot timer accounts
for processing time before scheduling the next frame, without skipping scientific
measurements to catch up.

Content-driven diagnostic-table resizing caused timing spikes in an initial GUI
run. Fixed column widths and batched updates removed these in a follow-up run:
74 real-video advances took median 15.48 ms and 95th percentile 23.13 ms, including
decoding, diagnostics, rendering and headless event processing. This does not
guarantee native desktop timing. Processing is synchronous for this first version;
larger videos may need a worker after profiling.

## Validation completed

27 tests passed: segmentation geometry and edge/connectivity definitions; inclusive
area limits and preserved rejected labels; manual override and click bounds;
distance linking and failure reasons; correction/Undo and point uniqueness;
CSV/JSON metadata; video seeking/unknown FPS/decode failure; Qt AUTO/MANUAL states,
failure/recovery, keyboard focus and display/detection separation.

An actual distance-only run on flour_bacteria_test1.mp4, starting at frame 0 with
seed click (615,502), stored 195 points through frame 194. It passed the earlier
failure frames 41, 54, 61 and 74. At frame 195, a nearby 183-pixel region was
rejected by maximum area 150. The mask appears to connect adjacent fluorescent
objects; identity and merge behavior require user review.

The actual GUI reproduced this stop while retaining AUTO mode. A test then used
M, selected the rejected region manually, changed maximum area to 250 and resumed
AUTO successfully at frame 196. This verifies mechanics only; raising the maximum
to admit an apparent merged region is not a scientific recommendation.

CSV output was checked for unique frame indices, settings and source labels.
JSON output was checked for unset spatial calibration, current settings, failure
candidate diagnostics and absence of prediction. The application launcher/help
and headless widget rendering were checked. Native desktop smoothness and cell
identity accuracy remain user validation tasks.

## Deliberately unresolved scientific decisions

No true cell boundary is inferred from the fluorescent threshold mask. No
physical calibration is guessed. The video-reported FPS supplies nominal time
assuming constant frame rate. Further annotated review should include other cells,
dim/focus changes and merges before choosing validated defaults or publishing
biological measurements.
