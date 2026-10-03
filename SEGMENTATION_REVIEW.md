# Step 2: segmentation and measurements

## What was compared

Reuse the four full-frame preprocessing arrays from step 1. Compare the PI's
default processed-image threshold 30; thresholds 40, 60 and 80 on its Gaussian-
smoothed grayscale image; and V1's full-frame Otsu threshold and radius-1 closing.
The thresholds 40/60/80 are exploratory settings spanning the target intensity,
not validated or automatically optimized values. They are the same across frames.

For the fixed-threshold experiment, bypass PI clipping/rescaling and threshold
the blurred image directly. This separates display contrast from detection and
states the threshold in original grayscale units (0..255 for this decoded video).

Every mask is measured using 8-connected components and foreground pixel counts,
with no area rejection. This isolates mask behavior and gives comparable units.
It does NOT reproduce the PI contour-size filter or its minimum contour-vertex
requirement. Contour area and foreground pixel count are different measurements.

The inspection point comes from the earlier trajectory, not independent annotation.
The table reports the nearest component centroid within 20 pixels. This is a
descriptive candidate measurement, not evidence of correct identity or a linker.

## Results: nearest component area, in pixels squared

| Frame | PI defaults | Gaussian T=40 | Gaussian T=60 | Gaussian T=80 | V1 full-frame Otsu |
|---|---:|---:|---:|---:|---:|
| 0 | 32 | 112 | 75 | 56 | 78 |
| 41 | none | 117 | 61 | 22 | 89 |
| 54 | none | 107 | 46 | 17 | 72 |
| 61 | none | 95 | 32 | 4 | 66 |

All three Gaussian thresholds produce a nearby component on these four frames.
T=80 reduces the dim frame-61 target to four pixels, which would fail a minimum
area of six. T=40 retains a broader region but exceeds a maximum area of 100 at
frames 0, 41 and 54. Acceptance rules must therefore be evaluated together with
the segmentation settings, rather than treated as biological size constants.

T=60 preserves the bright center in these examples. Its area changes from 75 to
32 pixels across frames, and the frame-61 centroid is about one pixel from the
inspection point. This is not a measured localization error because the inspection
point is not ground truth. T=40 captures a broader fluorescent footprint; without
annotations we cannot claim either boundary is the true physical cell boundary.

V1 Otsu here operates on the whole frame. Its areas differ from the previous
crop-based AUTO diagnosis (e.g. 72 rather than 111 at frame 54). Background
estimation, noise estimation and the Otsu histogram change when processing a crop.
These comparison values must not be presented as current crop-tracker outputs.

The V1 column also includes closing, so the five columns compare complete mask
recipes, not solely threshold algorithms. No separate scientific benefit of
closing has been established in this experiment.

## Proposed initial direction, still provisional

Keep the first detection prototype simple: Gaussian smoothing, an explicit fixed
threshold, and connected-component centroid/area measurements. Connected components
give a label image usable for both manual selection and AUTO candidate inspection.
Keep display contrast separate from segmentation and report size rejections.

Use T=60 as an exploratory starting point, adjustable by the user. T=40 is also
worth inspecting if a broader footprint is desired. Four frames are insufficient
to choose final defaults, establish robustness over the video, or compare false
positives. Next evaluate a longer contiguous interval and brighter/dimmer objects
before committing to the detector. No linking, motion prediction, or GUI is added.

## Files and reproduction

segmentation.py provides two small functions: threshold_mask and measure_regions.
Neither depends on video files or Qt. compare_segmentation.py handles saved input,
plots and measurements separately.

```sh
python compare_segmentation.py
python -m unittest test_segmentation
```

Run from the project directory after step 1. Parameters and source settings are
stored in outputs/segmentation/settings.json; numerical results in measurements.csv;
all full-frame masks in masks_XXXX.npz. The mask archive's column order is recorded.
Use --thresholds 40 60 80 and --output PATH to make alternative runs explicit.

Five tests check area and fractional centroid on a known rectangle, threshold
equality, diagonal connectivity, boundary flagging, blank-frame behavior and
invalid inputs. These test measurement definitions, not biological accuracy.
