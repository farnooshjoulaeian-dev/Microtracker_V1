# Step 1: preprocessing comparison

## Scope

Compare the PI defaults confirmed by the user with MicroTracker V1 defaults.
Both methods process the full frame; 71-pixel display crops show an approximate
target position from the previous trajectory. These positions are not annotated
ground truth. No tracking, segmentation, or size filtering is implemented here.

## Concepts and tradeoffs

Gaussian blur reduces local noise but can smear small objects or blend neighbors.
The PI then clips intensities below vmin and above vmax and maps that fixed window
to 0..255. This is fast and easy to adjust. It is not background subtraction:
a dim target below vmin becomes zero, even if it is distinguishable from its
local background in the original image.

V1 estimates noise, applies non-local means, estimates a slowly varying background
with a Gaussian of sigma 50 pixels, subtracts it, and clips negative residuals.
This retains more dim signal here but also retains background texture. It costs
more computation. The Gaussian background assumes background varies on scales
larger than the cell. Negative residuals represent fluctuations relative to the
estimated background; clipping is a processing choice, not a physical necessity.

## Observations

At frames 41 and 61 the PI-processed target inspection crop is entirely zero.
At frame 54 its maximum is 2/255, below the PI default threshold of 30.
Thus the defaults cannot segment this target there. This does not contradict
successful tracking of other, brighter cells. The PI algorithm can also append
predictions when detections disappear after a prediction buffer is available.

Across four frames, one preprocessing call took approximately 2..6 ms for the PI
method and 583..629 ms for V1 on this machine. These exclude decoding, detection,
plotting and GUI work. They are exploratory timings, not a controlled benchmark;
they do not estimate current crop-based V1 playback speed.

The comparison uses fixed display scales for each method, with different units.
The brighter appearance of the V1 column does not by itself establish better
segmentation or more accurate physical fluorescence measurements.

## Reproduce

From this project folder:

```sh
python compare_preprocessing.py --video ../flour_bacteria_test1.mp4
```

Outputs: comparison.png, settings.json, measurements.csv and intermediate .npz
arrays in outputs/preprocessing. Display crops do not affect processing.
Change PI settings explicitly with --blur, --vmin and --vmax; use a different
--output folder to preserve previous runs. Numpy, OpenCV, scikit-image and
Matplotlib are required; exact versions used are recorded in settings.json.

Both preprocessing functions were checked against their existing reference
formulas on a deterministic synthetic image. Invalid intensity bounds are rejected.

## Next incremental decision

Do not select a final algorithm from appearance alone. Next compare masks on
these frames: first assess whether simpler Gaussian smoothing with a suitable
fixed intensity/threshold setting retains the target. Add background subtraction
only if it addresses demonstrated illumination variation or segmentation errors.
Keep predictions out of measured trajectories and bacterial linking distance-only.
