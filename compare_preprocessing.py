"""Compare preprocessing only; no detection, area filtering, or linking.

Run: python compare_preprocessing.py --video ../flour_bacteria_test1.mp4
"""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import filters, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma


def pi_preprocessing(gray, blur=3, vmin=104, vmax=200):
    """Reproduce the PI's blur and fixed intensity-window transformation."""
    if vmax <= vmin:
        raise ValueError("vmax must exceed vmin")
    kernel = blur // 2 * 2 + 1
    blurred = cv2.GaussianBlur(gray, (kernel, kernel), blur)
    clipped = np.clip(blurred, vmin, vmax)
    scaled = (255 * (clipped.astype(float) - vmin) / (vmax - vmin)).astype(np.uint8)
    return scaled, blurred


def v1_preprocessing(gray):
    """Reproduce V1 defaults on the full frame, returning intermediate images."""
    normalized = img_as_float(gray)
    noise = float(estimate_sigma(normalized, channel_axis=None))
    denoised = denoise_nl_means(
        normalized, h=1.15 * noise, patch_size=3, patch_distance=3,
        fast_mode=True, channel_axis=None,
    )
    background = filters.gaussian(denoised, sigma=50, mode="nearest", preserve_range=True)
    residual = denoised - background
    return np.clip(residual, 0, None), background, noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/preprocessing"))
    parser.add_argument("--blur", type=int, default=3)
    parser.add_argument("--vmin", type=int, default=104)
    parser.add_argument("--vmax", type=int, default=200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    # Prior trajectory supplies approximate locations for inspection, not ground truth.
    samples = [(0, 615, 502), (41, 639, 546), (54, 646, 560), (61, 653, 568)]
    metadata = {
        "video": str(args.video.resolve()), "frame_indexing": "zero-based",
        "fps_reported": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "processing_scope": "full frame for both methods; crops are display only",
        "pi": {"blur": args.blur, "vmin": args.vmin, "vmax": args.vmax},
        "v1": {"h_factor": 1.15, "patch_size": 3, "patch_distance": 3, "background_sigma": 50},
        "versions": {"opencv": cv2.__version__, "numpy": np.__version__,
                     "skimage": skimage.__version__, "matplotlib": matplotlib.__version__},
        "display": "raw/PI: fixed 0..255; V1 residual: fixed 0..0.2 normalized intensity",
        "timing": "one call per method per frame; excludes decoding, plotting and GUI; not a benchmark",
        "sample_locations": samples,
    }
    fig, axes = plt.subplots(len(samples), 3, figsize=(10, 11), constrained_layout=True)
    rows = []
    try:
        for row, (number, x, y) in enumerate(samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, number)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read frame {number}")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            start = perf_counter()
            pi_image, blurred = pi_preprocessing(gray, args.blur, args.vmin, args.vmax)
            pi_ms = (perf_counter() - start) * 1000
            start = perf_counter()
            corrected, background, noise = v1_preprocessing(gray)
            v1_ms = (perf_counter() - start) * 1000
            h, w = gray.shape
            ys, xs = slice(max(0, y-35), min(h, y+36)), slice(max(0, x-35), min(w, x+36))
            for column, (image, high, title) in enumerate([
                (gray, 255, "Raw grayscale (0–255)"),
                (pi_image, 255, "PI: blur + intensity window (0–255)"),
                (corrected, 0.2, "V1: background-subtracted (0–0.2)"),
            ]):
                axes[row, column].imshow(image[ys, xs], cmap="gray", vmin=0, vmax=high,
                                         interpolation="nearest", extent=(xs.start, xs.stop, ys.stop, ys.start))
                axes[row, column].set_title(title if row == 0 else f"Frame {number}")
                axes[row, column].set_xlabel("x (pixels)")
                axes[row, column].set_ylabel(f"Frame {number}\ny (pixels)")
            rows.append({"frame": number, "pi_ms": pi_ms, "v1_ms": v1_ms,
                         "v1_noise_estimate": noise,
                         "pi_fraction_clipped_low": float(np.mean(blurred <= args.vmin)),
                         "pi_fraction_clipped_high": float(np.mean(blurred >= args.vmax)),
                         "pi_target_crop_max": int(pi_image[ys, xs].max())})
            np.savez_compressed(args.output / f"frame_{number:04d}.npz",
                                gray=gray, pi_processed=pi_image, pi_blurred=blurred,
                                v1_corrected=corrected, v1_background=background)
    finally:
        cap.release()
    fig.savefig(args.output / "comparison.png", dpi=140)
    plt.close(fig)
    (args.output / "settings.json").write_text(json.dumps(metadata, indent=2))
    with (args.output / "measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
