"""Review detection availability over a sequence; this is not a tracker."""
import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage.segmentation import find_boundaries

from detection import DetectionSettings, detect_frame


# Positions supplied by the user's previous manual selection logs.
# Interpolation is an inspection guide, not ground truth or tracking prediction.
ANCHORS = [(0, 614.9359, 502.3974), (41, 638.8, 546.1),
           (51, 641.7160, 555.0123), (57, 652.0694, 562.8889),
           (61, 652.6667, 568.2121), (62, 653.6944, 570.3333),
           (72, 651.2609, 578.9855), (74, 648.6, 581.2)]


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("../flour_bacteria_test1.mp4"))
    parser.add_argument("--output", type=Path, default=Path("outputs/detection_sequence"))
    parser.add_argument("--min-area", type=int, default=6)
    parser.add_argument("--max-area", type=int, default=150)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = DetectionSettings(min_area_px=args.min_area, max_area_px=args.max_area)
    thresholds = [40, 60, 80]
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows, all_regions, timings = [], [], []
    snapshots = [0, 20, 41, 54, 61, 74]
    fig, axes = plt.subplots(len(snapshots), 3, figsize=(11, 17), constrained_layout=True)
    anchors = np.array(ANCHORS)
    try:
        for frame_number in range(75):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read frame {frame_number}")
            x_ref = float(np.interp(frame_number, anchors[:, 0], anchors[:, 1]))
            y_ref = float(np.interp(frame_number, anchors[:, 0], anchors[:, 2]))
            for column, threshold in enumerate(thresholds):
                settings = replace(base, threshold_gray=threshold)
                start = perf_counter()
                result = detect_frame(frame, settings)
                elapsed = 1000 * (perf_counter() - start)
                timings.append({"frame": frame_number, "threshold": threshold, "detection_ms": elapsed})
                nearby = []
                for region in result.regions:
                    distance = float(np.hypot(region["x_px"]-x_ref, region["y_px"]-y_ref))
                    all_regions.append({"frame": frame_number, "threshold": threshold,
                                        **region, "distance_to_inspection_reference_px": distance})
                    if distance <= 10:
                        nearby.append((region, distance))
                nearest, distance = min(nearby, key=lambda pair: pair[1]) if nearby else (None, None)
                status = nearest["area_status"] if nearest else "no_region_within_10px"
                rows.append({"frame": frame_number, "threshold": threshold,
                             "reference_x_px": x_ref, "reference_y_px": y_ref,
                             "nearby_raw_regions": len(nearby),
                             "nearby_accepted_regions": sum(r["accepted_for_auto"] for r, _ in nearby),
                             "nearest_area_px": nearest["area_px"] if nearest else None,
                             "nearest_distance_px": distance, "nearest_status": status})
                if frame_number in snapshots:
                    row = snapshots.index(frame_number)
                    h, w = result.gray.shape
                    xs = slice(max(0, int(x_ref)-30), min(w, int(x_ref)+31))
                    ys = slice(max(0, int(y_ref)-30), min(h, int(y_ref)+31))
                    rgb = np.repeat(result.gray[ys, xs, None].astype(float)/255, 3, axis=2)
                    boundaries = find_boundaries(result.labels[ys, xs], mode="inner")
                    rgb[boundaries] = [0, 1, 1]
                    ax = axes[row, column]
                    ax.imshow(rgb, interpolation="nearest", extent=(xs.start, xs.stop, ys.stop, ys.start))
                    ax.plot(x_ref, y_ref, "+", color="yellow")
                    ax.set_title(f"Frame {frame_number}, T={threshold}\n{status}, area={nearest['area_px'] if nearest else 'none'}")
                    ax.set_xlabel("x (pixels)"); ax.set_ylabel("y (pixels)")
    finally:
        cap.release()
    fig.suptitle("Cyan: segmentation boundary | yellow: approximate inspection reference")
    fig.savefig(args.output / "snapshots.png", dpi=120)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    summary = {}
    for threshold in thresholds:
        selected = [r for r in rows if r["threshold"] == threshold]
        areas = [r["nearest_area_px"] if r["nearest_area_px"] is not None else np.nan for r in selected]
        frames = [r["frame"] for r in selected]
        axes[0].plot(frames, areas, label=f"T={threshold}")
        axes[1].plot(frames, [r["nearby_accepted_regions"] for r in selected], label=f"T={threshold}")
        missing = [r["frame"] for r in selected if r["nearest_status"] != "accepted"]
        ms = [r["detection_ms"] for r in timings if r["threshold"] == threshold]
        summary[str(threshold)] = {"nearest_accepted_frames": 75-len(missing),
                                   "nearest_not_accepted_frames": missing,
                                   "frames_with_multiple_nearby_accepted_regions": [r["frame"] for r in selected if r["nearby_accepted_regions"] > 1],
                                   "median_detection_ms": float(np.median(ms)),
                                   "p95_detection_ms": float(np.percentile(ms, 95))}
    axes[0].axhline(base.min_area_px, color="gray", linestyle=":", label="area bounds")
    axes[0].axhline(base.max_area_px, color="gray", linestyle=":")
    axes[0].set_ylabel("Nearest region area (pixels²)"); axes[0].legend()
    axes[1].set_ylabel("Accepted regions within 10 px"); axes[1].set_xlabel("Frame index (zero-based)")
    axes[1].legend()
    fig.savefig(args.output / "sequence.png", dpi=140)
    plt.close(fig)
    write_csv(args.output / "availability.csv", rows)
    write_csv(args.output / "regions.csv", all_regions)
    write_csv(args.output / "timings.csv", timings)
    metadata = {"video": str(args.video.resolve()), "frames": [0, 74], "fps_reported": fps,
                "settings": asdict(base), "thresholds": thresholds, "inspection_anchors": ANCHORS,
                "inspection_radius_px": 10, "processing_scope": "full frame",
                "versions": {"opencv": cv2.__version__, "numpy": np.__version__, "skimage": skimage.__version__},
                "timing_scope": "one detection call per frame/threshold, excludes decoding, plotting and GUI; repeated order, not independent benchmark",
                "limitation": "Manual anchor interpolation is an approximate guide, not annotation, linking or proof of identity.",
                "summary": summary}
    (args.output / "settings.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
