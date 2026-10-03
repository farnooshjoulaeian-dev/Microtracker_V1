"""Run actual distance linking from one seed; stop at the first missing match.

No interpolation, prediction, or manual corrections are inserted into this run.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skimage

from detection import DetectionSettings, detect_frame
from linking import link_nearest, select_region_at_click
from trajectory import Trajectory
from video import Video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("../flour_bacteria_test1.mp4"))
    parser.add_argument("--output", type=Path, default=Path("outputs/linking"))
    parser.add_argument("--threshold", type=int, default=40)
    parser.add_argument("--radius", type=float, default=20)
    parser.add_argument("--seed", type=float, nargs=2, default=[615, 502])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = DetectionSettings(threshold_gray=args.threshold)
    video = Video(args.video)
    traj = Trajectory()
    elapsed, snapshots = [], []
    stop = {"reason": "end_of_video", "frame": video.n_frames-1}
    try:
        frame = video.read(0)
        result = detect_frame(frame, settings)
        seed = select_region_at_click(result.labels, result.regions, *args.seed)
        if seed is None:
            raise ValueError("No segmented object near the seed click")
        previous = traj.record(0, seed, "manual", settings, args.radius, time_s=0 if video.fps else None)
        for number in range(1, video.n_frames):
            frame = video.read(number)
            start = perf_counter()
            result = detect_frame(frame, settings)
            linked = link_nearest(result.regions, (previous["x_px"], previous["y_px"]), args.radius)
            elapsed.append(1000*(perf_counter()-start))
            if linked.selected is None:
                stop = {"reason": linked.reason, "frame": number,
                        "previous_position_px": [previous["x_px"], previous["y_px"]],
                        "nearest_candidates": linked.candidates[:10]}
            else:
                previous = traj.record(number, linked.selected, "auto", settings, args.radius,
                                       time_s=number/video.fps if video.fps else None)
            if number in (20, 41, 54, 61, 74, 106, 141, 200, 290) or number == video.n_frames-1 or linked.selected is None:
                x, y = round(previous["x_px"]), round(previous["y_px"])
                h, w = frame.shape[:2]
                xs, ys = slice(max(0, x-30), min(w, x+31)), slice(max(0, y-30), min(h, y+31))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                contours, _ = cv2.findContours(result.mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, contours, -1, (0, 220, 220), 1)
                cv2.drawMarker(rgb, (x, y), (255, 255, 0), cv2.MARKER_CROSS, 7, 1)
                snapshots.append((number, rgb[ys, xs].copy(), linked.reason))
            if linked.selected is None:
                break
        metadata = {"software": "FluorescenceTracker", "version": "0.1.0", "video": video.metadata(),
                    "settings": asdict(settings), "seed_click_px": args.seed,
                    "search_radius_px": args.radius, "linking": "nearest area-accepted centroid, distance only",
                    "prediction": "none", "stop": stop,
                    "median_detection_and_link_ms": float(np.median(elapsed)) if elapsed else None,
                    "limitation": "No independently annotated identities; uninterrupted linking is not proof of correct identity.",
                    "dependencies": {"opencv": cv2.__version__, "numpy": np.__version__, "skimage": skimage.__version__}}
        traj.save(args.output/"trajectory.csv", metadata)
        if not snapshots:
            x, y = round(previous["x_px"]), round(previous["y_px"])
            h, w = frame.shape[:2]
            snapshots.append((0, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[max(0,y-30):min(h,y+31),max(0,x-30):min(w,x+31)], "manual seed"))
        fig, axes = plt.subplots(2, (len(snapshots)+1)//2, figsize=(15, 6), squeeze=False, constrained_layout=True)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (number, image, reason) in zip(axes.flat, snapshots):
            ax.imshow(image, interpolation="nearest")
            ax.set_title(f"Frame {number}: {reason}")
        fig.suptitle("Actual distance-only links | cyan: mask boundary | yellow: linked centroid or last accepted position")
        fig.savefig(args.output/"snapshots.png", dpi=140)
        plt.close(fig)
        print(json.dumps({"points": len(traj.points), "stop": stop,
                          "median_detection_and_link_ms": metadata["median_detection_and_link_ms"]}, indent=2))
    finally:
        video.close()


if __name__ == "__main__":
    main()
