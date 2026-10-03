"""Compare masks from the saved preprocessing experiment; do not track cells."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage.segmentation import find_boundaries

from segmentation import threshold_mask, measure_regions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/preprocessing"))
    parser.add_argument("--output", type=Path, default=Path("outputs/segmentation"))
    parser.add_argument("--thresholds", type=float, nargs=3, default=[40, 60, 80],
                        help="Three fixed thresholds in blurred grayscale units, 0..255")
    args = parser.parse_args()
    source_settings = json.loads((args.input / "settings.json").read_text())
    samples = source_settings["sample_locations"]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    fig, axes = plt.subplots(len(samples), 6, figsize=(19, 11), constrained_layout=True)
    for row, (frame, x, y) in enumerate(samples):
        with np.load(args.input / f"frame_{frame:04d}.npz") as data:
            gray = data["gray"]
            h, w = gray.shape
            xs = slice(max(0, x-35), min(w, x+36))
            ys = slice(max(0, y-35), min(h, y+36))
            extent = (xs.start, xs.stop, ys.stop, ys.start)
            axes[row, 0].imshow(gray[ys, xs], cmap="gray", vmin=0, vmax=255,
                                interpolation="nearest", extent=extent)
            axes[row, 0].set_title("Raw grayscale")
            axes[row, 0].set_ylabel(f"Frame {frame}\ny (pixels)")
            methods = [("PI defaults", data["pi_processed"], 30, 0)]
            methods += [(f"Gaussian, T={t:g}", data["pi_blurred"], t, 0) for t in args.thresholds]
            methods += [("V1 full-frame Otsu", data["v1_corrected"], None, 1)]
            masks = {}
            for column, (name, image, threshold, closing) in enumerate(methods, start=1):
                mask, used = threshold_mask(image, threshold, closing)
                labels, regions = measure_regions(mask)
                # Nearest component is descriptive only, not validated identity or a tracker.
                distances = [np.hypot(r["x_px"]-x, r["y_px"]-y) for r in regions]
                within = [(r, d) for r, d in zip(regions, distances) if d <= 20]
                nearest, distance = min(within, key=lambda pair: pair[1]) if within else (None, None)
                crop_mask = mask[ys, xs]
                boundary = find_boundaries(labels[ys, xs], connectivity=2, mode="inner")
                rgb = np.repeat(gray[ys, xs, None].astype(float)/255, 3, axis=2)
                rgb[crop_mask] = 0.65 * rgb[crop_mask] + 0.35 * np.array([0, 1, 1])
                rgb[boundary] = [1, 0.4, 0]
                ax = axes[row, column]
                ax.imshow(rgb, interpolation="nearest", extent=extent)
                if nearest:
                    ax.plot(nearest["x_px"], nearest["y_px"], "+", color="yellow", markersize=7)
                area = nearest["area_px"] if nearest else None
                ax.set_title(f"{name}\nnearest area: {area if area is not None else 'none'}")
                rows.append({"frame": frame, "method": name, "threshold_used": used,
                             "threshold_units": "normalized intensity" if threshold is None else "0..255",
                             "closing_radius_px": closing, "full_frame_regions": len(regions),
                             "regions_with_centroid_within_20px": len(within),
                             "nearest_area_px": area, "nearest_distance_px": distance,
                             "nearest_x_px": nearest["x_px"] if nearest else None,
                             "nearest_y_px": nearest["y_px"] if nearest else None,
                             "foreground_fraction": float(mask.mean())})
                masks[f"method_{column}"] = mask
            np.savez_compressed(args.output / f"masks_{frame:04d}.npz", **masks)
        for ax in axes[row]:
            ax.set_xlabel("x (pixels)")
    fig.suptitle("All regions before area filtering | cyan: foreground | orange: boundary | yellow: nearest centroid", fontsize=13)
    fig.savefig(args.output / "comparison.png", dpi=130)
    plt.close(fig)
    with (args.output / "measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    settings = {
        "source_preprocessing": source_settings,
        "fixed_thresholds": args.thresholds,
        "pi_threshold": 30,
        "v1_threshold": "Otsu on full-frame normalized background residual",
        "v1_closing_radius_px": 1,
        "other_closing_radius_px": 0,
        "mask_archive_order": [name for name, *_ in methods],
        "connectivity": 8, "area_measurement": "foreground pixel count for every method",
        "area_filter": "none", "centroid_search_radius_px": 20,
        "scope": "full-frame segmentation; comparison crops for display only",
        "limitation": "Nearest regions are inspection candidates, not validated target identities. PI contour rejection is not applied.",
        "versions": {"numpy": np.__version__, "skimage": skimage.__version__, "matplotlib": matplotlib.__version__},
    }
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2))
    for item in rows:
        print(f"Frame {item['frame']:3d} | {item['method']:24s} | near regions {item['regions_with_centroid_within_20px']} | area {item['nearest_area_px']}")


if __name__ == "__main__":
    main()
