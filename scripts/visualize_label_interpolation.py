"""
Visualise grid‑based label interpolation in the updated DSECDet.

For every requested sample index we step through the event span in chunks
of --time_step_us micro‑seconds, accumulate events of that slice, render them
together with the interpolated boxes (LEFT) and, for comparison, render the
original ground‑truth boxes on the raw image (RIGHT).

A single PNG file is written for every slice:
    <out_dir>/<sample_idx>_<slice_idx>.png
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import hdf5plugin              # noqa: F401  (required by h5py back‑end)
import matplotlib.pyplot as plt

from dsec_det.dataset import (
    DSECDet,
    render_events_on_image,
    render_object_detections_on_image,
)


def to_gamma_corrected(img):
    return (255 * (img.astype("float32") / 255) ** (1.0 / 2.2)).astype("uint8")


def _slice_events(events, t_start, t_end):
    """Return a dict that contains only the events inside [t_start,t_end)."""
    m = (events["t"] >= t_start) & (events["t"] < t_end)
    return {k: v[m] for k, v in events.items()}


def _slice_tracks(tracks, t_start, t_end):
    """Structured array rows whose 't' field is inside [t_start,t_end)."""
    if "t" not in tracks.dtype.names:
        # non‑interpolated → take everything (they live on the key‑frame)
        return tracks
    m = (tracks["t"] >= t_start) & (tracks["t"] < t_end)
    return tracks[m]



def visualise_interpolation(
    dsec_root: Path,
    split: str,
    start_idx: int,
    end_idx: int,
    time_step_us: int,
    sync: str,
    out_dir: Path = None,
):
    ds_interp = DSECDet(
        dsec_root,
        split=split,
        sync=sync,
        interpolate_labels=True,
        interp_step_us=time_step_us,
        singleton_policy="internal",
    )
    ds_orig = DSECDet(dsec_root, split=split, sync=sync, interpolate_labels=False, interp_step_us=time_step_us)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(start_idx, end_idx):
        try:
            img_rgb = to_gamma_corrected(ds_interp.get_image(idx).copy())
            events = ds_interp.get_events(idx)            # absolute µs
            t_min, t_max = events["t"][0], events["t"][-1]

            tr_interp = ds_interp.get_tracks(idx)
            tr_orig = ds_orig.get_tracks(idx)


            slice_id = 0
            t_cursor = int(t_min)
            while t_cursor < t_max:
                t_next = min(t_cursor + time_step_us, t_max)

                ev_slice = _slice_events(events, t_cursor, t_next)
                tr_slice = _slice_tracks(tr_interp, t_cursor, t_next)

                left = img_rgb.copy()
                if len(ev_slice["t"]):
                    left = render_events_on_image(
                        left, ev_slice["x"], ev_slice["y"], ev_slice["p"]
                    )
                if len(tr_slice):
                    left = render_object_detections_on_image(
                        left, tr_slice, label="interp"
                    )

                right = img_rgb.copy()
                if len(tr_orig):
                    right = render_object_detections_on_image(
                        right, tr_orig, label="orig"
                    )

                fig, axs = plt.subplots(
                    1,
                    2,
                    figsize=(2 * (left.shape[1] / 100), left.shape[0] / 100),
                    dpi=100,
                )
                axs[0].imshow(left[..., ::-1])
                axs[0].set_title(
                    f"interp boxes  [{(t_cursor-t_min)/1e3:.1f}‑{(t_next-t_min)/1e3:.1f} ms]",
                    fontsize=7,
                )
                axs[1].imshow(right[..., ::-1])
                axs[1].set_title("original GT boxes", fontsize=7)
                for ax in axs:
                    ax.axis("off")
                fig.subplots_adjust(
                    left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01
                )

                if out_dir is None:
                    plt.show()
                else:
                    out_path = (
                        out_dir / f"{idx:05d}_{slice_id:03d}.png"
                    )
                    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0)
                    print(f"[✓] {out_path.relative_to(out_dir.parent)}")
                    plt.close(fig)

                slice_id += 1
                t_cursor = t_next

        except Exception as exc:
            raise exc
            print(f"[!] sample {idx} failed: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualise label interpolation on DSEC")
    parser.add_argument("--dsec_root", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "test", "val"])
    parser.add_argument("--sync", default="front", choices=["front", "back"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument(
        "--time_step_us",
        type=int,
        default=10_000,
        help="Step between successive slices in µs (e.g. 10000 = 10ms)",
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    visualise_interpolation(
        args.dsec_root,
        args.split,
        args.start,
        args.end,
        args.time_step_us,
        args.sync,
        args.out_dir,
    )