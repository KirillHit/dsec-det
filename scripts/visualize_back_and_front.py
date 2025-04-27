"""
Visualise the new *back-and-front* sync mode.

For every window (sample) we build a 3-panel figure:

    • LEFT   – “back” image  (frame *i*) with its **original** boxes
    • MIDDLE – events accumulated over one --time_step_us slice,
               plus **interpolated** boxes valid inside that slice
    • RIGHT  – “front” image (frame *i+1*) with its **original** boxes

One PNG per slice is written to
    <out_dir>/<sample_idx>_<slice_idx>.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import hdf5plugin  # noqa: F401 (makes h5py load the Blosc plugin)
import matplotlib.pyplot as plt

from dsec_det.dataset import (
    DSECDet,
    render_events_on_image,
    render_object_detections_on_image,
)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def gamma(img):
    return (255 * (img.astype("float32") / 255) ** (1.0 / 2.2)).astype("uint8")


def slice_events(ev, t0, t1):
    m = (ev["t"] >= t0) & (ev["t"] < t1)
    return {k: v[m] for k, v in ev.items()}


def slice_tracks(tr, t0, t1):
    if "t" not in tr.dtype.names:
        return tr
    m = (tr["t"] >= t0) & (tr["t"] < t1)
    return tr[m]


# ----------------------------------------------------------------------
# main routine
# ----------------------------------------------------------------------
def visualise_back_and_front(
    dsec_root: Path,
    split: str,
    start_idx: int,
    end_idx: int,
    step_us: int,
    out_dir: Path,
):
    # dataset with interpolation (back & front)
    ds_intp = DSECDet(
        dsec_root,
        split=split,
        sync="back_and_front",
        interpolate_labels=True,
        interp_step_us=step_us,
        singleton_policy="internal",
    )

    # dataset for *raw* boxes on individual frames
    ds_raw = DSECDet(
        dsec_root,
        split=split,
        sync="back",              # one label-set per image
        interpolate_labels=False,
    )

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ensure we never request the very last window (has no front image)
    end_idx = min(end_idx, len(ds_intp) - 1)

    for idx in range(start_idx, end_idx):
        try:
            # ----------------------------------------------------------
            # load everything once per sample
            # ----------------------------------------------------------
            back_img = gamma(ds_intp.get_image(idx).copy())
            front_img = gamma(ds_intp.get_image(idx + 1).copy())

            back_gt = ds_raw.get_tracks(idx)        # boxes @ frame i
            front_gt = ds_raw.get_tracks(idx + 1)   # boxes @ frame i+1

            ev = ds_intp.get_events(idx)
            t_min, t_max = ev["t"][0], ev["t"][-1]

            tr_intp = ds_intp.get_tracks(idx)

            slice_id = 0
            t_cur = int(t_min)
            while t_cur < t_max:
                t_next = min(t_cur + step_us, t_max)

                ev_slice = slice_events(ev, t_cur, t_next)
                tr_slice = slice_tracks(tr_intp, t_cur, t_next)

                # ------------------------------------------------------
                # LEFT  – back image + GT boxes
                # ------------------------------------------------------
                left = back_img.copy()
                if len(back_gt):
                    left = render_object_detections_on_image(
                        left, back_gt, label="back_gt"
                    )

                # ------------------------------------------------------
                # MIDDLE – events + interpolated boxes in this slice
                # ------------------------------------------------------
                mid = back_img.copy()  # use back image as background
                if len(ev_slice["t"]):
                    mid = render_events_on_image(
                        mid, ev_slice["x"], ev_slice["y"], ev_slice["p"]
                    )
                if len(tr_slice):
                    mid = render_object_detections_on_image(
                        mid, tr_slice, label="interp"
                    )

                # ------------------------------------------------------
                # RIGHT – front image + GT boxes
                # ------------------------------------------------------
                right = front_img.copy()
                if len(front_gt):
                    right = render_object_detections_on_image(
                        right, front_gt, label="front_gt"
                    )

                # ------------------------------------------------------
                # compose figure
                # ------------------------------------------------------
                fig, axs = plt.subplots(
                    1,
                    3,
                    figsize=(3 * (left.shape[1] / 100), left.shape[0] / 100),
                    dpi=100,
                )
                axs[0].imshow(left[..., ::-1])
                axs[0].set_title("back frame (GT)", fontsize=7)
                axs[1].imshow(mid[..., ::-1])
                axs[1].set_title(
                    f"slice {slice_id}  [{(t_cur - t_min)/1e3:.1f}–{(t_next - t_min)/1e3:.1f} ms]",
                    fontsize=7,
                )
                axs[2].imshow(right[..., ::-1])
                axs[2].set_title("front frame (GT)", fontsize=7)

                for ax in axs:
                    ax.axis("off")
                fig.subplots_adjust(
                    left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01
                )

                if out_dir is None:
                    plt.show()
                else:
                    fname = out_dir / f"{idx:05d}_{slice_id:03d}.png"
                    fig.savefig(fname, bbox_inches="tight", pad_inches=0)
                    print(f"[✓] {fname.relative_to(out_dir.parent)}")
                    plt.close(fig)

                slice_id += 1
                t_cur = t_next

        except Exception as exc:
            print(f"[!] sample {idx} failed: {exc}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser("Visualise back-and-front sync on DSEC")
    p.add_argument("--dsec_root", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=200)
    p.add_argument(
        "--time_step_us",
        type=int,
        default=10_000,
        help="slice size in µs (e.g. 10000 → 10 ms)",
    )
    p.add_argument("--out_dir", type=Path, default=None)
    args = p.parse_args()

    visualise_back_and_front(
        args.dsec_root,
        args.split,
        args.start,
        args.end,
        args.time_step_us,
        args.out_dir,
    )