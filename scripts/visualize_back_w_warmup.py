#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualise *back_w_warmup* sync.

LEFT   – warm-up image (frame i-1) + original boxes
MIDDLE – current image  (frame i  ) + original boxes
RIGHT  – events in one slice + interpolated boxes that belong to the slice
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import hdf5plugin        # noqa: F401  (enables Blosc for h5py)
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
def visualise_warmup(
    dsec_root: Path,
    split: str,
    start_idx: int,
    end_idx: int,
    step_us: int,
    out_dir: Path,
):
    # dataset that returns warm-up + interpolated tracks
    ds_intp = DSECDet(
        dsec_root,
        split=split,
        sync="back_w_warmup",
        interpolate_labels=True,
        interp_step_us=step_us,
        singleton_policy="internal",
    )

    # dataset that gives raw ground-truth boxes on individual images
    ds_raw = DSECDet(
        dsec_root,
        split=split,
        sync="back",
        interpolate_labels=False,
    )

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(start_idx, min(end_idx, len(ds_intp))):
        try:
            # ----------------------------------------------------------
            # load once per window
            # ----------------------------------------------------------
            sample = ds_intp[idx]

            warm_img = gamma(sample["image_warm_up"])
            curr_img = gamma(sample["image"])

            warm_gt = sample.get("tracks_warm_up")
            if warm_gt is None:
                warm_gt = np.empty(0, dtype=sample["tracks"].dtype)

            curr_gt = ds_raw.get_tracks(idx)

            ev = sample["events"]
            tr_intp = sample["tracks"]

            t_min, t_max = ev["t"][0], ev["t"][-1]

            # ----------------------------------------------------------
            # static LEFT and MIDDLE panels
            # ----------------------------------------------------------
            left_panel = warm_img.copy()
            if len(warm_gt):
                left_panel = render_object_detections_on_image(
                    left_panel, warm_gt, label="warm_gt"
                )

            mid_panel = curr_img.copy()
            if len(curr_gt):
                mid_panel = render_object_detections_on_image(
                    mid_panel, curr_gt, label="curr_gt"
                )

            # ----------------------------------------------------------
            # walk through event window in slices
            # ----------------------------------------------------------
            slice_id = 0
            t_cur = int(t_min)
            while t_cur < t_max:
                t_next = min(t_cur + step_us, t_max)

                ev_slice = slice_events(ev, t_cur, t_next)
                tr_slice = slice_tracks(tr_intp, t_cur, t_next)

                # RIGHT panel: events + interpolated boxes
                right_panel = curr_img.copy()
                if len(ev_slice["t"]):
                    right_panel = render_events_on_image(
                        right_panel,
                        ev_slice["x"],
                        ev_slice["y"],
                        ev_slice["p"],
                    )
                if len(tr_slice):
                    right_panel = render_object_detections_on_image(
                        right_panel, tr_slice, label="interp"
                    )

                # ------------------------------------------------------
                # compose 3-panel figure
                # ------------------------------------------------------
                fig, axs = plt.subplots(
                    1,
                    3,
                    figsize=(3 * (curr_img.shape[1] / 100), curr_img.shape[0] / 100),
                    dpi=100,
                )

                axs[0].imshow(left_panel[..., ::-1])
                axs[0].set_title("warm-up (i-1)", fontsize=7)

                axs[1].imshow(mid_panel[..., ::-1])
                axs[1].set_title("current (i)", fontsize=7)

                axs[2].imshow(right_panel[..., ::-1])
                axs[2].set_title(
                    f"slice {slice_id}  [{(t_cur - t_min)/1e3:.1f}–{(t_next - t_min)/1e3:.1f} ms]",
                    fontsize=7,
                )

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
    p = argparse.ArgumentParser("Visualise back_w_warmup sync on DSEC")
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

    visualise_warmup(
        args.dsec_root,
        args.split,
        args.start,
        args.end,
        args.time_step_us,
        args.out_dir,
    )
