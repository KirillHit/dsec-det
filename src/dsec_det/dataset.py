from pathlib import Path 
import cv2
import numpy as np
import torch
from dsec_det.directory import DSECDirectory

from dsec_det.preprocessing import compute_img_idx_to_track_idx, interpolate_tracks, interpolate_tracks_series
from dsec_det.io import extract_from_h5_by_timewindow
from dsec_det.visualize import render_object_detections_on_image, render_events_on_image
from dsec_det.label import CLASSES
from torch.utils.data import Dataset


class DSECDet:
    def __init__(self, root: Path,
                 split: str="train",
                 sync: str="front",
                 debug: bool=False,
                 split_config=None,
                 interpolate_labels: bool = False,
                 interp_step_us: int = 10_000,
                 singleton_policy: str = "internal"
        ):
        """
        root: Root to the the DSEC dataset (the one that contains 'train' and 'test'
        split: Can be one of ['train', 'test']
        window_size: Number of microseconds of data
        sync: Can be either 'front' (last event ts), or 'back' (first event ts). Whether the front of the window or
              the back of the window is synced with the images.
        interpolate_labels
            If True, bounding boxes are linearly interpolated on a regular
            time grid between the *two* images that bound the current
            event window.
        interp_step_us
            Grid spacing deltat in µs (e.g. 10000 → 10ms between grid‑points).
        singleton_policy
            What to do with objects that exist in only one of the two
            key‑frames:
              • "internal" – keep them on the key‑frames, drop them on the
                             interpolated interior points   (default,
                             corresponds to your spec).
              • "never"    – keep them everywhere.
              • "always"   – drop them everywhere.

        Each sample of this dataset loads one image, events, and labels at a timestamp. The behavior is different for
        sync='front' and sync='back', and these are visualized below.

        Legend:
        . = events
        | = image
        L = label

        sync='front'
        -------> time
        .......|
               L

        sync='back'
        -------> time
        |.......
        L

        """
        assert root.exists()
        assert split in ['train', 'test', 'val']
        #assert (root / split).exists()
        assert sync in ['front', 'back', 'back_and_front'], f"unknown sync mode '{sync}'"

        if sync == 'back_and_front':
            assert interpolate_labels, "'back_and_front' requires interpolate_labels=True"


        self.debug = debug
        self.classes = CLASSES

        self.root = root / ("train" if split in ['train', 'val'] else "test")
        self.sync = sync

        self.interpolate_labels = interpolate_labels
        self.interp_step_us = interp_step_us
        self.singleton_policy = singleton_policy

        self.height = 480
        self.width = 640

        self.directories = dict()
        self.img_idx_track_idxs = dict()

        if split_config is None:
            self.subsequence_directories = list(self.root.glob("*/"))
        else:
            available_dirs = list(self.root.glob("*/"))
            self.subsequence_directories = [self.root / s for s in split_config[split] if self.root / s in available_dirs]

        self.subsequence_directories = sorted(self.subsequence_directories, key=self.first_time_from_subsequence)

        for f in self.subsequence_directories:
            directory = DSECDirectory(f)
            self.directories[f.name] = directory
            self.img_idx_track_idxs[f.name] = compute_img_idx_to_track_idx(directory.tracks.tracks['t'],
                                                                           directory.images.timestamps)

    def first_time_from_subsequence(self, subsequence):
        return np.genfromtxt(subsequence / "images/timestamps.txt", dtype="int64")[0]

    def _build_interpolated_tracks(self, directory, idx_prev, idx_next):
        """
        Interpolate detections between two image indices (inclusive) on a
        regular deltat grid.  Returns **one** stacked structured array whose rows
        carry the correct 't' field (absolute micro‑seconds).
        """
        img2trk = self.img_idx_track_idxs[directory.root.name]

        # detections on the bounding key‑frames
        p0, p1 = img2trk[idx_prev]
        n0, n1 = img2trk[idx_next]
        det0 = directory.tracks.tracks[p0:p1]
        det1 = directory.tracks.tracks[n0:n1]

        if len(det0) == 0 or len(det1) == 0:
            return det0 if len(det1) == 0 else det1

        det_series = interpolate_tracks_series(det0, det1,
                                               step_us=self.interp_step_us,
                                               drop_singletons=self.singleton_policy)

        if len(det_series) == 0:
            return det0[:0]

        return np.concatenate(det_series, axis=0)


    def __len__(self):
        return sum(len(v)-1 for v in self.img_idx_track_idxs.values())

    def __getitem__(self, item):
        output = {}
        output['image'] = self.get_image(item)
        if self.sync == 'back_and_front':
            # idx_local is guaranteed < len(front_files)-1 because
            # __len__() exposes only N-1 windows per subsequence.
            output['image_front'] = self.get_image(item + 1)

        output['events'] = self.get_events(item)
        output['tracks'] = self.get_tracks(item)

        if self.debug:
            # visualize tracks and events
            events = output['events']
            image = (255 * (output['image'].astype("float32") / 255) ** (1/2.2)).astype("uint8")
            output['debug'] = render_events_on_image(image, x=events['x'], y=events['y'], p=events['p'])
            output['debug'] = render_object_detections_on_image(output['debug'], output['tracks'])

        return output

    def get_index_window(self, index, num_idx, sync="back"):
        if sync == "front":
            if index == 0: # there is no “previous frame” for image0  → fall back to 'back'
                i_0 = 0
                i_1 = min(1, num_idx - 1)
            else:
                i_0 = index - 1
                i_1 = index
        else:
            #assert 0 <= index < num_idx - 1
            i_0 = index
            i_1 = np.clip(index + 1, 0, num_idx - 1)

        return i_0, i_1

    def get_tracks(self, index, mask=None, directory_name=None):
        index, img_idx_to_track_idx, directory = self.rel_index(index, directory_name)
        idx0, idx1 = img_idx_to_track_idx[index]
        tracks_main = directory.tracks.tracks[idx0:idx1]

        if self.interpolate_labels:
            # neighbouring frame indices according to sync
            prev_idx, next_idx = self.get_index_window(index,
                                                       len(img_idx_to_track_idx),
                                                       sync=self.sync)
            tracks = self._build_interpolated_tracks(directory,
                                                     prev_idx, next_idx)
        else:
            tracks = tracks_main

        if mask is not None:
            tracks = tracks[mask[idx0:idx1]]

        return tracks

    def get_events(self, index, directory_name=None):
        index, img_idx_to_track_idx, directory = self.rel_index(index, directory_name)
        i_0, i_1 = self.get_index_window(index, len(img_idx_to_track_idx), sync=self.sync)
        t_0, t_1 = directory.images.timestamps[[i_0, i_1]]
        events = extract_from_h5_by_timewindow(directory.events.event_file, t_0, t_1)
        return events

    def get_image(self, index, directory_name=None):
        index, img_idx_to_track_idx, directory = self.rel_index(index, directory_name)
        image_files = directory.images.image_files_distorted
        image = cv2.imread(str(image_files[index]))
        return image

    def rel_index(self, index, directory_name=None):
        if directory_name is not None:
            img_idx_to_track_idx = self.img_idx_track_idxs[directory_name]
            directory = self.directories[directory_name]
            return index, img_idx_to_track_idx, directory

        for f in self.subsequence_directories:
            img_idx_to_track_idx = self.img_idx_track_idxs[f.name]
            if len(img_idx_to_track_idx)-1 <= index:
                index -= (len(img_idx_to_track_idx)-1)
                continue
            else:
                return index, img_idx_to_track_idx, self.directories[f.name]
        else:
            raise ValueError


class SlidingWindowDSEC(Dataset):
    """
    Each sample is a window of fixed length delta = image_interval_us which is
    randomly positioned between the *previous* and *next* image.
    All events in the window receive interpolated labels on an equidistant
    grid (deltat = step_us).

        t_(i‑1)        t_i        t_(i+1)
     |------------|----IMAGE----|------------|
    <-- events A --> <- events B -> <- events C ->
              ^ image may sit anywhere in this 2‑delta span

    """  # TODO test this
    def __init__(self, base: DSECDet,
                 step_us      : int   = 10_000,   # 10 ms labels
                 drop_singletons: str = "internal"):
        assert base.interpolate_labels is True, \
               "base must be created with interpolate_labels=True"
        self.base = base
        self.step_us = step_us
        self.drop_singletons = drop_singletons

        # enumerate (image_idx) once; each index will later expand to 1 sample
        self._indices = list(range(len(base)))

    def __len__(self): return len(self._indices)

    def __getitem__(self, idx):
        img_idx = self._indices[idx]

        # ---------------------------------------------------------------
        # Resolve subsequence, global → local index, etc.
        # ---------------------------------------------------------------
        rel_idx, img2trk, subseq = self.base.rel_index(img_idx)

        ts_all = subseq.images.timestamps
        n_imgs = len(ts_all)

        # timestamps of the three bounding images
        ts_prev = ts_all[max(rel_idx - 1, 0)]
        ts_curr = ts_all[rel_idx]
        ts_next = ts_all[min(rel_idx + 1, n_imgs - 1)]

        delta = ts_curr - ts_prev                    # fixed window length
        # random shift inside the [prev, next‑delta] span
        t_start = ts_prev + np.random.rand() * (ts_next - ts_prev - delta)
        t_end = t_start + delta

        ev = extract_from_h5_by_timewindow(
            subseq.events.event_file, int(t_start), int(t_end)
        )

        # ---------------------------------------------------------------
        # LABELS  (absolute µs on a deltat grid)
        # ---------------------------------------------------------------
        # --- key‑frame detections without any extra interpolation ------
        #     we go straight to the .npy to guarantee “raw”
        beg0, end0 = img2trk[max(rel_idx - 1, 0)]
        beg1, end1 = img2trk[min(rel_idx + 1, n_imgs - 1)]
        det_before = subseq.tracks.tracks[beg0:end0]
        det_after = subseq.tracks.tracks[beg1:end1]

        det_series = interpolate_tracks_series(
            det_before,
            det_after,
            step_us=self.step_us,
            drop_singletons=self.drop_singletons,
        )

        # shift the grid so it starts at *exactly* t_start
        label_grid = []
        for i, det in enumerate(det_series):
            lbl = det.copy()
            lbl["t"] = int(t_start + i * self.step_us)
            label_grid.append(lbl)


        return dict(
            events=ev,                    # absolute µs
            labels=label_grid,            # list[np.ndarray], abs µs
            t_start_us=int(t_start),
            t_end_us=int(t_end),
            t_window_us=int(delta),
            step_us=self.step_us,
            image_index=img_idx,
            sequence=str(subseq.root.name),
        )