import numpy as np


def compute_img_idx_to_track_idx(t_track, t_image):
    x, counts = np.unique(t_track, return_counts=True)
    i, j = (x.reshape((-1,1)) == t_image.reshape((1,-1))).nonzero()
    deltas = np.zeros_like(t_image)

    deltas[j] = counts[i]

    idx = np.concatenate([np.array([0]), deltas]).cumsum()
    return np.stack([idx[:-1], idx[1:]], axis=-1).astype("uint64")



def interpolate_tracks_series(det0, det1, *, step_us: int,
                              drop_singletons: str = "internal"):
    """
    Interpolates on a regular time grid between t0 and t1.

    Parameters
    ----------
    step_us : int
        Grid spacing in micro‑seconds (e.g. 10_000 → 10ms).
    drop_singletons : {"internal", "never", "always"}, default "internal"
        What to do with objects visible only in *one* key‑frame.
          • "internal" – keep them **only** at the key‑frames, drop them on
                         the interpolated interior grid‑points  ← your request
          • "never"    – keep them everywhere (propagate unchanged)
          • "always"   – drop them everywhere
    Returns
    -------
    list[np.ndarray]  (length = ceil((t1-t0)/step_us)+1)
    """
    if len(det0) == 0 and len(det1) == 0:
        return []

    t0, t1 = int(det0['t'][0]), int(det1['t'][0])
    n_steps = int(np.ceil((t1 - t0) / step_us))
    out = []
    for i in range(n_steps + 1):
        t = t0 + i * step_us
        if   i == 0:              # exactly at t0
            keep = drop_singletons in ("internal", "never")
            out.append(det0.copy() if keep else det0.copy()[[]])
        elif i == n_steps:        # exactly at t1
            keep = drop_singletons in ("internal", "never")
            out.append(det1.copy() if keep else det1.copy()[[]])
        else:                     # inside (t0,t1)
            keep = drop_singletons in ("never")
            out.append(interpolate_tracks(det0, det1, t,
                                           drop_singletons=not keep))
    return out            # len(out) == n_steps+1



def interpolate_tracks(det0, det1, t, drop_singletons=True):
    """
    Linearly interpolate bounding boxes between det0 (t0) and det1 (t1).

    Parameters
    ----------
    det0, det1 : np.ndarray
        Structured arrays with fields ('track_id','t','x','y','w','h','class_id').
        They do **not** need to have the same length.
    t : int
        Target time stamp (micro‑seconds).
    drop_singletons : bool, default True
        * True  – keep only objects that exist in **both** frames.
        * False – propagate the single box (no interpolation) if the object
                   exists in just one frame.

    Returns
    -------
    det_out : np.ndarray with the same dtype as the inputs.
    """
    if len(det0) == 0 and len(det1) == 0:
        return det0  # empty

    # Align on track_id -------------------------------------------------
    #   left join because some objects may disappear / appear

    order0 = np.argsort(det0['track_id'])
    order1 = np.argsort(det1['track_id'])
    det0   = det0[order0]
    det1   = det1[order1]

    ids0 = det0['track_id']
    ids1 = det1['track_id']
    common_ids = np.intersect1d(ids0, ids1, assume_unique=True)

    if len(common_ids) == 0 and drop_singletons:
        return det0 if (t - det0['t'][0]) < (det1['t'][0] - t) else det1

    dtype = det0.dtype
    det_out = np.empty(len(common_ids) if drop_singletons else len(np.union1d(ids0, ids1)),
                       dtype=dtype)

    # linear interpolation factor r ∈ [0,1]
    t0, t1 = det0['t'][0], det1['t'][0]
    r = np.float32((t - t0) / (t1 - t0))

    idx0 = np.searchsorted(ids0, common_ids)
    idx1 = np.searchsorted(ids1, common_ids)
    det_out[:len(common_ids)] = det0[idx0]         # copy meta fields
    for k in 'xywh':
        det_out[k][:len(common_ids)] = (det0[k][idx0] * (1.0 - r) +
                                        det1[k][idx1] * r)
        det_out['t'][:] = t

        # 2) singletons -----------------------------------------------------
    if not drop_singletons:
        single_0 = np.setdiff1d(ids0, common_ids, assume_unique=True)
        single_1 = np.setdiff1d(ids1, common_ids, assume_unique=True)
        ofs = len(common_ids)
        if len(single_0):
            det_out[ofs:ofs + len(single_0)] = det0[np.searchsorted(ids0, single_0)]
            ofs += len(single_0)
        if len(single_1):
            det_out[ofs:ofs + len(single_1)] = det1[np.searchsorted(ids1, single_1)]


    return det_out