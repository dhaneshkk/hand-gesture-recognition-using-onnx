"""Minimal one-to-one palm tracker.

Replaces the per-detection nearest-neighbour lookup that used to live in
app.py. That lookup could hand the same track id to two detections in one
frame (the second silently overwrote the first in a dict), and never dropped
tracks for hands that had left the view.

Assignment here is *global greedy matching*: every (track, detection) pair
within ``max_distance`` is sorted by distance and accepted only if neither the
track nor the detection has been used yet. Each detection therefore receives a
distinct track id (see ``proofs/TrackerInjective.lean`` for the formal
statement of that invariant). Unmatched detections open new tracks; tracks not
matched for more than ``max_missed`` consecutive frames are removed.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np


def greedy_match(
    track_points: np.ndarray,
    det_points: np.ndarray,
    max_distance: float,
) -> List[Tuple[int, int]]:
    """Return (track_index, detection_index) pairs, each index used at most once.

    Pairs are accepted in increasing order of Euclidean distance; ties are
    broken by (track_index, detection_index) so the result is deterministic.
    """
    if len(track_points) == 0 or len(det_points) == 0:
        return []
    diff = track_points[:, None, :] - det_points[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    ti, di = np.nonzero(dist <= max_distance)
    order = np.lexsort((di, ti, dist[ti, di]))
    used_t, used_d, pairs = set(), set(), []
    for k in order:
        t, d = int(ti[k]), int(di[k])
        if t in used_t or d in used_d:
            continue
        used_t.add(t)
        used_d.add(d)
        pairs.append((t, d))
    return pairs


class HandTracker:
    def __init__(self, max_distance: float = 100.0, max_missed: int = 5):
        self.max_distance = float(max_distance)
        self.max_missed = int(max_missed)
        self.centers: Dict[int, np.ndarray] = {}
        self.missed: Dict[int, int] = {}
        self._next_id = 1

    @property
    def active_ids(self) -> List[int]:
        return list(self.centers.keys())

    def update(self, det_centers: Sequence[Sequence[float]]) -> List[int]:
        """Assign a track id to each detection centre (same order as input)."""
        dets = np.asarray(det_centers, dtype=np.float32).reshape(-1, 2)
        ids = list(self.centers.keys())
        tracks = (
            np.stack([self.centers[i] for i in ids])
            if ids else np.zeros((0, 2), np.float32)
        )
        assigned: List[int] = [-1] * len(dets)
        for t, d in greedy_match(tracks, dets, self.max_distance):
            assigned[d] = ids[t]

        for d, tid in enumerate(assigned):
            if tid == -1:
                tid = self._next_id
                self._next_id += 1
                assigned[d] = tid
            self.centers[tid] = dets[d].copy()
            self.missed[tid] = 0

        seen = set(assigned)
        for tid in list(self.centers.keys()):
            if tid in seen:
                continue
            self.missed[tid] += 1
            if self.missed[tid] > self.max_missed:
                del self.centers[tid]
                del self.missed[tid]
        return assigned
