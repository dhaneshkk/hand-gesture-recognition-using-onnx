import os
import sys
import random

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import app  # noqa: E402
from utils.tracker import HandTracker, greedy_match  # noqa: E402


# ---------------------------------------------------------------- tracker
def test_two_close_hands_get_distinct_ids():
    t = HandTracker(max_distance=100)
    assert t.update([(100, 100)]) == [1]
    ids = t.update([(100, 100), (150, 100)])  # old code: both -> id 1
    assert ids == [1, 2]


def test_ids_follow_hands_when_detection_order_swaps():
    t = HandTracker(max_distance=100)
    a, b = t.update([(100, 100), (400, 100)])
    assert t.update([(405, 100), (102, 100)]) == [b, a]


def test_stale_tracks_are_pruned():
    t = HandTracker(max_distance=100, max_missed=2)
    t.update([(100, 100)])
    for _ in range(3):
        t.update([])
    assert t.active_ids == []
    assert t.update([(100, 100)]) == [2]  # a new hand does not inherit id 1


def test_greedy_match_is_injective_randomised():
    rng = random.Random(0)
    for _ in range(500):
        tr = np.array([[rng.uniform(0, 300), rng.uniform(0, 300)] for _ in range(rng.randint(0, 6))]).reshape(-1, 2)
        de = np.array([[rng.uniform(0, 300), rng.uniform(0, 300)] for _ in range(rng.randint(0, 6))]).reshape(-1, 2)
        pairs = greedy_match(tr, de, 120)
        ts, ds = [p[0] for p in pairs], [p[1] for p in pairs]
        assert len(set(ts)) == len(ts) and len(set(ds)) == len(ds)
        for t_i, d_i in pairs:
            assert np.linalg.norm(tr[t_i] - de[d_i]) <= 120


def test_update_ids_unique_randomised():
    rng = random.Random(1)
    t = HandTracker(max_distance=80, max_missed=3)
    for _ in range(300):
        dets = [(rng.uniform(0, 640), rng.uniform(0, 480)) for _ in range(rng.randint(0, 5))]
        ids = t.update(dets)
        assert len(ids) == len(dets) and len(set(ids)) == len(ids)


# ---------------------------------------------------------------- logging
def _loadtxt_like_notebook(path, n_features):
    x = np.loadtxt(path, delimiter=',', dtype='float32', usecols=list(range(1, n_features + 1)))
    y = np.loadtxt(path, delimiter=',', dtype='int32', usecols=(0))
    return x.reshape(-1, n_features), np.atleast_1d(y)


def test_logged_rows_match_notebook_schema(tmp_path, monkeypatch):
    kp, ph = tmp_path / 'kp.csv', tmp_path / 'ph.csv'
    monkeypatch.setattr(app, 'KEYPOINT_CSV', str(kp))
    monkeypatch.setattr(app, 'POINT_HISTORY_CSV', str(ph))
    lm = app.pre_process_landmark(np.arange(42).reshape(21, 2) * 3)
    hist = app.pre_process_point_history(640, 480, [[i, 2 * i] for i in range(16)])
    app.logging_csv(4, 1, lm, hist)
    app.logging_csv(3, 2, lm, hist)
    app.logging_csv(3, 2, lm, hist[:10])  # partial history must not be logged
    x, y = _loadtxt_like_notebook(kp, 42)
    assert y.tolist() == [4] and np.allclose(x[0], lm)
    x, y = _loadtxt_like_notebook(ph, 32)
    assert y.tolist() == [3] and np.allclose(x[0], hist)
    with open(kp) as f:
        assert len(f.readline().split(',')) == 43


def test_shipped_csvs_still_load():
    _loadtxt_like_notebook(app.KEYPOINT_CSV, 42)
    _loadtxt_like_notebook(app.POINT_HISTORY_CSV, 32)


def test_pre_process_landmark_degenerate():
    assert app.pre_process_landmark(np.zeros((21, 2))) == [0.0] * 42


# ---------------------------------------------------------------- end to end
def _run_pipeline(img):
    from model import PalmDetection, HandLandmark
    from utils.utils import rotate_and_crop_rectangle
    pd, hl = PalmDetection(providers=['CPUExecutionProvider']), HandLandmark(providers=['CPUExecutionProvider'])
    h, w = img.shape[:2]
    rects = app.palms_to_rects(pd(img), w, h)
    assert len(rects) >= 1, 'no palm detected'
    crops = rotate_and_crop_rectangle(img, rects, 'padding')
    lms, _, keep = hl(crops, rects)
    assert len(lms) == len(keep) >= 1
    return lms[0].astype(np.float64)


@pytest.fixture(scope='module')
def hand_img():
    img = cv2.imread(os.path.join(ROOT, 'hand.png'))
    assert img is not None
    return img


def test_landmarks_scale_with_resolution(hand_img):
    """Same picture at two sizes -> landmarks in the same relative positions.
    (Old code scaled by --width/--height, not the real frame, and broke here.)"""
    h, w = hand_img.shape[:2]
    small = cv2.resize(hand_img, (640, round(640 * h / w)))
    large = cv2.resize(hand_img, (1280, round(1280 * h / w)))
    a = _run_pipeline(small) / [small.shape[1], small.shape[0]]
    b = _run_pipeline(large) / [large.shape[1], large.shape[0]]
    assert np.abs(a - b).max() < 0.02


def test_landmarks_portrait_matches_landscape(hand_img):
    """Rotate the frame 90 deg: landmarks must rotate with it (portrait padding fix)."""
    h, w = hand_img.shape[:2]
    land = cv2.resize(hand_img, (640, round(640 * h / w)))
    if land.shape[0] >= land.shape[1]:
        pytest.skip('test image is not landscape')
    port = cv2.rotate(land, cv2.ROTATE_90_CLOCKWISE)
    a = _run_pipeline(land)
    b = _run_pipeline(port)
    # clockwise: (x, y) -> (H - 1 - y, x)
    a_rot = np.stack([land.shape[0] - 1 - a[:, 1], a[:, 0]], axis=1)
    err = np.abs(a_rot - b).max() / max(port.shape[:2])
    assert err < 0.03, err
