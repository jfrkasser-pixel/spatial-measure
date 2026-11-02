"""Tests for the A4 multi-marker target, lens calibration and automatic outline."""
import cv2
import numpy as np
import pytest

from measure import core, targets

PPM = 4.0                      # sheet render scale, px per mm
SHEET = (210.0, 297.0)


def sheet_image(object_rect_mm=None):
    """A4 target on a larger white canvas-free plane; optional dark textured rectangle."""
    g = targets.render_a4_target(PPM)
    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if object_rect_mm:
        x0, y0, x1, y1 = (int(v * PPM) for v in object_rect_mm)
        rng = np.random.default_rng(0)
        tex = np.clip(rng.normal(70, 18, (y1 - y0, x1 - x0, 3)) + np.array([40, 0, 0]), 0, 255)
        img[y0:y1, x0:x1] = tex.astype(np.uint8)
    return img


def tilt(img, s=0.10):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * s, h * s * 0.6], [w * (1 - s * 0.4), h * s * 0.1],
                      [w * (1 - s * 0.1), h], [w * s * 0.2, h * (1 - s * 0.5)]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(210, 210, 210)), M


def proj(M, pts_mm):
    p = np.float32(pts_mm).reshape(-1, 1, 2) * PPM
    return cv2.perspectiveTransform(p, M).reshape(-1, 2)


# ---------------------------------------------------------------- multi marker
def test_a4_target_prints_to_exact_size():
    g = targets.render_a4_target(11.81)
    assert g.shape[1] / 11.81 == pytest.approx(210, abs=0.1)
    assert g.shape[0] / 11.81 == pytest.approx(297, abs=0.1)


def test_multi_marker_measures_far_distance_accurately():
    img, M = tilt(sheet_image())
    cal = core.calibrate_multi(img)
    assert cal.n_markers == 4 and cal.source == "aruco_multi"
    assert cal.residual_mm < 1.0
    # length across the whole sheet between two points far from any single marker
    p = proj(M, [(20, 80), (190, 280)])
    true = float(np.hypot(170, 200))
    assert core.distance_mm(cal, p[0], p[1]) == pytest.approx(true, rel=0.01)


def test_multi_marker_works_with_partial_view():
    img, _ = tilt(sheet_image())
    img[: int(60 * PPM), int(100 * PPM):] = 255        # hide marker 1 (top-right)
    cal = core.calibrate_multi(img)
    assert cal.n_markers == 3


def test_multi_marker_none_found():
    with pytest.raises(core.MeasureError):
        core.calibrate_multi(np.full((300, 300, 3), 255, np.uint8))


# ------------------------------------------------------------------------ lens
def distort(img_ideal, K, dist):
    """Make a lens-distorted version of an ideal (pinhole) image."""
    h, w = img_ideal.shape[:2]
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts = np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2)
    ideal = cv2.undistortPoints(pts, K, dist, P=K).reshape(h, w, 2)
    return cv2.remap(img_ideal, ideal[..., 0], ideal[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(210, 210, 210))


def test_lens_calibration_recovers_distortion():
    rng = np.random.default_rng(3)
    K = np.array([[1000, 0, 640], [0, 1000, 480], [0, 0, 1]], np.float64)
    dist = np.array([-0.12, 0.05, 0.0005, -0.0004, 0.0])
    obj = core.board_object_points((9, 6), 25.0)
    objs, imgs = [], []
    for _ in range(14):
        rvec = rng.normal(0, 0.25, 3)
        tvec = np.array([-100 + rng.normal(0, 25), -60 + rng.normal(0, 20), 450 + rng.normal(0, 60)])
        pts, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        pts = pts.reshape(-1, 2) + rng.normal(0, 0.25, (len(obj), 2))
        if pts.min() < 0 or pts[:, 0].max() > 1280 or pts[:, 1].max() > 960:
            continue
        objs.append(obj)
        imgs.append(pts.astype(np.float32))
    lens = core.calibrate_lens(objs, imgs, (1280, 960))
    assert lens.rms_px < 0.6
    assert lens.dist[0] == pytest.approx(-0.12, abs=0.05)
    again = core.LensModel.from_json(lens.to_json())
    assert np.allclose(again.K, lens.K)


def test_lens_correction_improves_length_near_edge():
    base, M = tilt(sheet_image(), 0.0)
    h, w = base.shape[:2]
    K = np.array([[900, 0, w / 2], [0, 900, h / 2], [0, 0, 1]], np.float64)
    dist = np.array([-0.18, 0.04, 0, 0, 0])
    bad = distort(base, K, dist)
    lens = core.LensModel(K, dist, (w, h))
    # 100 mm segment near the left edge (x 8 to 108 mm... use a vertical one along the edge)
    pts_ideal = proj(M, [(10, 60), (10, 160)]).astype(np.float64)
    # distorted pixel position of an ideal point
    d_pts, _ = cv2.projectPoints(
        np.hstack([(pts_ideal - K[:2, 2]) / 900.0, np.ones((2, 1))]).astype(np.float64),
        np.zeros(3), np.zeros(3), K, dist)
    d_pts = d_pts.reshape(-1, 2)
    try:
        cal_raw = core.calibrate_multi(bad)
        e_raw = abs(core.distance_mm(cal_raw, d_pts[0], d_pts[1]) - 100.0) / 100.0
    except core.MeasureError:
        e_raw = 1.0
    good = core.undistort_image(bad, lens)
    cal_ok = core.calibrate_multi(good)
    u = core.undistort_points(d_pts, lens, (w, h))
    e_ok = abs(core.distance_mm(cal_ok, u[0], u[1]) - 100.0) / 100.0
    assert e_ok < 0.02
    assert e_ok < e_raw


# --------------------------------------------------------------------- outline
def test_outline_area_of_rectangle_on_tilted_sheet():
    img, M = tilt(sheet_image((50, 110, 150, 170)), 0.08)     # 100 x 60 mm
    cal = core.calibrate_multi(img)
    seed = proj(M, [(100, 140)])[0]
    o = core.detect_outline(img, cal, seed)
    assert o["area_mm2"] == pytest.approx(6000.0, rel=0.06)
    assert o["rect_long_mm"] == pytest.approx(100.0, rel=0.05)
    assert o["rect_short_mm"] == pytest.approx(60.0, rel=0.06)


def test_outline_tap_outside_image():
    img, _ = tilt(sheet_image(), 0.05)
    cal = core.calibrate_multi(img)
    with pytest.raises(core.MeasureError):
        core.detect_outline(img, cal, (-5, 10))
