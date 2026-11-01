"""Synthetic ground-truth tests: render a 50 mm marker plus known rectangle, warp with a
perspective transform (simulated tilted camera), then check recovered mm values."""
import math

import cv2
import numpy as np
import pytest

from measure import core

PX_PER_MM = 8.0          # plane rendering scale
MARKER_MM = 50.0
CANVAS_MM = (300, 200)   # plane size in mm


def make_plane_image():
    """Top-down plane: white sheet, marker at (20,20) mm, 100x60 mm black rectangle at (120,60) mm."""
    w, h = int(CANVAS_MM[0] * PX_PER_MM), int(CANVAS_MM[1] * PX_PER_MM)
    img = np.full((h, w, 3), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    m = cv2.aruco.generateImageMarker(d, 0, int(MARKER_MM * PX_PER_MM))
    x0 = y0 = int(20 * PX_PER_MM)
    img[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    # white quiet zone around the marker is already white
    return img


def mm2px(x, y):
    return (x * PX_PER_MM, y * PX_PER_MM)


def warp(img, strength=0.0):
    """Perspective-warp to simulate a tilted camera; returns image and 3x3 matrix."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    s = strength
    dst = np.float32([[w * s, h * s], [w * (1 - s * 0.3), h * s * 0.2],
                      [w * (1 - s * 0.1), h], [w * s * 0.2, h * (1 - s)]]) if s else src
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(200, 200, 200)), M


def project(M, pts):
    return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), M).reshape(-1, 2)


@pytest.mark.parametrize("tilt", [0.0, 0.08, 0.15])
def test_length_and_area_under_tilt(tilt):
    plane = make_plane_image()
    img, M = warp(plane, tilt)
    cal = core.calibrate_plane(img, MARKER_MM)

    # 100 mm edge of the rectangle: (120,60) -> (220,60)
    p = project(M, [mm2px(120, 60), mm2px(220, 60)])
    L = core.distance_mm(cal, p[0], p[1])
    assert L == pytest.approx(100.0, rel=0.02)

    # 100 x 60 rectangle area = 6000 mm2
    rect = [mm2px(120, 60), mm2px(220, 60), mm2px(220, 120), mm2px(120, 120)]
    A = core.polygon_area_mm2(cal, project(M, rect))
    assert A == pytest.approx(6000.0, rel=0.03)


def test_no_marker_raises():
    blank = np.full((400, 400, 3), 255, np.uint8)
    with pytest.raises(core.MeasureError):
        core.calibrate_plane(blank, 50)


def test_volume_formulas():
    assert core.volume_m3("box", length_mm=1000, width_mm=500, height_mm=200) == pytest.approx(0.1)
    assert core.volume_m3("cylinder", diameter_mm=100, height_mm=100) == pytest.approx(math.pi * 50**2 * 100 * 1e-9)
    assert core.volume_m3("sphere", diameter_mm=100) == pytest.approx(4 / 3 * math.pi * 50**3 * 1e-9)
    assert core.volume_m3("prism", base_area_mm2=6000, height_mm=100) == pytest.approx(6e-4)
    with pytest.raises(core.MeasureError):
        core.volume_m3("blob")


def test_mass_band():
    m = core.mass_kg(0.001, "steel")  # 1 litre of steel
    assert m["nominal_kg"] == pytest.approx(7.85)
    assert m["low_kg"] < m["nominal_kg"] < m["high_kg"]
    assert core.mass_kg(0.001, "steel", 0.5)["nominal_kg"] == pytest.approx(3.925)


def test_uncertainty_grows_with_small_marker():
    plane = make_plane_image()
    img, _ = warp(plane, 0.0)
    cal = core.calibrate_plane(img, MARKER_MM)
    u_big = core.length_uncertainty_mm(cal, 100.0)
    cal.marker_edge_px /= 4
    cal.mm_per_px *= 4
    assert core.length_uncertainty_mm(cal, 100.0) > u_big
