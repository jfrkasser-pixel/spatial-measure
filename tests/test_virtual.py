"""Virtual square fitted from two pairs of parallel lines."""
import numpy as np
import cv2
from measure import core

W, H_IMG, F = 1000, 750, 1000.0
K = np.array([[F, 0, W / 2], [0, F, H_IMG / 2], [0, 0, 1.0]])


def project(pts_mm, rvec, tvec):
    P = np.hstack([np.asarray(pts_mm, float), np.zeros((len(pts_mm), 1))])
    px, _ = cv2.projectPoints(P, np.asarray(rvec, float), np.asarray(tvec, float), K, None)
    return px.reshape(-1, 2)


def scene(rvec=(-0.9, 0.25, 0.05), tvec=(-150.0, -100.0, 600.0)):
    # world plane z=0 in mm. Rows (direction x) at y=0 and y=250, sides (direction y) at x=0 and x=300
    def seg(a, b): return project([a, b], rvec, tvec)
    pts = np.vstack([seg((0, 0), (300, 0)), seg((0, 250), (300, 250)), seg((0, 0), (0, 250)), seg((300, 0), (300, 250))])
    return pts, rvec, tvec


def side_px_for(rvec, tvec, X, pts=None):
    """Apparent size of an X mm square at the lines' centroid: square root of its image area."""
    if pts is None:
        pts, _, _ = scene(rvec, tvec)
    world = np.array([[0, 0], [300, 0], [300, 250], [0, 250]], float)
    Ht, _ = cv2.findHomography(project(world, rvec, tvec), world)
    cen = pts.mean(axis=0)
    c = cv2.perspectiveTransform(cen.reshape(1, 1, 2), Ht).reshape(2)
    q = project([c + [-X / 2, -X / 2], c + [X / 2, -X / 2], c + [X / 2, X / 2], c + [-X / 2, X / 2]], rvec, tvec)
    return float(np.sqrt(cv2.contourArea(q.astype(np.float32))))


def test_lengths_recovered_and_f_estimated():
    pts, rvec, tvec = scene()
    X = 40.0
    cal = core.calibrate_virtual_square((W, H_IMG), pts, X, side_px_for(rvec, tvec, X))
    assert abs(cal.fit_info["f_px"] - F) / F < 0.02
    for a, b, true in [((20, 30), (220, 30), 200.0), ((50, 40), (50, 200), 160.0), ((30, 30), (250, 200), np.hypot(220, 170))]:
        pa, pb = project([a, b], rvec, tvec)
        assert abs(core.distance_mm(cal, pa, pb) - true) / true < 0.02


def test_scale_is_the_only_free_parameter():
    pts, rvec, tvec = scene()
    a, b = project([(20, 30), (220, 30)], rvec, tvec)
    sp = side_px_for(rvec, tvec, 40.0)
    L40 = core.distance_mm(core.calibrate_virtual_square((W, H_IMG), pts, 40.0, sp), a, b)
    L80 = core.distance_mm(core.calibrate_virtual_square((W, H_IMG), pts, 80.0, sp), a, b)
    assert abs(L80 / L40 - 2.0) < 1e-6


def test_virtual_square_drawn_as_square_on_the_plane():
    pts, rvec, tvec = scene()
    cal = core.calibrate_virtual_square((W, H_IMG), pts, 40.0, side_px_for(rvec, tvec, 40.0))
    q = core.to_plane(cal.H, cal.corners_px)
    sides = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    assert all(abs(s - 40.0) < 0.01 for s in sides)


def test_bad_input_rejected():
    pts, rvec, tvec = scene()
    import pytest
    with pytest.raises(core.MeasureError):
        core.calibrate_virtual_square((W, H_IMG), pts[:6], 40.0, 50.0)
    with pytest.raises(core.MeasureError):
        core.calibrate_virtual_square((W, H_IMG), pts, 0.0, 50.0)
