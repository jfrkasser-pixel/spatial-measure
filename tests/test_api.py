"""HTTP-level tests of /api/measure and /api/estimate."""
import io
import json

import cv2
import numpy as np
import pytest

import app as app_module
from measure import core
import test_features as t


@pytest.fixture()
def client():
    app_module.LENS = None
    return app_module.app.test_client()


def post(client, img, payload):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    r = client.post("/api/measure", data={"frame": (io.BytesIO(buf.tobytes()), "f.jpg"),
                                          "payload": json.dumps(payload)})
    return r.status_code, r.get_json()


def test_a4_length(client):
    img, M = t.tilt(t.sheet_image(), 0.08)
    pts = t.proj(M, [(20, 80), (190, 80)]).tolist()
    code, d = post(client, img, {"mode": "length", "points": pts, "reference": "a4"})
    assert code == 200 and d["source"] == "aruco_multi" and d["n_markers"] == 4
    assert d["length_mm"] == pytest.approx(170.0, rel=0.01)


def test_outline_endpoint(client):
    img, M = t.tilt(t.sheet_image((50, 110, 150, 170)), 0.08)
    seed = t.proj(M, [(100, 140)])[0].tolist()
    code, d = post(client, img, {"mode": "outline", "points": [seed], "reference": "a4"})
    assert code == 200
    assert d["area_mm2"] == pytest.approx(6000, rel=0.06)
    assert len(d["polygon_px"]) >= 4 and len(d["marker_polys_px"]) == 4


def test_lens_is_applied_and_overlays_stay_on_raw_photo(client):
    base, M = t.tilt(t.sheet_image(), 0.0)
    h, w = base.shape[:2]
    K = np.array([[900, 0, w / 2], [0, 900, h / 2], [0, 0, 1]], float)
    lens = core.LensModel(K, np.array([-0.18, 0.04, 0, 0, 0]), (w, h))
    bad = t.distort(base, K, lens.dist)
    app_module.LENS = lens
    try:
        ideal = t.proj(M, [(10, 60), (10, 160)]).astype(float)
        raw = core.distort_points(ideal, lens, (w, h))
        code, d = post(client, bad, {"mode": "length", "points": raw.tolist(), "reference": "a4"})
    finally:
        app_module.LENS = None
    assert code == 200 and d["lens_corrected"] is True
    assert d["length_mm"] == pytest.approx(100.0, rel=0.02)
    # marker outlines come back in raw-photo coordinates: a marker corner lands on its true raw position
    true_tl = core.distort_points(t.proj(M, [(10, 10)]), lens, (w, h))[0]
    corners = np.array([pt for poly in d["marker_polys_px"] for pt in poly])
    assert np.min(np.linalg.norm(corners - true_tl, axis=1)) < 6


def test_errors_are_400_not_500(client):
    img = np.full((300, 300, 3), 255, np.uint8)
    code, d = post(client, img, {"mode": "length", "points": [[1, 1], [2, 2]], "reference": "a4"})
    assert code == 400 and d["ok"] is False
    code, d = post(client, img, {"mode": "bogus", "points": [], "reference": "object",
                                 "ref": {"points": [[0, 0], [9, 0], [9, 9], [0, 9]], "width_mm": 40, "height_mm": 50}})
    assert code == 400


def test_estimate(client):
    r = client.post("/api/estimate", json={"shape": "box", "dims_mm": {"length_mm": 400, "width_mm": 300, "height_mm": 200},
                                           "material": "plywood", "fill_factor": 1})
    d = r.get_json()
    assert d["ok"] and d["volume_l"] == pytest.approx(24.0)
