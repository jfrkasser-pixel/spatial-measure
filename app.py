"""Flask server: serves the phone UI and the measurement API over HTTPS."""
import json
import math
import os
import socket

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from measure import core

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB per upload


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/materials")
def materials():
    return jsonify({k: {"low": v[0], "nominal": v[1], "high": v[2]} for k, v in core.DENSITIES.items()})


def _decode(file_storage):
    data = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise core.MeasureError("Image could not be decoded.")
    return img


LENS = None
_lens_path = os.environ.get("LENS_JSON", "lens.json")
if os.path.exists(_lens_path):
    try:
        LENS = core.LensModel.from_json(open(_lens_path).read())
        print(f"Lens calibration loaded from {_lens_path} (rms {LENS.rms_px:.2f} px)")
    except (ValueError, KeyError) as e:
        print("Ignoring bad lens file:", e)


@app.get("/api/lens")
def lens_status():
    return jsonify({"loaded": LENS is not None, "rms_px": LENS.rms_px if LENS else None})


@app.post("/api/measure")
def measure():
    """multipart: frame (jpeg), payload (json)

    payload: {mode: "length"|"area"|"outline", points: [[x,y],...] in image pixels,
              reference: "aruco" (default) | "a4" | "object",
              marker_size_mm, dict, marker_id          (reference "aruco")
              ref: {points:[4], width_mm, height_mm}   (reference "object")}
    """
    try:
        if "frame" not in request.files:
            raise core.MeasureError("No frame uploaded.")
        p = json.loads(request.form.get("payload", "{}"))
        img = _decode(request.files["frame"])
        size = (img.shape[1], img.shape[0])
        pts = p.get("points", [])
        ref = p.get("ref")
        if LENS is not None:
            img = core.undistort_image(img, LENS)
            if pts:
                pts = core.undistort_points(pts, LENS, size).tolist()
            if ref:
                ref = dict(ref, points=core.undistort_points(ref["points"], LENS, size).tolist())

        reference = p.get("reference") or ("object" if ref else "aruco")
        if reference == "object":
            if not ref:
                raise core.MeasureError("Reference object corners missing.")
            cal = core.calibrate_from_reference(ref["points"], ref["width_mm"], ref["height_mm"])
        elif reference == "a4":
            cal = core.calibrate_multi(img)
        else:
            cal = core.calibrate_plane(img, float(p["marker_size_mm"]), p.get("dict", "4x4_50"), p.get("marker_id"))

        def to_raw(px):
            """Draw positions on the photo the phone showed (undistorted -> raw)."""
            return core.distort_points(px, LENS, size).tolist() if LENS is not None else np.asarray(px).tolist()

        out = {
            "ok": True,
            "source": cal.source,
            "lens_corrected": LENS is not None,
            "n_markers": cal.n_markers,
            "marker_id": cal.marker_id,
            "mm_per_px": cal.mm_per_px,
            "tilt_ratio": cal.tilt_ratio,
            "fit_residual_mm": cal.residual_mm,
            "warnings": [],
            "marker_polys_px": [to_raw(q) for q in (cal.marker_polys_px or [cal.corners_px.tolist()])],
        }
        if cal.marker_edge_px < 60:
            out["warnings"].append("Marker is small in the image (<60 px): move closer or use a bigger marker.")
        if cal.source == "reference_object":
            out["warnings"].append("Hand-tapped reference: valid only for points at the same height as the reference's top face.")
        if cal.tilt_ratio < 0.7:
            out["warnings"].append("Strong tilt: accuracy drops. Shoot closer to top-down.")
        if LENS is None:
            out["warnings"].append("No lens calibration loaded: expect 1-3 % extra error near the image edges.")

        mode = p.get("mode", "length")
        if mode == "length":
            if len(pts) != 2:
                raise core.MeasureError("Length needs exactly 2 points.")
            L = core.distance_mm(cal, pts[0], pts[1])
            u = core.length_uncertainty_mm(cal, L)
            out.update(length_mm=L, u_mm=u, U_mm_k2=2 * u)
        elif mode == "area":
            if len(pts) < 3:
                raise core.MeasureError("Area needs at least 3 points.")
            A = core.polygon_area_mm2(cal, pts)
            P = core.polygon_perimeter_mm(cal, pts)
            # area relative uncertainty ~ 2 x equivalent-length relative uncertainty
            Leq = A ** 0.5
            rel = 2 * core.length_uncertainty_mm(cal, Leq) / max(Leq, 1e-9)
            out.update(area_mm2=A, area_m2=A * 1e-6, perimeter_mm=P, rel_u=rel, U_rel_k2=2 * rel)
        elif mode == "outline":
            if len(pts) != 1:
                raise core.MeasureError("Outline needs exactly 1 tap inside the object.")
            o = core.detect_outline(img, cal, pts[0])
            A = o["area_mm2"]
            Leq = A ** 0.5
            rel = 2 * core.length_uncertainty_mm(cal, Leq) / max(Leq, 1e-9)
            rel = math.sqrt(rel ** 2 + 0.03 ** 2)          # + edge-finding allowance (assumed 3 %)
            out.update(area_mm2=A, area_m2=A * 1e-6, perimeter_mm=o["perimeter_mm"],
                       rect_long_mm=o["rect_long_mm"], rect_short_mm=o["rect_short_mm"],
                       U_rel_k2=rel, polygon_px=to_raw(o["polygon_px"]))
            out["warnings"].append("Outline is the silhouette seen by the camera: exact for flat objects or footprints, "
                                   "overstated for tall objects.")
        else:
            raise core.MeasureError("Unknown mode.")
        return jsonify(out)
    except (core.MeasureError, KeyError, ValueError, TypeError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.post("/api/estimate")
def estimate():
    """json: {shape, dims_mm:{...}, material, fill_factor, rel_u_dims}"""
    try:
        p = request.get_json(force=True)
        V = core.volume_m3(p["shape"], **p["dims_mm"])
        rel_dims = float(p.get("rel_u_dims", 0.03))
        # volume ~ product of up to 3 lengths -> relative u adds in quadrature
        rel_V = core.combine_relative(rel_dims, rel_dims, rel_dims)
        m = core.mass_kg(V, p["material"], float(p.get("fill_factor", 1.0)), p.get("density_override"))
        spread = (m["high_kg"] - m["low_kg"]) / 2 / max(m["nominal_kg"], 1e-12)
        rel_m = core.combine_relative(rel_V, spread)
        return jsonify({
            "ok": True,
            "volume_m3": V, "volume_l": V * 1000,
            "U_rel_volume_k2": 2 * rel_V,
            "mass": m,
            "U_rel_mass_k2": 2 * rel_m,
            "note": "Estimate from an image; not legal-for-trade. Weight assumes homogeneous material.",
        })
    except (core.MeasureError, KeyError, ValueError, TypeError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400


def _lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    if os.environ.get("NO_TLS"):
        # Plain http: use "Take / load photo" (phone camera app) or the Chrome insecure-origin flag.
        print(f"\nOpen on your phone:  http://{_lan_ip()}:{port}   (no TLS)\n")
        app.run(host="0.0.0.0", port=port, debug=False)
    else:
        cert, key = os.environ.get("TLS_CERT", "certs/cert.pem"), os.environ.get("TLS_KEY", "certs/key.pem")
        print(f"\nOpen on your phone:  https://{_lan_ip()}:{port}\n")
        app.run(host="0.0.0.0", port=port, ssl_context=(cert, key), debug=False)
