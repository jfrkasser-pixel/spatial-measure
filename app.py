"""Flask server: serves the phone UI and the measurement API over HTTPS."""
import json
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


@app.post("/api/measure")
def measure():
    """multipart: frame (jpeg), payload (json)

    payload: {mode: "length"|"area", points: [[x,y],...] in image pixels,
              marker_size_mm: float, dict: "4x4_50", marker_id: int|null}
    """
    try:
        if "frame" not in request.files:
            raise core.MeasureError("No frame uploaded.")
        p = json.loads(request.form.get("payload", "{}"))
        img = _decode(request.files["frame"])
        cal = core.calibrate_plane(img, float(p["marker_size_mm"]), p.get("dict", "4x4_50"), p.get("marker_id"))
        pts = p.get("points", [])
        out = {
            "ok": True,
            "marker_id": cal.marker_id,
            "mm_per_px": cal.mm_per_px,
            "tilt_ratio": cal.tilt_ratio,
            "warnings": [],
            "marker_corners_px": cal.corners_px.tolist(),
        }
        if cal.marker_edge_px < 60:
            out["warnings"].append("Marker is small in the image (<60 px): move closer or use a bigger marker.")
        if cal.tilt_ratio < 0.7:
            out["warnings"].append("Strong tilt: accuracy drops. Shoot closer to top-down.")

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
        else:
            raise core.MeasureError("Unknown mode.")
        return jsonify(out)
    except (core.MeasureError, KeyError, ValueError) as e:
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
    cert, key = os.environ.get("TLS_CERT", "certs/cert.pem"), os.environ.get("TLS_KEY", "certs/key.pem")
    port = int(os.environ.get("PORT", "5000"))
    print(f"\nOpen on your phone:  https://{_lan_ip()}:{port}\n")
    if os.environ.get("NO_TLS"):
        app.run(host="0.0.0.0", port=port, debug=False)
    else:
        app.run(host="0.0.0.0", port=port, ssl_context=(cert, key), debug=False)
