"""Core metrology: marker-based plane calibration, length/area, volume, weight, uncertainty.

All lengths in millimetres unless a name says otherwise. Everything here is pure
functions so it can be unit-tested without a camera.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Material densities, kg/m^3: (low, nominal, high). Indicative handbook ranges.
# Weight from a photo is only as good as the material assumption.
# --------------------------------------------------------------------------
DENSITIES = {
    "water": (997, 997, 997),
    "wood_pine": (350, 500, 600),
    "wood_oak": (600, 750, 900),
    "plywood": (500, 600, 700),
    "cardboard_box_filled": (100, 250, 500),
    "pvc": (1300, 1400, 1450),
    "abs_plastic": (1030, 1050, 1070),
    "aluminium": (2600, 2700, 2800),
    "steel": (7750, 7850, 8050),
    "concrete": (2200, 2400, 2500),
    "sand_dry": (1400, 1600, 1700),
    "rice_bulk": (750, 850, 900),
}

ARUCO_DICTS = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "6x6_250": cv2.aruco.DICT_6X6_250,
}


class MeasureError(ValueError):
    pass


@dataclass
class PlaneCalibration:
    H: np.ndarray            # image px -> plane mm (3x3)
    mm_per_px: float         # mean local scale at the marker
    marker_edge_px: float
    marker_id: int
    corners_px: np.ndarray   # (4,2)
    tilt_ratio: float        # shortest/longest marker edge; 1.0 = fronto-parallel
    corner_unc_px: float = 0.5   # 1-sigma error of each reference corner (px)
    source: str = "aruco"        # "aruco", "aruco_multi" or "reference_object"
    n_markers: int = 1
    residual_mm: float = 0.0     # RMS fit residual of the reference corners (multi-marker)
    marker_polys_px: list | None = None   # every detected marker outline, for masking


# --------------------------------------------------------------------------
# Calibration from a fiducial marker
# --------------------------------------------------------------------------
def detect_marker(img_bgr, dict_name="4x4_50", marker_id=None):
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(img_bgr)
    if ids is None or len(ids) == 0:
        raise MeasureError("No marker found. Keep it flat, fully visible, in focus.")
    ids = ids.flatten()
    if marker_id is not None:
        sel = np.where(ids == marker_id)[0]
        if len(sel) == 0:
            raise MeasureError(f"Marker id {marker_id} not visible (saw {ids.tolist()}).")
        i = int(sel[0])
    else:
        i = 0
    return corners[i].reshape(4, 2).astype(np.float64), int(ids[i])


def calibrate_plane(img_bgr, marker_size_mm, dict_name="4x4_50", marker_id=None) -> PlaneCalibration:
    """Homography mapping image pixels to millimetres on the marker's plane.

    Using a homography (not a single ratio) corrects perspective, so the camera
    may be tilted. Valid ONLY for points lying on the marker's plane.
    """
    corners, mid = detect_marker(img_bgr, dict_name, marker_id)
    s = float(marker_size_mm)
    dst = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float64)
    H, _ = cv2.findHomography(corners, dst)
    if H is None:
        raise MeasureError("Could not compute plane homography.")
    edges = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
    return PlaneCalibration(
        H=H,
        mm_per_px=s / float(np.mean(edges)),
        marker_edge_px=float(np.mean(edges)),
        marker_id=mid,
        corners_px=corners,
        tilt_ratio=float(min(edges) / max(edges)),
    )


def calibrate_from_reference(corners_px, width_mm, height_mm, corner_unc_px=4.0) -> PlaneCalibration:
    """Calibrate from the four hand-tapped corners of a known rectangular object.

    corners_px: 4 points in the order top-left, top-right, bottom-right, bottom-left
    as seen in the image. width_mm is the top/bottom edge, height_mm the left/right edge.
    Taps are less exact than an ArUco detection, hence the larger default corner error.
    The object's top face must lie in the same plane as what you measure.
    """
    c = np.asarray(corners_px, dtype=np.float64).reshape(-1, 2)
    if c.shape != (4, 2):
        raise MeasureError("Reference needs exactly 4 corner points (TL, TR, BR, BL).")
    w, h = float(width_mm), float(height_mm)
    if w <= 0 or h <= 0:
        raise MeasureError("Reference width and height must be positive.")
    # Corners must form a convex quadrilateral in the given order (no bow-tie).
    def cross2(a, b):
        return a[0] * b[1] - a[1] * b[0]
    cross = [cross2(c[(i + 1) % 4] - c[i], c[(i + 2) % 4] - c[(i + 1) % 4]) for i in range(4)]
    if not (all(x > 0 for x in cross) or all(x < 0 for x in cross)):
        raise MeasureError("Reference corners are not in order (TL, TR, BR, BL) or not convex.")
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    H, _ = cv2.findHomography(c, dst)
    if H is None:
        raise MeasureError("Could not compute plane homography from the reference.")
    top, right, bottom, left = [np.linalg.norm(c[(i + 1) % 4] - c[i]) for i in range(4)]
    tilt = min(min(top, bottom) / max(top, bottom), min(left, right) / max(left, right))
    mean_edge = float(np.mean([top, right, bottom, left]))
    mm_per_px = float(np.mean([w / top, w / bottom, h / left, h / right]))
    return PlaneCalibration(H=H, mm_per_px=mm_per_px, marker_edge_px=mean_edge, marker_id=-1,
                            corners_px=c, tilt_ratio=float(tilt),
                            corner_unc_px=float(corner_unc_px), source="reference_object")


def to_plane(H, pts_px):
    pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def distance_mm(cal: PlaneCalibration, p1_px, p2_px) -> float:
    a, b = to_plane(cal.H, [p1_px, p2_px])
    return float(np.linalg.norm(a - b))


def polygon_area_mm2(cal: PlaneCalibration, pts_px) -> float:
    """Shoelace area of a polygon whose vertices lie on the marker plane."""
    if len(pts_px) < 3:
        raise MeasureError("Need at least 3 points for an area.")
    p = to_plane(cal.H, pts_px)
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_perimeter_mm(cal: PlaneCalibration, pts_px) -> float:
    p = to_plane(cal.H, pts_px)
    return float(np.sum(np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1)))


# --------------------------------------------------------------------------
# Volume (SI: result in m^3 and litres)
# --------------------------------------------------------------------------
def volume_m3(shape: str, **d) -> float:
    """Volume from simple solids. Inputs in mm.

    prism:    base_area_mm2, height_mm   (any footprint polygon extruded)
    box:      length_mm, width_mm, height_mm
    cylinder: diameter_mm, height_mm
    sphere:   diameter_mm
    cone:     diameter_mm, height_mm
    """
    mm3_to_m3 = 1e-9
    if shape == "prism":
        v = d["base_area_mm2"] * d["height_mm"]
    elif shape == "box":
        v = d["length_mm"] * d["width_mm"] * d["height_mm"]
    elif shape == "cylinder":
        v = math.pi * (d["diameter_mm"] / 2) ** 2 * d["height_mm"]
    elif shape == "sphere":
        v = 4 / 3 * math.pi * (d["diameter_mm"] / 2) ** 3
    elif shape == "cone":
        v = math.pi * (d["diameter_mm"] / 2) ** 2 * d["height_mm"] / 3
    else:
        raise MeasureError(f"Unknown shape '{shape}'.")
    return v * mm3_to_m3


# --------------------------------------------------------------------------
# Uncertainty (GUM-style first-order propagation, k=2 -> ~95 %)
# --------------------------------------------------------------------------
def length_uncertainty_mm(cal: PlaneCalibration, length_mm, tap_px=3.0, corner_px=None):
    """Combined standard uncertainty of a length on the plane.

    Components (assumed, to be replaced by your own calibration data):
      - scale: marker corner localisation, relative ~ corner_px*sqrt(2)/edge_px
      - point picking: tap_px on both end points
      - tilt: extra 1 % per 0.1 of lost tilt_ratio (conservative placeholder)
    """
    if corner_px is None:
        corner_px = cal.corner_unc_px
    rel_scale = corner_px * math.sqrt(2) / cal.marker_edge_px
    u_pick = tap_px * math.sqrt(2) * cal.mm_per_px
    rel_tilt = 0.1 * (1.0 - cal.tilt_ratio)
    u_scale = rel_scale * length_mm
    u_tilt = rel_tilt * length_mm
    return math.sqrt(u_scale**2 + u_pick**2 + u_tilt**2 + cal.residual_mm**2)


def mass_kg(volume: float, material: str, fill_factor: float = 1.0, density_override=None):
    """Mass with a low/nominal/high band from the density range.

    fill_factor: fraction of the bounding volume that is solid material
    (hollow boxes, packed goods...). 1.0 = fully solid.
    """
    lo, nom, hi = (density_override,) * 3 if density_override else DENSITIES[material]
    f = max(0.0, min(1.0, fill_factor))
    return {
        "nominal_kg": volume * nom * f,
        "low_kg": volume * lo * f,
        "high_kg": volume * hi * f,
        "density_kg_m3": nom,
        "fill_factor": f,
    }


def combine_relative(*rel):
    return math.sqrt(sum(r * r for r in rel))


# --------------------------------------------------------------------------
# Multi-marker printed target (A4 sheet, four markers at the corners)
# --------------------------------------------------------------------------
# Top-left corner of each marker, in mm, on an upright 210 x 297 mm sheet.
A4_TARGET = {
    "dict": "4x4_50",
    "marker_size_mm": 40.0,
    "sheet_mm": (210.0, 297.0),
    "markers": {0: (10.0, 10.0), 1: (160.0, 10.0), 2: (160.0, 247.0), 3: (10.0, 247.0)},
}


def _all_markers(img_bgr, dict_name):
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(img_bgr)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) for c, i in zip(corners, ids.flatten())}


def calibrate_multi(img_bgr, target=None) -> PlaneCalibration:
    """Fit one homography to every detected marker of a known printed target.

    More markers spread over a larger area give a better scale, less extrapolation
    error and a residual that tells you how flat/accurate the print and photo are.
    """
    t = target or A4_TARGET
    s = float(t["marker_size_mm"])
    found = _all_markers(img_bgr, t["dict"])
    used = {i: c for i, c in found.items() if i in t["markers"]}
    if not used:
        raise MeasureError("No marker of the target found. Show the printed sheet, flat and in focus.")
    src, dst, polys, tilts, edges = [], [], [], [], []
    for i, c in used.items():
        x, y = t["markers"][i]
        src.append(c)
        dst.append([[x, y], [x + s, y], [x + s, y + s], [x, y + s]])
        polys.append(c)
        e = [np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)]
        tilts.append(min(e) / max(e))
        edges.append(np.mean(e))
    src = np.vstack(src)
    dst = np.vstack(dst).astype(np.float64)
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        raise MeasureError("Could not fit the target homography.")
    res = np.linalg.norm(to_plane(H, src) - dst, axis=1)
    pair = [np.linalg.norm(a - b) for a in src for b in src]
    baseline = float(max(pair)) if len(used) > 1 else float(np.mean(edges))
    return PlaneCalibration(
        H=H, mm_per_px=s / float(np.mean(edges)), marker_edge_px=baseline,
        marker_id=min(used), corners_px=np.vstack(polys[:1]), tilt_ratio=float(min(tilts)),
        source="aruco_multi", n_markers=len(used),
        residual_mm=float(np.sqrt(np.mean(res**2))), marker_polys_px=[p.tolist() for p in polys],
    )


# --------------------------------------------------------------------------
# Lens distortion (chessboard calibration, then undistort image and taps)
# --------------------------------------------------------------------------
import json  # noqa: E402


@dataclass
class LensModel:
    K: np.ndarray
    dist: np.ndarray
    size: tuple            # (width, height) of the calibration images
    rms_px: float = 0.0

    def for_size(self, w, h):
        """Same lens at another capture resolution (same field of view assumed)."""
        sx, sy = w / self.size[0], h / self.size[1]
        K = self.K.copy()
        K[0, 0] *= sx; K[0, 2] *= sx; K[1, 1] *= sy; K[1, 2] *= sy
        return LensModel(K, self.dist.copy(), (w, h), self.rms_px)

    def to_json(self):
        return json.dumps({"K": self.K.tolist(), "dist": self.dist.ravel().tolist(),
                           "size": list(self.size), "rms_px": self.rms_px}, indent=1)

    @staticmethod
    def from_json(text):
        d = json.loads(text)
        return LensModel(np.array(d["K"], dtype=np.float64), np.array(d["dist"], dtype=np.float64),
                         tuple(d["size"]), float(d.get("rms_px", 0.0)))


def board_object_points(pattern=(9, 6), square_mm=25.0):
    """3D points (z=0) of the inner chessboard corners, in mm."""
    p = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    p[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_mm
    return p


def calibrate_lens(obj_pts, img_pts, size) -> LensModel:
    """obj_pts / img_pts: lists (one entry per photo) of Nx3 / Nx2 float32 arrays."""
    if len(obj_pts) < 5:
        raise MeasureError("Need at least 5 photos of the chessboard (10+ is better).")
    rms, K, dist, _, _ = cv2.calibrateCamera(
        [np.asarray(o, np.float32) for o in obj_pts],
        [np.asarray(i, np.float32).reshape(-1, 1, 2) for i in img_pts], tuple(size), None, None)
    return LensModel(K, dist.ravel(), tuple(size), float(rms))


def undistort_image(img, lens: LensModel):
    h, w = img.shape[:2]
    lm = lens.for_size(w, h) if (w, h) != tuple(lens.size) else lens
    return cv2.undistort(img, lm.K, lm.dist, None, lm.K)


def undistort_points(pts_px, lens: LensModel, img_size):
    w, h = img_size
    lm = lens.for_size(w, h) if (w, h) != tuple(lens.size) else lens
    p = np.asarray(pts_px, np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(p, lm.K, lm.dist, P=lm.K).reshape(-1, 2)


def distort_points(pts_px, lens: LensModel, img_size):
    """Inverse of undistort_points: where ideal pixel positions appear in the raw photo."""
    w, h = img_size
    lm = lens.for_size(w, h) if (w, h) != tuple(lens.size) else lens
    p = np.asarray(pts_px, np.float64).reshape(-1, 2)
    n = (p - lm.K[:2, 2]) / np.array([lm.K[0, 0], lm.K[1, 1]])
    d, _ = cv2.projectPoints(np.hstack([n, np.ones((len(n), 1))]), np.zeros(3), np.zeros(3), lm.K, lm.dist)
    return d.reshape(-1, 2)


# --------------------------------------------------------------------------
# Automatic outline: one tap inside the object, GrabCut finds its silhouette
# --------------------------------------------------------------------------
def detect_outline(img_bgr, cal: PlaneCalibration, seed_px, max_dim=800):
    """Segment the object under seed_px and measure its outline on the marker plane.

    Returns polygon (image px), area (mm2), perimeter (mm) and the oriented
    bounding rectangle (long, short in mm). The silhouette of a tall object includes
    its sides, so area/size are exact only for flat objects or footprints.
    """
    H0, W0 = img_bgr.shape[:2]
    f = min(1.0, max_dim / max(H0, W0))
    im = cv2.resize(img_bgr, (int(W0 * f), int(H0 * f)), interpolation=cv2.INTER_AREA) if f < 1 else img_bgr.copy()
    h, w = im.shape[:2]
    sx, sy = seed_px[0] * f, seed_px[1] * f
    if not (0 <= sx < w and 0 <= sy < h):
        raise MeasureError("Tap inside the image.")

    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    b = max(2, int(0.02 * min(h, w)))
    mask[:b, :] = mask[-b:, :] = mask[:, :b] = mask[:, -b:] = cv2.GC_BGD
    polys = cal.marker_polys_px or [cal.corners_px.tolist()]
    for poly in polys:
        pp = (np.array(poly) * f).astype(np.int32)
        cv2.fillConvexPoly(mask, cv2.convexHull(pp), cv2.GC_BGD)
        cv2.polylines(mask, [pp], True, cv2.GC_BGD, max(3, int(0.01 * min(h, w))))
    r_big, r_small = int(0.10 * min(h, w)), max(3, int(0.03 * min(h, w)))
    cv2.circle(mask, (int(sx), int(sy)), r_big, cv2.GC_PR_FGD, -1)
    cv2.circle(mask, (int(sx), int(sy)), r_small, cv2.GC_FGD, -1)
    try:
        bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(im, mask, None, bg, fg, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error as e:
        raise MeasureError(f"Segmentation failed: {e}")
    fgm = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgm = cv2.morphologyEx(cv2.morphologyEx(fgm, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(fgm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise MeasureError("No object found at that point. Tap inside it, on a plain background.")
    inside = [c for c in cnts if cv2.pointPolygonTest(c, (sx, sy), False) >= 0]
    cnt = max(inside or cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.0005 * w * h:
        raise MeasureError("Detected region is too small. Tap in the middle of the object.")

    poly_px = cnt.reshape(-1, 2).astype(np.float64) / f
    plane = to_plane(cal.H, poly_px)
    x, y = plane[:, 0], plane[:, 1]
    area = float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)
    perim = float(np.sum(np.linalg.norm(plane - np.roll(plane, -1, axis=0), axis=1)))
    (_, _), (rw, rh), _ = cv2.minAreaRect(plane.astype(np.float32))
    approx = cv2.approxPolyDP(cnt, 0.004 * cv2.arcLength(cnt, True), True).reshape(-1, 2) / f
    return {"polygon_px": approx.tolist(), "area_mm2": area, "perimeter_mm": perim,
            "rect_long_mm": float(max(rw, rh)), "rect_short_mm": float(min(rw, rh))}
