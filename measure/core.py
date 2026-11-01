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
def length_uncertainty_mm(cal: PlaneCalibration, length_mm, tap_px=3.0, corner_px=0.5):
    """Combined standard uncertainty of a length on the plane.

    Components (assumed, to be replaced by your own calibration data):
      - scale: marker corner localisation, relative ~ corner_px*sqrt(2)/edge_px
      - point picking: tap_px on both end points
      - tilt: extra 1 % per 0.1 of lost tilt_ratio (conservative placeholder)
    """
    rel_scale = corner_px * math.sqrt(2) / cal.marker_edge_px
    u_pick = tap_px * math.sqrt(2) * cal.mm_per_px
    rel_tilt = 0.1 * (1.0 - cal.tilt_ratio)
    u_scale = rel_scale * length_mm
    u_tilt = rel_tilt * length_mm
    return math.sqrt(u_scale**2 + u_pick**2 + u_tilt**2)


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
