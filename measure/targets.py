"""Printable targets: the A4 four-marker sheet and the lens-calibration chessboard.

Print both at 100 % scale ("actual size", no fit-to-page) and check with a ruler.
"""
import cv2
import numpy as np

from . import core


def render_a4_target(px_per_mm=11.81):
    """A4 sheet (210 x 297 mm) with four 40 mm ArUco markers. 11.81 px/mm = 300 dpi."""
    t = core.A4_TARGET
    W, H = (int(round(v * px_per_mm)) for v in t["sheet_mm"])
    img = np.full((H, W), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(core.ARUCO_DICTS[t["dict"]])
    s = int(round(t["marker_size_mm"] * px_per_mm))
    for mid, (x, y) in t["markers"].items():
        m = cv2.aruco.generateImageMarker(d, mid, s)
        x0, y0 = int(round(x * px_per_mm)), int(round(y * px_per_mm))
        img[y0:y0 + s, x0:x0 + s] = m
    # tiny caption so the printed sheet identifies itself
    cv2.putText(img, "A4 target: markers 0-3, 40 mm. Print at 100%.", (int(15 * px_per_mm), int(150 * px_per_mm)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * px_per_mm / 6, 0, max(1, int(px_per_mm / 6)))
    return img


def render_chessboard(pattern=(9, 6), square_mm=25.0, px_per_mm=11.81):
    """Chessboard with (pattern[0]+1) x (pattern[1]+1) squares, centred on a LANDSCAPE A4 sheet
    (297 x 210 mm; the default 10 x 7 squares of 25 mm are 250 x 175 mm)."""
    W, H = int(297 * px_per_mm), int(210 * px_per_mm)
    img = np.full((H, W), 255, np.uint8)
    nx, ny = pattern[0] + 1, pattern[1] + 1
    sq = square_mm * px_per_mm
    ox, oy = (W - nx * sq) / 2, (H - ny * sq) / 2
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                x0, y0 = int(round(ox + i * sq)), int(round(oy + j * sq))
                img[y0:int(round(oy + (j + 1) * sq)), x0:int(round(ox + (i + 1) * sq))] = 0
    return img
