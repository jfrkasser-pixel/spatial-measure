"""Calibrate the phone lens from photos of the printed chessboard.

Usage: python tools/calibrate_lens.py photo1.jpg photo2.jpg ... [--out lens.json] [--square 25] [--cols 9] [--rows 6]

Take 10-20 photos with the SAME phone, camera and zoom as the measuring app, at the SAME
resolution (use the web page's Capture on the chessboard or the phone camera), filling
different parts of the frame, tilted differently. The result is lens.json; put it in the
project folder and restart the server: every measurement is then undistorted automatically."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from measure import core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("photos", nargs="+")
ap.add_argument("--out", default="lens.json")
ap.add_argument("--square", type=float, default=25.0, help="square size in mm")
ap.add_argument("--cols", type=int, default=9, help="inner corners along the long side")
ap.add_argument("--rows", type=int, default=6, help="inner corners along the short side")
a = ap.parse_args()

pattern = (a.cols, a.rows)
obj = core.board_object_points(pattern, a.square)
objs, imgs, size = [], [], None
for f in a.photos:
    im = cv2.imread(f)
    if im is None:
        print("skip (unreadable):", f)
        continue
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if size is None:
        size = (g.shape[1], g.shape[0])
    elif (g.shape[1], g.shape[0]) != size:
        print("skip (different resolution):", f)
        continue
    ok, c = cv2.findChessboardCornersSB(g, pattern)
    if not ok:
        ok, c = cv2.findChessboardCorners(g, pattern)
        if ok:
            c = cv2.cornerSubPix(g, c, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    print(("ok   " if ok else "no board"), f)
    if ok:
        objs.append(obj)
        imgs.append(c.reshape(-1, 2))

lens = core.calibrate_lens(objs, imgs, size)
Path(a.out).write_text(lens.to_json())
print(f"\n{len(objs)} photos used, image size {size}")
print(f"reprojection error {lens.rms_px:.2f} px   (good: below 0.7)")
print(f"focal length {lens.K[0, 0]:.0f} px, distortion k1 {lens.dist[0]:+.3f}")
print("saved", a.out)
