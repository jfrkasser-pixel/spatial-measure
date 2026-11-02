"""Show how marker candidates are found, step by step.

Usage: python tools/debug_marker.py photo.jpg [outdir] [dict]
Saves: 1_gray.png 2_threshold.png 3_contours.png 4_candidates.png 5_detected.png
"""
import sys
from pathlib import Path

import cv2
import numpy as np

DICTS = {"4x4_50": cv2.aruco.DICT_4X4_50, "5x5_100": cv2.aruco.DICT_5X5_100, "6x6_250": cv2.aruco.DICT_6X6_250}

img = cv2.imread(sys.argv[1])
if img is None:
    sys.exit("Cannot read image: " + sys.argv[1])
out = Path(sys.argv[2] if len(sys.argv) > 2 else "debug_out")
out.mkdir(parents=True, exist_ok=True)
dname = sys.argv[3] if len(sys.argv) > 3 else "4x4_50"
H, W = img.shape[:2]

# 1. greyscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(str(out / "1_gray.png"), gray)

# 2. adaptive threshold (handles uneven light better than one global value)
win = max(3, (int(min(H, W) * 0.03) // 2) * 2 + 1)
thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, win, 7)
cv2.imwrite(str(out / "2_threshold.png"), thr)

# 3. all outlines
contours, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
v = img.copy()
cv2.drawContours(v, contours, -1, (0, 255, 255), 1)
cv2.imwrite(str(out / "3_contours.png"), v)

# 4. keep four-cornered, convex, big-enough, roughly square outlines
min_area = 0.0005 * W * H
cands = []
for c in contours:
    if cv2.contourArea(c) < min_area:
        continue
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.03 * peri, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        continue
    pts = approx.reshape(4, 2).astype(float)
    e = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    if min(e) / max(e) < 0.4:          # too stretched to be a square seen at an angle
        continue
    cands.append(pts)
v = img.copy()
for p in cands:
    cv2.polylines(v, [p.astype(np.int32)], True, (0, 255, 0), 2)
cv2.imwrite(str(out / "4_candidates.png"), v)

# 5. real detector: reads the pattern inside each candidate and checks the dictionary
det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTS[dname]), cv2.aruco.DetectorParameters())
corners, ids, rejected = det.detectMarkers(img)
v = img.copy()
if ids is not None:
    cv2.aruco.drawDetectedMarkers(v, corners, ids)
for r in rejected:
    cv2.polylines(v, [r.reshape(4, 2).astype(np.int32)], True, (0, 0, 255), 1)
cv2.imwrite(str(out / "5_detected.png"), v)

print(f"image {W}x{H}  threshold window {win}px")
print(f"outlines found:            {len(contours)}")
print(f"square-ish candidates:     {len(cands)}   (green in 4_candidates.png)")
print(f"rejected by dictionary:    {len(rejected)}   (red in 5_detected.png)")
if ids is None:
    print("MARKER NOT FOUND for dictionary", dname)
else:
    for c, i in zip(corners, ids.flatten()):
        e = [np.linalg.norm(c[0][k] - c[0][(k + 1) % 4]) for k in range(4)]
        print(f"MARKER id={i} edge={np.mean(e):.1f}px  tilt_ratio={min(e)/max(e):.2f}")
print("images saved in", out.resolve())
