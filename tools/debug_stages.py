"""Show how a photo goes from pixels to a detected ArUco marker, step by step.

usage: python tools/debug_stages.py photo.jpg [outdir]
Writes 1_gray.png 2_threshold.png 3_contours.png 4_quads.png 5_aruco.png
"""
import sys, os
import cv2
import numpy as np

path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "debug_out"
os.makedirs(out, exist_ok=True)

img = cv2.imread(path)
if img is None:
    sys.exit(f"cannot read {path}")
h, w = img.shape[:2]

# 1. greyscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(f"{out}/1_gray.png", gray)

# 2. adaptive threshold (ArUco uses windows 3..23 step 10, constant 7; one window shown)
win = 23
thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY_INV, win, 7)
cv2.imwrite(f"{out}/2_threshold.png", thr)

# 3. outlines
contours, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
vis = img.copy()
cv2.drawContours(vis, contours, -1, (0, 200, 255), 1)
cv2.imwrite(f"{out}/3_contours.png", vis)

# 4. keep convex shapes with exactly 4 corners and a sensible size
min_perim = 0.03 * max(h, w)
quads = []
for c in contours:
    p = cv2.arcLength(c, True)
    if p < min_perim:
        continue
    approx = cv2.approxPolyDP(c, 0.05 * p, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        quads.append(approx.reshape(4, 2))
vis = img.copy()
for q in quads:
    cv2.polylines(vis, [q.astype(np.int32)], True, (0, 255, 0), 2)
cv2.imwrite(f"{out}/4_quads.png", vis)

# 5. final detection with the real detector, trying every supported dictionary
found = False
for name, d in [("4x4_50", cv2.aruco.DICT_4X4_50), ("5x5_100", cv2.aruco.DICT_5X5_100),
                ("6x6_250", cv2.aruco.DICT_6X6_250)]:
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(d), cv2.aruco.DetectorParameters())
    corners, ids, rejected = det.detectMarkers(img)
    if ids is not None:
        vis = img.copy()
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        cv2.imwrite(f"{out}/5_aruco.png", vis)
        edge = np.mean([np.linalg.norm(corners[0][0][i] - corners[0][0][(i + 1) % 4]) for i in range(4)])
        print(f"dictionary {name}: marker id(s) {ids.flatten().tolist()}, edge about {edge:.0f} px")
        found = True
        break
print(f"image {w}x{h} | contours {len(contours)} | four-corner shapes {len(quads)}")
if not found:
    print("no marker found: look at 2_threshold.png (is the marker border solid black?) and 4_quads.png (is it outlined green?)")
print(f"images saved in {out}/")
