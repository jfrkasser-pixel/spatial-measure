"""Write a print-ready ArUco marker as SVG with exact millimetre size.

usage: python tools/make_marker.py [id] [size_mm] [dict]    default: 0 50 4x4_50
Print the SVG at 100 % (no 'fit to page'), then check the black square with a ruler.
"""
import sys
import cv2

mid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
size = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
name = sys.argv[3] if len(sys.argv) > 3 else "4x4_50"
dicts = {"4x4_50": (cv2.aruco.DICT_4X4_50, 4), "5x5_100": (cv2.aruco.DICT_5X5_100, 5),
         "6x6_250": (cv2.aruco.DICT_6X6_250, 6)}
dd, n = dicts[name]
cells = n + 2                                  # inner grid + 1 cell black border each side
img = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(dd), mid, cells * 10)
cell = size / cells
margin = size * 0.25                           # white quiet zone
total = size + 2 * margin
rects = []
for r in range(cells):
    for c in range(cells):
        if img[r * 10 + 5, c * 10 + 5] < 128:
            rects.append(f'<rect x="{margin + c * cell:.3f}" y="{margin + r * cell:.3f}" '
                         f'width="{cell:.3f}" height="{cell:.3f}"/>')
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{total}mm" height="{total}mm" '
       f'viewBox="0 0 {total} {total}"><rect width="100%" height="100%" fill="white"/>'
       f'<g fill="black" shape-rendering="crispEdges">' + "".join(rects) + '</g></svg>')
fn = f"marker_{name}_id{mid}_{size:g}mm.svg"
open(fn, "w").write(svg)
print(f"wrote {fn}: black square {size:g} mm, page {total:g} mm")
