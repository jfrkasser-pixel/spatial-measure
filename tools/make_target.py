"""Write printable sheets.  Usage: python tools/make_target.py [outdir]
  a4_target.png      four 40 mm ArUco markers (use with Reference = A4 target)
  chessboard.png     9x6 inner-corner board, 25 mm squares (for lens calibration)
Print both at 100 % (actual size, NOT fit-to-page) on flat paper and check with a ruler."""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from measure import targets  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out / "a4_target.png"), targets.render_a4_target())
cv2.imwrite(str(out / "chessboard.png"), targets.render_chessboard())
print("wrote", out / "a4_target.png", "and", out / "chessboard.png", "(A4, 300 dpi; chessboard prints LANDSCAPE)")
