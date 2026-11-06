# Spatial Measure: blueprint

Smartphone camera app that measures length, area, volume and estimates mass, with a stated uncertainty.
Phase 1 (this repo): web app on a Debian 13 server, phone used as camera and touch UI.

## 1. Principle

A printed fiducial marker of known size lies on the same plane as the thing measured.
Its four corners give a homography H (image px to plane mm). Any point tapped on that plane maps to mm,
so camera distance and camera tilt cancel out. Valid only for points on the marker plane.

```
phone (getUserMedia, HTTPS) -> frozen frame + taps -> /api/measure
   -> detect ArUco -> H -> mm coordinates -> length / shoelace area / perimeter
   -> /api/estimate: shape model -> volume (m3) -> x density x fill -> mass band
   -> GUM-style uncertainty (k=2)
```

## 2. Pipeline

1. Capture a still (not live video) so taps and detection use the same pixels.
2. Detect marker, compute H, report scale (mm/px) and tilt ratio. Reject if marker < 60 px or not found.
3. Map taps through H. Length = Euclidean distance. Area = shoelace on mapped points.
4. Height of a 3D object cannot be read from a top-down plane. Take a second photo of a side face
   with the marker on that face (or enter the value). Volume = footprint x height, or a solid model.
5. Mass = V x rho x fill. rho comes from a material table with low/nominal/high.
6. Report value, uncertainty, warnings.

## 3. Volume models

| Shape | Inputs | Formula |
|---|---|---|
| box | L, W, H | L W H |
| prism | footprint area, H | A H |
| cylinder | d, H | pi (d/2)^2 H |
| sphere | d | 4/3 pi (d/2)^3 |
| cone | d, H | 1/3 pi (d/2)^2 H |

Irregular objects: bounding box times a fill factor is a rough estimate only.
Better later: multi-view photogrammetry, or LiDAR/depth on native apps.

## 4. Weight

Weight from vision is volume times an assumed density. Errors: material identity (density spread),
hollowness (fill factor), shape approximation. Always show the low/high band, never a single number as truth.
Mass in kg; weight force (N) = m g, g = 9.80665 m/s2.

## 5. Norms and references

| Topic | Reference | Use here |
|---|---|---|
| Units | SI brochure (BIPM); ISO 80000-1 | mm, m2, m3, L, kg only in output |
| Uncertainty | JCGM 100 (GUM) | combine components in quadrature, report k=2 (about 95 %) |
| Accuracy terms | ISO 5725 | trueness vs precision when you validate |
| Lab calibration | ISO/IEC 17025 | needed if you want traceable results |
| Dimensional systems | ISO 10360 family | reference for acceptance tests of coordinate systems |
| Geodetic frame | WGS84 / ECEF / ENU | optional geolocation of measurements |
| Legal for trade | OIML R 76 (scales), national type approval | a camera estimate is NOT legal for trade |

Check the current editions of each standard before claiming compliance. This app is an estimator, not a certified instrument.

## 6. Uncertainty model (current, assumed)

- Scale: marker corner localisation 0.5 px per corner, relative about 0.7 / edge_px.
- Point picking: 3 px per tap.
- Tilt: 0.1 x (1 - tilt_ratio) relative, placeholder.
- Volume: dimension relative uncertainties combined in quadrature (default 3 % each).
- Mass: volume term combined with density half-range.

Replace these assumptions with measured values: photograph a calibrated gauge block set or steel rule
at many distances and angles, compute residuals, fit.

## 7. Known limits

- Points off the marker plane (the top of a tall box) are mis-scaled. Use side views.
- Lens distortion not corrected. Calibrate the camera (cv2.calibrateCamera, chessboard) and undistort for sub-percent work.
- Glossy or blurred markers fail detection. Print matte, flat, not scaled by the printer (verify with a ruler).
- Browser JPEG capture is not raw; autofocus and digital zoom change nothing for the homography but blur hurts corners.

## 8. Working modes (current)

| Reference | When | Accuracy |
|---|---|---|
| Printed ArUco marker (single) | small scenes, one marker beside the objects | about 1 % |
| A4 target (4 markers) | desks, floors, groups of objects: markers surround the area | about 0.5-1 %, residual shown |
| Known object, 4 taps | no printed marker available | 3-5 % and only at the object's own height |

Measuring modes: length (2 taps), area (polygon taps), auto outline (1 tap inside the object).
Outline uses GrabCut from the tap, then maps the silhouette through the plane homography.
It is exact for flat objects and footprints, and overstates tall objects (the silhouette includes the sides).

### A4 target
    python tools/make_target.py prints
Print `prints/a4_target.png` at 100 % on A4 (actual size, check the 40 mm markers with a ruler).
Lay it flat on the surface, objects on or beside it, and choose Reference = A4 target.
Any one to four markers visible is enough; more markers over a wider area give a better scale.

### Lens calibration
1. Print `prints/chessboard.png` landscape at 100 % (25 mm squares, 9 x 6 inner corners) on a flat board.
2. Take 10-20 photos with the same phone, camera and resolution as the app, different positions and tilts, board filling different parts of the frame.
3. `python tools/calibrate_lens.py photos/*.jpg --out lens.json` (aim for reprojection error under 0.7 px).
4. Put `lens.json` next to `app.py` and restart. Every photo and tap is then undistorted automatically.
Without it, expect 1-3 % extra error near the image edges on a typical phone.

### Height from a side photo
A single top-down photo cannot give height. Photograph the object from the side with the marker standing in the same vertical plane as its front face, measure the vertical distance in Length mode, press *Use as: Height*.
The perspective error of anything off the marker plane is about (height off the plane) / (camera distance).

### Take / load photo
The button opens the phone's own camera app. It works over plain http with no certificate or browser flag (the live camera view needs HTTPS or a trusted origin).

## 9. Roadmap

1. Several photos of one object combined into a 3D shape (photogrammetry with the markers fixing the scale).
2. Depth from a native app (ARKit / ARCore, LiDAR) feeding the same geometry code.
3. Browser-side port of the geometry so the app runs as a static site (GitHub Pages) with real HTTPS.
4. Validation report: bias, repeatability and k=2 coverage from a set of photos of known objects.
5. Optional geolocation: WGS84 to ECEF to ENU for georeferenced records.


## Virtual square (fit from parallel lines)

When no printed marker is available, the surface orientation can be recovered from the scene itself.
Tap two pairs of lines that are parallel on the surface and perpendicular to each other (A1, A2 and B1, B2:
rows and sides of a keyboard, floor tiles). Their vanishing points give the focal length
(f^2 = -(v1-c).(v2-c)) and the plane orientation, so pixels map to the plane up to ONE scale.
The scale is a free choice: a square of side X (mm, chosen by the user) is declared to look a given size in the photo.
Everything measured is then X-scaled and provisional until X is confirmed with a real dimension (Confirm size / cross-check).
Synthetic accuracy with 300 px lines in a 1000 px photo: median length error 0.6 % (1 px tap noise), 1.2 % (2 px), 2.4 % (4 px);
the app assumes 4 % fit uncertainty. Not valid for curved surfaces; lens distortion is corrected only if lens.json is loaded.
