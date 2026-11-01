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

## 8. Roadmap

1. Lens calibration and undistort step.
2. Auto-detect object outline (segmentation, e.g. OpenCV GrabCut or a small model) instead of tapping.
3. Two-view height: marker on floor and on a side face, or marker plus vertical reference.
4. Native app (ARKit/ARCore, LiDAR/depth) with the same core API and the marker as precision override.
5. Optional geolocation: WGS84 to ECEF to ENU for georeferenced records.
6. Validation report generator (bias, repeatability, k=2 coverage).
