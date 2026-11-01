# Spatial Measure

Measure length, area, volume and estimated mass from a smartphone photo, using a printed ArUco marker (4x4_50, 50 mm) as the scale reference. Reports an uncertainty (k=2).

**Estimator, not a certified instrument. Not legal for trade.**

## Run (Debian 13)
    ./setup.sh
    ./venv/bin/python app.py
Open the printed https://<ip>:5000 on a phone on the same Wi-Fi and accept the certificate warning.

## Test
    ./venv/bin/python -m pytest -q tests

## Limits
Only points on the marker's plane are measured correctly. Weight = volume x assumed density, shown as a range. See docs/BLUEPRINT.md.
