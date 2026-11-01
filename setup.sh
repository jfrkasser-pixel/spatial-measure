#!/usr/bin/env bash
# Debian 13 (trixie) setup: venv, deps, local HTTPS certificate.
set -euo pipefail
cd "$(dirname "$0")"

sudo apt update
sudo apt install -y python3 python3-venv python3-pip openssl

python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

mkdir -p certs
IP="$(ip route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}')"
echo "LAN IP: $IP"

if command -v mkcert >/dev/null 2>&1; then
  # Trusted by devices where you installed the mkcert root CA.
  mkcert -cert-file certs/cert.pem -key-file certs/key.pem localhost 127.0.0.1 "$IP"
else
  # Self-signed fallback: phone browser shows a warning; accept it once and the camera works.
  openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
    -keyout certs/key.pem -out certs/cert.pem \
    -subj "/CN=spatialmeasure" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:$IP"
fi

echo "Done. Run:  ./venv/bin/python app.py"
