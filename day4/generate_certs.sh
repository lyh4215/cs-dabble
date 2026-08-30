#!/usr/bin/env bash
set -euo pipefail

CERT_DIR="certs"

mkdir -p "$CERT_DIR"

echo "[1/5] Generating CA private key..."
openssl genrsa \
  -out "$CERT_DIR/ca.key" \
  2048


echo "[2/5] Generating self-signed CA certificate..."
openssl req \
  -x509 \
  -new \
  -key "$CERT_DIR/ca.key" \
  -sha256 \
  -days 365 \
  -out "$CERT_DIR/ca.crt" \
  -subj "/CN=Day4 Toy CA"


echo "[3/5] Generating server private key..."
openssl genrsa \
  -out "$CERT_DIR/server.key" \
  2048


echo "[4/5] Generating server CSR..."
openssl req \
  -new \
  -key "$CERT_DIR/server.key" \
  -out "$CERT_DIR/server.csr" \
  -subj "/CN=localhost"


echo "[5/5] Signing server certificate with Toy CA..."

cat > "$CERT_DIR/server.ext" <<'EOF'
subjectAltName=DNS:localhost,IP:127.0.0.1
EOF

openssl x509 \
  -req \
  -in "$CERT_DIR/server.csr" \
  -CA "$CERT_DIR/ca.crt" \
  -CAkey "$CERT_DIR/ca.key" \
  -CAcreateserial \
  -out "$CERT_DIR/server.crt" \
  -days 365 \
  -sha256 \
  -extfile "$CERT_DIR/server.ext"


echo
echo "Certificates generated successfully:"
echo
ls -l "$CERT_DIR"

echo
echo "Verify server certificate:"
openssl verify \
  -CAfile "$CERT_DIR/ca.crt" \
  "$CERT_DIR/server.crt"