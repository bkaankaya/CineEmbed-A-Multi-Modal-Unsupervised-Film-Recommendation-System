#!/usr/bin/env bash
# scripts/demo-smoke.sh — hit every endpoint and validate JSON shape
set -euo pipefail

BASE="${1:-http://localhost:8000}"
PASS="\033[32m✓\033[0m"
FAIL="\033[31m✗\033[0m"
FAILED=0

check() {
  local name="$1"
  local url="$2"
  local jq_filter="$3"
  local result
  result=$(curl -sf "$url" 2>/dev/null | jq -e "$jq_filter" 2>/dev/null || echo "")
  if [ -n "$result" ]; then
    echo -e "  $PASS $name"
  else
    echo -e "  $FAIL $name :: $url"
    FAILED=$((FAILED + 1))
  fi
}

echo "=== CineEmbed demo smoke @ $BASE ==="
check "health"             "$BASE/api/health"                                   '.status == "ok"'
check "backbones"          "$BASE/api/backbones"                                'length == 3'
check "search inception"   "$BASE/api/films/search?q=inception&backbone=ae_z32" 'length > 0'
check "film 27205"         "$BASE/api/films/27205?backbone=ae_z32"              '.id == 27205'
check "similar 27205"      "$BASE/api/films/27205/similar?backbone=ae_z32"      'length == 10 and (.[0].cosine | type == "number")'
check "cosine-dist 27205"  "$BASE/api/films/27205/cosine-dist?backbone=ae_z32"  '.bins | length > 0'
check "clusters"           "$BASE/api/clusters?backbone=ae_z32"                 'length == 21'
check "cluster 0"          "$BASE/api/clusters/0?backbone=ae_z32"               '.id == 0'
check "gallery"            "$BASE/api/gallery"                                  '.matrix | length > 0'

echo "==="
if [ $FAILED -eq 0 ]; then
  echo -e "$PASS all checks passed"
  exit 0
else
  echo -e "$FAIL $FAILED checks failed"
  exit 1
fi
