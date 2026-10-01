#!/usr/bin/env bash
# Optional upstream source; never installs its historical requirements.
set -euo pipefail
extension_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${extension_root}/third_party/bitrap"
revision="${1:-main}"
if [[ -e "$destination" ]]; then
  echo "Already exists: $destination. Set bitrap.upstream_root to that checkout or choose a new checkout." >&2
  exit 1
fi
git clone --quiet https://github.com/umautobots/bidirection-trajectory-predicter "$destination"
git -C "$destination" checkout --quiet "$revision"
echo "Optional BiTraP source ready at $destination"
git -C "$destination" rev-parse HEAD
