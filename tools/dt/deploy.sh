#!/bin/sh
# DT round: copy reviewed files from the staging copy into the live repo, atomically, keeping a one-time backup of the
# live version. Usage: tools/dt/deploy.sh path/relative/to/repo [more paths]
set -e
MAIN=/data1/lyq/code/mesh/EventHands1
STAGE=/data1/lyq/code/mesh/EventHands1_dt
for f in "$@"; do
  [ -f "$STAGE/$f" ] || { echo "missing in STAGE: $f" >&2; exit 1; }
  bk="$MAIN/outputs/dt/ref/main_orig/$f"
  if [ -f "$MAIN/$f" ] && [ ! -f "$bk" ]; then
    mkdir -p "$(dirname "$bk")"; cp -p "$MAIN/$f" "$bk"
  fi
  mkdir -p "$(dirname "$MAIN/$f")"
  cp "$STAGE/$f" "$MAIN/$f.dt_tmp" && mv "$MAIN/$f.dt_tmp" "$MAIN/$f"
  echo "deployed $f ($(md5sum "$MAIN/$f" | cut -c1-12))"
done
