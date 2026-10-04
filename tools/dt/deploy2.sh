#!/bin/sh
# DT2 round: copy reviewed files from a staging copy into the live repo, atomically, keeping a one-time backup of the
# live version. Usage: tools/dt/deploy2.sh <stage_dir> path/relative/to/repo [more paths]
set -e
MAIN=/data1/lyq/code/mesh/EventHands1
STAGE=$1; shift
for f in "$@"; do
  [ -f "$STAGE/$f" ] || { echo "missing in STAGE: $f" >&2; exit 1; }
  bk="$MAIN/outputs/dt2/ref/main_orig/$f"
  if [ -f "$MAIN/$f" ] && [ ! -f "$bk" ]; then
    mkdir -p "$(dirname "$bk")"; cp -p "$MAIN/$f" "$bk"
  fi
  mkdir -p "$(dirname "$MAIN/$f")"
  cp "$STAGE/$f" "$MAIN/$f.dt2_tmp" && mv "$MAIN/$f.dt2_tmp" "$MAIN/$f"
  echo "deployed $f from $STAGE ($(md5sum "$MAIN/$f" | cut -c1-12))" | tee -a "$MAIN/outputs/dt2/deploy.log"
done
