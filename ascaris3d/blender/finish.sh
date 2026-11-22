#!/bin/bash
# Assemblaggio: anatomia 10 fps -> 30 fps interpolati, composizione, crediti, MP4.
# Uso: finish.sh WORKDIR OUT.mp4
set -e
W=$1; OUT=$2; HERE=$(cd "$(dirname "$0")" && pwd)
cd "$W"
mkdir -p seq interp comp
n=0; for f in $(ls R_anat/b_*.png | sort); do ln -sf "../$f" "seq/A_$(printf %05d $n).png"; n=$((n+1)); done
ffmpeg -y -loglevel error -framerate 10 -i seq/A_%05d.png \
  -vf "minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1" -frames:v 1980 -start_number 0 interp/I_%05d.png
python3 "$HERE/compose.py" --state state.json --anat interp --anat-name 'I_{:05d}.png' --over R_over --out comp --frames 0:1980:1 --outro
ffmpeg -y -loglevel error -framerate 30 -i comp/c_%05d.png -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -movflags +faststart "$OUT"
echo "ok $OUT"
