dataset=freiburgraw
arg=$1
checkpoints=(
"2021-04-08 14-28-freiburg-c6-sord-1,2,3-a1-logl2-rgb-epoch=74-val_loss=0.0061"
"2021-04-08 13-31-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474"
"2021-04-07 11-31-freiburg-c6-kl-0,1,2-rgb-epoch=43-val_loss=0.0771"
)

txt=(
  "Baseline"
  "SORD"
  "Loss weighting"
)

for checkpoint in "${checkpoints[@]}"
do
  for set in demo-2016-03-01-11-50-45
  do
    xp="${set}"
    ffmpeg -y -r 80 -f image2 -pattern_type glob -i "results/$dataset/*/overlayRgb-_${checkpoint}_affordances/${dataset}*-pred_overlay.png" -vf "drawtext=text='${txt[$i]}':x=1000:y=20:fontsize=24:fontcolor=white" -c:v libx264 -qp 0 "results/$dataset/$xp-${checkpoint}.mp4"
  done
done
