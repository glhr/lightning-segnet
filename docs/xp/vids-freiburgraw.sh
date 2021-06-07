dataset=freiburgraw
arg=$1
checkpoints=(
"2021-04-08 13-31-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474"
"2021-04-08 14-28-freiburg-c6-sord-1,2,3-a1-logl2-rgb-epoch=74-val_loss=0.0061"
"2021-04-07 11-31-freiburg-c6-kl-0,1,2-rgb-epoch=43-val_loss=0.0771"
)

txt=(
  "Baseline"
  "SORD"
  "Loss weighting"
)

for set in demo-2016-02-22-12-42-53 demo-2016-03-01-12-19-07 demo-2016-03-01-12-05-22 demo-2016-02-26-15-05-05 demo-2016-02-26-14-47-29 demo-2016-02-26-15-20-48 demo-2016-02-26-15-05-05 demo-2016-03-01-11-44-50 demo-2016-02-22-12-32-18 demo-2016-03-01-11-50-45
do
  xp="${set}"
  for i in ${!checkpoints[@]}
  do
    echo ${txt[$i]}
    ffmpeg -r 90 -f image2 -pattern_type glob -i "results/$dataset/$xp/overlayRgb-_${checkpoints[$i]}_affordances/${dataset}*-pred_overlay.png" -c:v libx264 -qp 0 -vf "drawtext=text='${txt[$i]}':x=700:y=20:fontsize=24:fontcolor=white" "results/$dataset/$xp-${checkpoints[$i]}.mp4"
  done
  ffmpeg -i "results/$dataset/$xp-${checkpoints[0]}.mp4" -i "results/$dataset/$xp-${checkpoints[1]}.mp4" -i "results/$dataset/$xp-${checkpoints[2]}.mp4" -filter_complex vstack=inputs=3 -c:v libx264 -qp 0 "results/$dataset/$dataset-$xp.mp4"
done

# ffmpeg -y -i results/cityscapesraw/cityscapesraw-base.mp4 -i results/cityscapesraw/cityscapesraw-sord.mp4 -i results/cityscapesraw/cityscapesraw-lw.mp4 -filter_complex vstack=inputs=3 -c:v libx264 -qp 0 results/cityscapesraw/cityscapesraw.mp4
