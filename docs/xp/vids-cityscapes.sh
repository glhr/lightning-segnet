dataset=cityscapesraw
arg=$1
checkpoints=(
"2021-04-08 21-07-cityscapes-c30-sord-1,2,3-a1-logl2-rgb-epoch=23-val_loss=0.0034"
"2021-04-09 03-40-cityscapes-c30-kl-rgb-epoch=18-val_loss=0.0918"
"2021-04-09 10-00-cityscapes-c30-kl-rgb-epoch=6-val_loss=0.0283"
)

for checkpoint in "${checkpoints[@]}"
do
  for set in stuttgart_00 stuttgart_01 stuttgart_02
  do
    xp="demo-${set}"
    ffmpeg -y -r 30 -f image2 -pattern_type glob -i "results/$dataset/${xp}/overlayRgb-_${checkpoint}_affordances/${dataset}*-pred_overlay.png" -c:v libx264 -qp 0 "results/$dataset/$xp-${checkpoint}.mp4"
  done
done

ffmpeg -y -f concat -safe 0 -i results/cityscapesraw/cityscapesraw-base.txt -vf "drawtext=text='Baseline':x=700:y=20:fontsize=24:fontcolor=white" -c:v libx264 -qp 0 results/cityscapesraw/cityscapesraw-base.mp4
ffmpeg -y -f concat -safe 0 -i results/cityscapesraw/cityscapesraw-sord.txt -vf "drawtext=text='SORD':x=700:y=20:fontsize=24:fontcolor=white" -c:v libx264 -qp 0 results/cityscapesraw/cityscapesraw-sord.mp4
ffmpeg -y -f concat -safe 0 -i results/cityscapesraw/cityscapesraw-lw.txt -vf "drawtext=text='Loss weighting':x=700:y=20:fontsize=24:fontcolor=white" -c:v libx264 -qp 0 results/cityscapesraw/cityscapesraw-lw.mp4

ffmpeg -y -i results/cityscapesraw/cityscapesraw-base.mp4 -i results/cityscapesraw/cityscapesraw-sord.mp4 -i results/cityscapesraw/cityscapesraw-lw.mp4 -filter_complex vstack=inputs=3 -c:v libx264 -qp 0 results/cityscapesraw/cityscapesraw.mp4
