# ffmpeg -y -r 15 -f image2 -s 1920x1080 -i results/synthia/combo/overlay/synthiaOmni_F_%06d-combo-Rgb-pred_overlay.png -vcodec libx264 -crf 25  -pix_fmt yuv420p results/synthia/synthia-combo-test.mp4

dataset=kaistped
arg=$1
checkpoints=(
  "fusionfusion-custom16rll-multi-2021-05-03 22-03-freiburgthermal-c3-sord-1,2,3-a1-logl2-lw-rgb,ir-epoch=41-val_loss=0.0016"
  "fusionfusion-custom16rll-multi-2021-05-13 09-17-freiburgthermal-c3-sord-1,2,3-a1-logl2-lw-rgb,ir-epoch=27-val_loss=0.0016"
)

for checkpoint in "${checkpoints[@]}"
do
  for set in 06 07 08
  do
    for v in V000 V001 V002 V003 V004
  do
    xp="demo-s${set}-${v}"
    ffmpeg -y -r 40 -f image2 -pattern_type glob -i "results/$dataset/${xp}/overlayRgbIr-_${checkpoint}_affordances/${dataset}*-RgbIr-pred_overlay.png" -c:v libx264 -qp 0 "results/$dataset/$xp-${checkpoint}.mp4"
  done
done
done





# dataset=freiburgthermal
# xp=combo
# ffmpeg -r 5 -f image2 -pattern_type glob -i "results/$dataset/${xp}/overlayRgbIr-_${checkpoint}_affordances/${dataset}*-RgbIr-pred_overlay.png" -c:v libx264 -qp 0 "results/$dataset/${checkpoint}.mp4"
