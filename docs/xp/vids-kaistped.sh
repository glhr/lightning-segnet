# ffmpeg -y -r 15 -f image2 -s 1920x1080 -i results/synthia/combo/overlay/synthiaOmni_F_%06d-combo-Rgb-pred_overlay.png -vcodec libx264 -crf 25  -pix_fmt yuv420p results/synthia/synthia-combo-test.mp4

dataset=freiburgthermal
arg=$1
xp=thermo
checkpoints=(
"2021-05-06 13-48-freiburgthermal-c13-sord-1,2,3-a1-logl2-rgb-epoch=23-val_loss=0.0037"
)

txt=(
  "Trained on Freiburg Thermal"
)

overlay=(
  "GtRgbIr"
)



# for checkpoint in "${checkpoints[@]}"
for i in ${!checkpoints[@]}
do
    echo ${checkpoints[$i]}
    # ffmpeg -r 3 -f image2 -pattern_type glob -i "results/$dataset/${xp}/overlay${overlay[$i]}-_${checkpoints[$i]}_affordances/${dataset}*-pred_overlay.png" -vf "drawtext=text='${txt[$i]}':x=550:y=20:fontsize=24:fontcolor=white" -c:v libx264 -qp 0 "results/kaistped/ref-${checkpoints[$i]}.mp4"
    ffmpeg -stream_loop -1 -y -pattern_type glob -r 40 -i  "results/$dataset/${xp}/overlay${overlay[$i]}-_${checkpoints[$i]}_affordances/${dataset}*-pred_overlay.png" -i results/kaistped/demo-lw-bottom.mp4 -filter_complex vstack=inputs=2 -c:v libx264 -qp 0 results/kaistped/kaistped-lw.mp4
    ffmpeg -y -i results/kaistped/kaistped-lw.mp4 -vf "drawtext=text='Trained on Freiburg Thermal':x=550:y=20:fontsize=24:fontcolor=white" -c:v libx264 -qp 0 results/kaistped/kaistped-lw-2.mp4
done

dataset=kaistped
arg=$1
checkpoints=(
  "fusionfusion-custom16rll-multi-2021-05-13 09-17-freiburgthermal-c3-sord-1,2,3-a1-logl2-lw-rgb,ir-epoch=27-val_loss=0.0016"
)

txt=(
  "Deep fusion (V+T) - SORD + LW"
)

overlay=(
  "RgbIr"
)

for checkpoint in "${checkpoints[@]}"
do
  for set in 06 07 08
  do
    for v in V000 V001 V002 V003 V004
  do
    xp="demo-s${set}-${v}"
    ffmpeg -r 40 -f image2 -pattern_type glob -i "results/$dataset/${xp}/overlay${overlay[$i]}-_${checkpoint}_affordances/${dataset}*-RgbIr-pred_overlay.png" -vf "drawtext=text='${txt[$i]}':x=1000:y=20:fontsize=24:fontcolor=white" -c:v libx264 -qp 0 "results/$dataset/$xp-${checkpoint}.mp4"
  done
done
done


ffmpeg -y -f concat -safe 0 -i results/$dataset/demo-lw.txt -c copy results/$dataset/demo-lw-bottom.mp4



# dataset=freiburgthermal
# xp=combo
# ffmpeg -r 5 -f image2 -pattern_type glob -i "results/$dataset/${xp}/overlayRgbIr-_${checkpoint}_affordances/${dataset}*-RgbIr-pred_overlay.png" -c:v libx264 -qp 0 "results/$dataset/${checkpoint}.mp4"
