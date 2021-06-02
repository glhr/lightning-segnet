dataset=freiburgthermal
arg=$1
xp=thermo
checkpoints=(
"2021-05-06 13-48-freiburgthermal-c13-sord-1,2,3-a1-logl2-rgb-epoch=23-val_loss=0.0037"
"fusionfusion-custom16rll-multi-2021-05-07 11-08-freiburgthermal-c3-sord-1,2,3-a1-logl2-rgb,ir-epoch=25-val_loss=0.0036"
"fusionfusion-custom16rll-multi-2021-05-13 09-17-freiburgthermal-c3-sord-1,2,3-a1-logl2-lw-rgb,ir-epoch=27-val_loss=0.0016"
)

txt=(
  "SegNet (V) - SORD"
  "Deep fusion (V+T) - SORD"
  "Deep fusion (V+T) - SORD + LW"
)

overlay=(
  "GtRgb"
  "RgbIr"
  "RgbIr"
)

# for checkpoint in "${checkpoints[@]}"
for i in ${!checkpoints[@]}
do
    echo ${checkpoints[$i]}
    ffmpeg -y -r 5 -f image2 -pattern_type glob -i "results/$dataset/${xp}/overlay${overlay[$i]}-_${checkpoints[$i]}_affordances/${dataset}*-pred_overlay.png" -vf "drawtext=text='${txt[$i]}':x=1000:y=20:fontsize=24:fontcolor=white" -c:v libx264 -qp 0 "results/$dataset/$xp-${checkpoints[$i]}.mp4"
done

ffmpeg -y -i "results/${dataset}/thermo-${checkpoints[0]}.mp4" -i "results/${dataset}/thermo-${checkpoints[1]}.mp4" -i "results/${dataset}/thermo-${checkpoints[2]}.mp4" -filter_complex vstack=inputs=3 -c:v libx264 -qp 0 results/${dataset}/vt-fusion.mp4

ffmpeg -i results/${dataset}/vt-fusion.mp4 -vf fps=1/25 results/${dataset}/out%d.png
