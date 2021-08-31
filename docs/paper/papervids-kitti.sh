dataset=kittiraw
seqs=(
2011_09_28_drive_0039_sync

)




for seq in "${seqs[@]}"
do
    txtoutput="results/${dataset}/txt/demo-${seq}-${checkpoints[$i]}.txt"
    echo "demo-${seq}-${checkpoints[$i]}"
    xp=demo-$seq
    isInFile=$(cat "$txtoutput" | grep -c "DATALOADER:0 TEST RESULTS")
    if [ ! -f "$txtoutput" ] || [ $isInFile -eq 0 ] ; then
      echo "-> saving predictions"
      ffmpeg -y -pattern_type glob -r 20 -i  "results/$dataset/*/overlayRgb-_modelcomp_2021-03-29 09-16-cityscapes-c30-kl-rgb-epoch=190-val_loss=0.4310_affordances_2021-08-26 07-09-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=109-val_loss=0.0212_affordances/${dataset}*-pred_overlay.png" -vf "select=not(mod(n\,20))" -c:v libx264 -qp 0 results/${dataset}/${xp}.mp4
    fi

done
