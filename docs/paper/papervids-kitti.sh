dataset=kittiraw
seqs=(
selected

)




for seq in "${seqs[@]}"
do
    txtoutput="results/${dataset}/txt/demo-${seq}-${checkpoints[$i]}.txt"
    echo "demo-${seq}-${checkpoints[$i]}"
    xp=demo-$seq
    isInFile=$(cat "$txtoutput" | grep -c "DATALOADER:0 TEST RESULTS")
    if [ ! -f "$txtoutput" ] || [ $isInFile -eq 0 ] ; then
      echo "-> saving predictions"
      ffmpeg -y -pattern_type glob -r 20 -i  "results/$dataset/selected/${dataset}*-pred_overlay.png" -vf "select=not(mod(n\,2))" -c:v libx264rgb -crf 0 results/${dataset}/${xp}.mkv
    fi

done
