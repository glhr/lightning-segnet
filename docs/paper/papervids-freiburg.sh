dataset=freiburgraw
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
      ffmpeg -y -pattern_type glob -framerate 20 -i  "results/$dataset/selected/${dataset}*-pred_overlay.png" -c:v copy results/${dataset}/${xp}.mkv
    fi

done
