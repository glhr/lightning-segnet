dataset=wilddash

  txtoutput="results/${dataset}/txt/demo-${seq}-${checkpoints[$i]}.txt"
  echo "demo"
  xp=demo
  isInFile=$(cat "$txtoutput" | grep -c "DATALOADER:0 TEST RESULTS")
  if [ ! -f "$txtoutput" ] || [ $isInFile -eq 0 ] ; then
    echo "-> saving predictions"
    ffmpeg -y -pattern_type glob -framerate 1.5 -i  "results/$dataset/selected/${dataset}*-pred_overlay.png" -c:v copy results/${dataset}/${xp}.mkv
  fi
