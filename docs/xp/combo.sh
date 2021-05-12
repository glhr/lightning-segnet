xp=combo

arg=$1

for dataset in freiburg cityscapes thermalvoc synthia kitti multispectralseg freiburgthermal lostfound
do
  checkpoint1="2021-05-12 11-24-combo-c30-sord-1,2,3-a1-logl2-rgb-epoch=42-val_loss=0.0060"
  txtoutput="results/${dataset}/${xp}/txt/${checkpoint1}.txt"

  isInFile=$(cat "$txtoutput" | grep -c "DATALOADER:0 TEST RESULTS")
  if [ ! -f "$txtoutput" ] || [ $isInFile -eq 0 ] ; then
    mkdir -p results/$dataset/$xp/txt
    echo "Running evaluation for ${dataset} | ${checkpoint1}"
    if [ -f "lightning_logs/${checkpoint1}.ckpt" ] ; then
      python3 lightning.py --dataset $dataset --bs 1 --save --save_xp $xp --save --dataset_combo_ntrain 180 --test_checkpoint "lightning_logs/${checkpoint1}.ckpt" --loss_weight > "$txtoutput" 2>&1
    else
      echo "checkpoint ${checkpoint1} not found :( bye"
    fi
  fi
  if [[ $arg == "overlay" ]]; then
    echo "--> generating overlays for ${dataset}"
    python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${checkpoint1}_affordances" --rgb --gt
    python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${checkpoint1}_affordances" --rgb
  fi
  echo "--> summary for ${dataset} | ${checkpoint1}"
  tail -14 "$txtoutput"

  checkpoint2="2021-05-12 14-34-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=49-val_loss=0.0023"
  txtoutput="results/${dataset}/${xp}/txt/${checkpoint2}.txt"

  isInFile=$(cat "$txtoutput" | grep -c "DATALOADER:0 TEST RESULTS")
  if [ ! -f "$txtoutput" ] || [ $isInFile -eq 0 ] ; then
    mkdir -p results/$dataset/$xp/txt
    echo "Running evaluation for ${dataset} | ${checkpoint2}"
    if [ -f "lightning_logs/${checkpoint2}.ckpt" ] ; then
      python3 lightning.py --dataset $dataset --bs 1 --save --save_xp $xp --save --dataset_combo_ntrain 180 --test_checkpoint "lightning_logs/${checkpoint2}.ckpt" --loss_weight > "$txtoutput" 2>&1
    else
      echo "checkpoint ${checkpoint2} not found :( bye"
    fi
  fi
  if [[ $arg == "overlay" ]]; then
    echo "--> generating overlays for ${dataset}"
    python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${checkpoint1}_affordances" --model2 "${checkpoint2}_affordances" --rgb --gt
    python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${checkpoint1}_affordances" --model2 "${checkpoint2}_affordances" --rgb
  fi
  echo "--> summary for ${dataset} | ${checkpoint2}"
  tail -14 "$txtoutput"

done
