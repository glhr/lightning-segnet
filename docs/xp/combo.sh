
xp=combo

for dataset in freiburg cityscapes thermalvoc synthia kitti multispectralseg freiburgthermal lostfound
do
  checkpoint1="2021-05-10 19-34-combo-c30-sord-1,2,3-a1-logl2-rgb-epoch=49-val_loss=0.0053"
  txtoutput="results/${dataset}/${xp}/txt/${checkpoint1}.txt"

  if [ ! -f "$txtoutput" ]; then
    mkdir -p results/$dataset/$xp/txt
    echo "Running evaluation for ${dataset}"
    python3 lightning.py --dataset $dataset --bs 1 --save --save_xp $xp --save --dataset_combo_ntrain 180 --test_checkpoint "lightning_logs/${checkpoint1}.ckpt" --loss_weight > $txtoutput 2>&1

    echo "--> generating overlays for ${dataset}"
    python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${checkpoint1}_affordances" --rgb --gt
    python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${checkpoint1}_affordances" --rgb
  fi

  checkpoint2="2021-05-10 22-40-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=52-val_loss=0.0023"
  txtoutput="results/${dataset}/${xp}/txt/${checkpoint2}.txt"
  if [ ! -f "$txtoutput" ]; then
    mkdir -p results/$dataset/$xp/txt
    echo "Running evaluation for ${dataset}"
    python3 lightning.py --dataset $dataset --bs 1 --save --save_xp $xp --save --dataset_combo_ntrain 180 --test_checkpoint "lightning_logs/${checkpoint2}.ckpt" --loss_weight > $txtoutput 2>&1

    echo "--> generating overlays for ${dataset}"
    python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${checkpoint1}_affordances" --model2 "${checkpoint2}_affordances" --rgb --gt
    python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${checkpoint1}_affordances" --model2 "${checkpoint2}_affordances" --rgb
  fi

done
