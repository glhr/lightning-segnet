xp=lw

arg=$1

run() {
  mkdir -p results/$dataset/$xp/txt
  for checkpoint in "${checkpoints[@]}"
  do
    txtoutput="results/${dataset}/${xp}/txt/${checkpoint}.txt"
    echo "$checkpoint"
    isInFile=$(cat "$txtoutput" | grep -c "DATALOADER:0 TEST RESULTS")
    if [ ! -f "$txtoutput" ] || [ $isInFile -eq 0 ] ; then
      echo "--> running eval"
      python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset $dataset --test_checkpoint "lightning_logs/${checkpoint}.ckpt" --save --save_xp $xp --loss_weight > "$txtoutput" 2>&1
    fi
    echo "--> summary"
    tail -14 "$txtoutput"
    if [[ $arg == "overlay" ]]; then
      echo "--> generating overlay"
      python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${checkpoints[0]}_affordances" --model2 "${checkpoints[1]}_affordances" --gt
    fi
  done
}

dataset=freiburg
checkpoints=(
"2021-04-08 13-31-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474"
"2021-04-07 11-31-freiburg-c6-kl-0,1,2-rgb-epoch=43-val_loss=0.0771"
)

run

dataset=cityscapes
checkpoints=(
"2021-04-09 03-40-cityscapes-c30-kl-rgb-epoch=18-val_loss=0.0918"
"2021-04-09 10-00-cityscapes-c30-kl-rgb-epoch=6-val_loss=0.0283"
)

run
