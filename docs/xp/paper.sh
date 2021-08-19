channels=2
dataset=combo
xp=paper

run() {
  mkdir -p results/$dataset/$xp/txt
  for checkpoint in "${checkpoints[@]}"
  do
    txtoutput="results/${dataset}/${xp}/txt/${checkpoint}.txt"
    echo "$checkpoint"
    isInFile=$(cat "$txtoutput" | grep -c "DATALOADER:0 TEST RESULTS")
    if [ ! -f "$txtoutput" ] || [ $isInFile -eq 0 ] ; then
      echo "--> running eval"
      python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset $dataset --test_checkpoint "lightning_logs/${checkpoint}" --save --save_xp $xp --workers 10 --gpu 1 --save_xp $xp --init_channels $channels --test_set test --bs 1 > "$txtoutput" 2>&1
    fi
    echo "--> summary"
    tail -14 "$txtoutput"
  done
}

checkpoints=(
"2021-08-15 11-58-combo-c30-kl-rgb-epoch=82-val_loss=0.1474.ckpt"
"2021-08-16 09-02-combo-c30-sord-1,2,3-a1-logl2-rgb-epoch=78-val_loss=0.0062.ckpt"
"2021-08-17 09-47-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=85-val_loss=0.0216.ckpt"
"2021-08-18 14-05-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=84-val_loss=0.0026.ckpt"
"2021-08-17 09-47-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=88-val_loss=0.0211.ckpt"
)

run
