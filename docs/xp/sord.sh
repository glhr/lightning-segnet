## Freiburg

run() {
  mkdir -p results/$dataset/$xp/txt
  for checkpoint in "${checkpoints[@]}"
  do
    txtoutput="results/${dataset}/${xp}/txt/${checkpoint}.txt"
    echo "$checkpoint"
    if [ ! -f "$txtoutput" ]; then
      echo "--> running eval"
      python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset $dataset --test_checkpoint "lightning_logs/${checkpoint}.ckpt" --save --save_xp sord > "$txtoutput" 2>&1
    else
      echo "--> summary"
      tail -14 "$txtoutput"
    fi
  done
}

xp=sord

dataset=freiburg
checkpoints=(
"2021-04-08 13-31-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474"
"2021-04-08 14-55-freiburg-c6-sord-1,2,3-a1-l1-rgb-epoch=66-val_loss=0.0195"
"2021-04-08 13-59-freiburg-c6-sord-1,2,3-a1-l2-rgb-epoch=66-val_loss=0.0278"
"2021-04-08 14-28-freiburg-c6-sord-1,2,3-a1-logl2-rgb-epoch=74-val_loss=0.0061"
"2021-04-08 16-53-freiburg-c6-sord-1,2,3-a2-l1-rgb-epoch=66-val_loss=0.0597"
"2021-04-09 08-26-freiburg-c6-sord-1,2,3-a2-l2-rgb-epoch=43-val_loss=0.1179"
"2021-04-09 09-04-freiburg-c6-sord-1,2,3-a2-logl2-rgb-epoch=74-val_loss=0.0498"
)

run

dataset=cityscapes
checkpoints=(
"2021-04-09 03-40-cityscapes-c30-kl-rgb-epoch=18-val_loss=0.0918"
"2021-04-09 05-09-cityscapes-c30-sord-1,2,3-a1-l1-rgb-epoch=23-val_loss=0.0134"
"2021-04-08 19-38-cityscapes-c30-sord-1,2,3-a1-l2-rgb-epoch=21-val_loss=0.0214"
"2021-04-08 21-07-cityscapes-c30-sord-1,2,3-a1-logl2-rgb-epoch=23-val_loss=0.0034"
"2021-04-08 23-08-cityscapes-c30-sord-1,2,3-a2-l1-rgb-epoch=23-val_loss=0.0391"
"2021-04-09 00-39-cityscapes-c30-sord-1,2,3-a2-l2-rgb-epoch=23-val_loss=0.0742"
"2021-04-09 02-09-cityscapes-c30-sord-1,2,3-a2-logl2-rgb-epoch=23-val_loss=0.0236"
)

run
