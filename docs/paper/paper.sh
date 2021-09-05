datasets=(
wilddash
)
xp=paper
test=test

run() {
  mkdir -p results/$dataset/$xp/txt
  for checkpoint in "${checkpoints[@]}"
  do
        for dataset in "${datasets[@]}"
        do
        txtoutput="results/${dataset}/${xp}/txt/${checkpoint}.txt"
        echo "$checkpoint"
        isInFile=$(cat "$txtoutput" | grep -c "DATALOADER:0 TEST RESULTS")
        if [ ! -f "$txtoutput" ] || [ $isInFile -eq 0 ] ; then
                echo "--> running eval"
                python3 lightning-combo.py --num_classes $cls --bs 1 --mode $mode --dataset $dataset --test_checkpoint "lightning_logs/${checkpoint}" --orig_dataset cityscapes --workers 10 --gpu 1 --init_channels $channels --test_set $test --bs 1 > "$txtoutput" 2>&1
        fi
        echo "--> summary ${dataset}"
        tail -14 "$txtoutput"
        done
  done
}

channels=3
mode=convert
cls=30

checkpoints=(
"2021-08-13 15-27-cityscapes-c30-kl-rgb-epoch=197-val_loss=0.4043.ckpt" #color
)


mode=affordances
cls=3
checkpoints=(
"2021-08-14 18-55-combo-c30-kl-rgb-epoch=68-val_loss=0.1411.ckpt" #color
)

channels=1

mode=convert
cls=30
checkpoints=(
"2021-03-29 09-16-cityscapes-c30-kl-rgb-epoch=190-val_loss=0.4310.ckpt" # gray
)


mode=affordances
cls=3
checkpoints=(
# "2021-08-15 11-58-combo-c30-kl-rgb-epoch=82-val_loss=0.1474.ckpt" #gray
# "2021-04-09 03-40-cityscapes-c30-kl-rgb-epoch=18-val_loss=0.0918.ckpt"
#"2021-04-08 21-07-cityscapes-c30-sord-1,2,3-a1-logl2-rgb-epoch=23-val_loss=0.0034.ckpt"
"2021-06-11 13-18-cityscapes-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=32-val_loss=0.0044.ckpt"

# "2021-08-16 09-02-combo-c30-sord-1,2,3-a1-logl2-rgb-epoch=78-val_loss=0.0062.ckpt"
# "2021-08-17 09-47-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=85-val_loss=0.0216.ckpt"
# "2021-08-18 14-05-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=84-val_loss=0.0026.ckpt"
# "2021-08-17 09-47-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=88-val_loss=0.0211.ckpt"
# "2021-08-22 10-21-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=92-val_loss=0.0215.ckpt"
# "2021-08-26 07-09-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=109-val_loss=0.0212.ckpt"
# "2021-08-26 07-09-cityscapes-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=108-val_loss=0.0051.ckpt"
)

run
