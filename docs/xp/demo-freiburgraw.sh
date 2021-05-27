checkpoints=(
# "2021-04-08 14-28-freiburg-c6-sord-1,2,3-a1-logl2-rgb-epoch=74-val_loss=0.0061"
"2021-04-08 13-31-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474"
"2021-04-07 11-31-freiburg-c6-kl-0,1,2-rgb-epoch=43-val_loss=0.0771"
)

dataset=freiburgraw
arg=$1
xp=demo

# seq = 2016-02-22-12-32-18  2016-02-22-12-47-00  2016-02-26-14-51-16  2016-02-26-15-26-10  2016-03-01-11-54-41  2016-03-01-12-11-45 2016-02-22-12-37-11  2016-02-26-14-39-14
# 2016-02-26-15-05-05  2016-03-01-11-44-50  2016-03-01-11-57-14  2016-03-01-12-19-07 2016-02-22-12-42-53  2016-02-26-14-47-29  2016-02-26-15-20-48  2016-03-01-11-50-45  2016-03-01-12-05-22  2016-03-01-12-40-50

mkdir -p results/$dataset/$xp/txt
for checkpoint in "${checkpoints[@]}"
do
  for seq in 2016-02-22-12-32-18  2016-02-22-12-47-00  2016-02-26-14-51-16  2016-02-26-15-26-10  2016-03-01-11-54-41  2016-03-01-12-11-45 2016-02-22-12-37-11  2016-02-26-14-39-14
  do
      txtoutput="results/${dataset}/${xp}/txt/demo-${seq}-${checkpoint}.txt"
      echo "demo-${seq}-${checkpoint}"
      isInFile=$(cat "$txtoutput" | grep -c "DATALOADER:0 TEST RESULTS")
      if [ ! -f "$txtoutput" ] || [ $isInFile -eq 0 ] ; then
        echo "-> saving predictions"
        python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset $dataset --test_checkpoint "lightning_logs/${checkpoint}.ckpt" --save --save_xp demo-$seq --dataset_seq $seq > "$txtoutput" 2>&1
        python3 overlay_imgs.py --dataset $dataset --xp demo-$seq --model "${checkpoint}_affordances" --rgb
      fi
  done
done
