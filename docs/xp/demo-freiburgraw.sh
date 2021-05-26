checkpoints=(
"2021-04-08 14-28-freiburg-c6-sord-1,2,3-a1-logl2-rgb-epoch=74-val_loss=0.0061"
)

dataset=freiburgraw
arg=$1
xp=

for checkpoint in "${checkpoints[@]}"
do
  for seq in 2016-02-22-12-32-18  2016-02-22-12-47-00  2016-02-26-14-51-16  2016-02-26-15-26-10  2016-03-01-11-54-41  2016-03-01-12-11-45 2016-02-22-12-37-11  2016-02-26-14-39-14  2016-02-26-15-05-05  2016-03-01-11-44-50  2016-03-01-11-57-14  2016-03-01-12-19-07 2016-02-22-12-42-53  2016-02-26-14-47-29  2016-02-26-15-20-48  2016-03-01-11-50-45  2016-03-01-12-05-22  2016-03-01-12-40-50
  do
      python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset $dataset --test_checkpoint "lightning_logs/${checkpoint}.ckpt" --save --save_xp demo-$seq --dataset_seq $seq
      python3 overlay_imgs.py --dataset $dataset --xp demo-$seq --model "${checkpoint}_affordances" --rgb
  done
done
