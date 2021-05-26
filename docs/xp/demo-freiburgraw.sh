checkpoints=(
"2021-04-08 14-28-freiburg-c6-sord-1,2,3-a1-logl2-rgb-epoch=74-val_loss=0.0061"
)

dataset=freiburgraw
arg=$1
xp=

for checkpoint in "${checkpoints[@]}"
do
  for seq in 2016-03-01-12-40-50
  do
      python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset $dataset --test_checkpoint "lightning_logs/${checkpoint}.ckpt" --save --save_xp demo-$seq --dataset_seq $seq
      python3 overlay_imgs.py --dataset $dataset --xp demo-$seq --model "${checkpoint}_affordances" --rgb
  done
done
