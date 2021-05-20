singlecheckpoints=(
  "2021-05-06 13-48-freiburgthermal-c13-sord-1,2,3-a1-logl2-rgb-epoch=23-val_loss=0.0037"
)
dataset=kaistped
arg=$1

for checkpoint in "${singlecheckpoints[@]}"
do
  for set in 06 07 08
  do
    for v in V000 V001 V002 V003 V004
    do
      python3 lightning.py --bs 1 --dataset $dataset --test_checkpoint "lightning_logs/${checkpoint}.ckpt" --save --save_xp demo-s$set-$v --modalities rgb --dataset_seq set$set/$v
      python3 overlay_imgs.py --dataset $dataset --xp demo-s$set-$v --model "${checkpoint}_affordances" --rgb --ir
    done
  done
done
