checkpoints=(
  "fusionfusion-custom16rll-multi-2021-05-03 22-03-freiburgthermal-c3-sord-1,2,3-a1-logl2-lw-rgb,ir-epoch=41-val_loss=0.0016"
  "fusionfusion-custom16rll-multi-2021-05-13 09-17-freiburgthermal-c3-sord-1,2,3-a1-logl2-lw-rgb,ir-epoch=27-val_loss=0.0016"
)

dataset=kaistped
arg=$1

for checkpoint in "${checkpoints[@]}"
do
  for set in 06 07 08
  do
    for v in V000 V001 V002 V003 V004
    do
      python3 fusion-test.py --bs 1 --fusion custom --modalities rgb,ir --save --save_xp demo-s$set-$v --decoders multi --fusion_activ softmax --dataset $dataset --test_checkpoint "lightning_logs/${checkpoint}.ckpt" --save --dataset_seq set$set/$v
      python3 overlay_imgs.py --dataset $dataset --xp demo-s$set-$v --model "${checkpoint}_affordances" --rgb --ir
    done
  done
done
