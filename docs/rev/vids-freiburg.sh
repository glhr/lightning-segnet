checkpoints=(
"2021-03-29 09-16-cityscapes-c30-kl-rgb-epoch=190-val_loss=0.4310.ckpt"
"2021-08-15 11-58-combo-c30-kl-rgb-epoch=82-val_loss=0.1474.ckpt"
"2021-08-22 10-21-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=92-val_loss=0.0215.ckpt"
)

commands=(
"--num_classes 30 --mode convert --orig_dataset cityscapes"
"--num_classes 3 --mode affordances --orig_dataset cityscapes"
"--num_classes 3 --mode affordances --orig_dataset cityscapes"
)

seqs=(
2016-02-22-12-32-18
2016-03-01-12-40-50
2016-03-01-11-44-50
2016-02-26-14-47-29
2016-03-01-12-05-22
)

dataset=freiburgraw
arg=$1

mkdir -p results/$dataset/$xp/txt
for seq in "${seqs[@]}"
do
  for i in ${!checkpoints[@]}
  do
      txtoutput="results/${dataset}/txt/demo-${seq}-${checkpoints[$i]}.txt"
      echo "demo-${seq}-${checkpoints[$i]}"
      xp=demo-$seq
      isInFile=$(cat "$txtoutput" | grep -c "DATALOADER:0 TEST RESULTS")
      if [ ! -f "$txtoutput" ] || [ $isInFile -eq 0 ] ; then
        echo "-> saving predictions"
        python3 lightning.py --bs 1 --dataset $dataset --test_checkpoint "lightning_logs/${checkpoints[$i]}" ${commands[$i]} --save --save_xp demo-$seq --dataset_seq $seq --gpu 1 > "$txtoutput" 2>&1
        # python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${checkpoint}_affordances" --rgb
        # python3 overlay_imgs.py --xp $xp --dataset $dataset
      fi
  done
  python3 overlay_imgs.py --dataset $dataset --xp $xp --model "2021-08-15 11-58-combo-c30-kl-rgb-epoch=82-val_loss=0.1474_affordances" --model2 "2021-08-22 10-21-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=92-val_loss=0.0215_affordances" --rgb
done
