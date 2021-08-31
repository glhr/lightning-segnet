checkpoints=(
"2021-03-29 09-16-cityscapes-c30-kl-rgb-epoch=190-val_loss=0.4310.ckpt"
"2021-08-26 07-09-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=109-val_loss=0.0212.ckpt"
)

commands=(
"--num_classes 30 --mode convert --orig_dataset cityscapes"
"--num_classes 3 --mode affordances --orig_dataset cityscapes"
)

seqs=(
2011_09_28_drive_0039_sync
2011_09_26_drive_0117_sync
2011_09_28_drive_0035_sync
2011_09_28_drive_0138_sync
2011_09_29_drive_0071_sync
2011_09_26_drive_0052_sync
2011_09_26_drive_0091_sync
2011_09_26_drive_0051_sync
)

dataset=kittiraw
arg=$1

mkdir -p results/$dataset/$xp/txt
for i in ${!checkpoints[@]}
do
  for seq in "${seqs[@]}"
  do
      txtoutput="results/${dataset}/txt/demo-${seq}-${checkpoints[$i]}.txt"
      echo "demo-${seq}-${checkpoints[$i]}"
      xp=demo-$seq
      isInFile=$(cat "$txtoutput" | grep -c "DATALOADER:0 TEST RESULTS")
      if [ ! -f "$txtoutput" ] || [ $isInFile -eq 0 ] ; then
        echo "-> saving predictions"
        python3 lightning.py --bs 1 --dataset $dataset --test_checkpoint "lightning_logs/${checkpoints[$i]}" ${commands[$i]} --save --save_xp demo-$seq --dataset_seq $seq --gpu 1 > "$txtoutput" 2>&1
        # python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${checkpoint}_affordances" --rgb
        python3 overlay_imgs.py --xp $xp
      fi

  done
done
