checkpoints=(
#"2021-08-26 07-09-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=109-val_loss=0.0212.ckpt"
"2021-08-22 10-21-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=92-val_loss=0.0215.ckpt"
#"2021-08-15 11-58-combo-c30-kl-rgb-epoch=82-val_loss=0.1474.ckpt"
#"2021-03-29 09-16-cityscapes-c30-kl-rgb-epoch=190-val_loss=0.4310.ckpt"
)

commands=(
"--num_classes 3 --mode affordances --orig_dataset cityscapes"
#"--num_classes 30 --mode convert --orig_dataset cityscapes"
)

seqs=(
"object"
"tracking"
)

xp=kittiobj
dataset=kittiobj
arg=$1

mkdir -p results/$dataset/${xp}-txt
for i in ${!checkpoints[@]}
do
      for s in ${!seqs[@]}
      do
        txtoutput="results/${dataset}/${xp}-txt/${seqs[$s]}-${checkpoints[$i]}.txt"
        echo "$txtoutput"
        isInFile=$(cat "$txtoutput" | grep -c "DATALOADER:0 TEST RESULTS")
        if [ ! -f "$txtoutput" ] || [ $isInFile -eq 0 ] ; then
          echo "-> saving predictions"
          python3 lightning.py --bs 8 --dataset $dataset --test_checkpoint "lightning_logs/${checkpoints[$i]}" ${commands[$i]} --save --gpus 1 --save_xp $xp --test_set full --dataset_seq ${seqs[$s]} > "$txtoutput" 2>&1
          # python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${checkpoint}_affordances" --rgb
        end
      fi
    done
      # python3 overlay_imgs.py --xp $xp

done
