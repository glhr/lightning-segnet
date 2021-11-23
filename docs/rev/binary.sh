checkpoints=(
#"2021-08-26 07-09-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=109-val_loss=0.0212.ckpt"
"2021-08-15 11-58-combo-c30-kl-rgb-epoch=82-val_loss=0.1474.ckpt" # driv
#"2021-03-29 09-16-cityscapes-c30-kl-rgb-epoch=190-val_loss=0.4310.ckpt"
"2021-11-19 14-08-combo-c30-kl-rgb-epoch=80-val_loss=0.0711.ckpt" # road
"2021-11-19 17-42-combo-c30-kl-rgb-epoch=82-val_loss=0.0912.ckpt" # free space
)

commands=(
"--num_classes 3 --mode affordances --gt driv"
"--num_classes 3 --mode affordances --gt freespace_seg"
"--num_classes 3 --mode affordances --gt road_seg"
#"--num_classes 30 --mode convert --orig_dataset cityscapes"
)

xp=rev-2class
dataset=combo
arg=$1

mkdir -p results/$dataset/${xp}-txt
for i in ${!checkpoints[@]}
do

      txtoutput="results/${dataset}/${xp}-txt/${checkpoints[$i]}.txt"
      echo "${checkpoints[$i]}"
      isInFile=$(cat "$txtoutput" | grep -c "DATALOADER:0 TEST RESULTS")
        echo "-> saving predictions"
        python3 lightning.py --bs 1 --dataset $dataset --test_checkpoint "lightning_logs/${checkpoints[$i]}" ${commands[$i]} --gpus 1 --test_set val > "$txtoutput" 2>&1
        # python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${checkpoint}_affordances" --rgb

      # python3 overlay_imgs.py --xp $xp

done
