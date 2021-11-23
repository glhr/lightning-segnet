checkpoints=(
#"2021-08-15 11-58-combo-c30-kl-rgb-epoch=82-val_loss=0.1474.ckpt"
"2021-08-16 09-02-combo-c30-sord-1,2,3-a1-logl2-rgb-epoch=78-val_loss=0.0062.ckpt"
# "2021-08-26 07-09-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=109-val_loss=0.0212.ckpt"
#"2021-08-22 10-21-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=92-val_loss=0.0215.ckpt"
)

commands=(
"--num_classes 3 --mode affordances --gt driv"
"--num_classes 3 --mode affordances --gt driv"
"--num_classes 3 --mode affordances --gt driv"
#"--num_classes 30 --mode convert --orig_dataset cityscapes"
)

datasets=(
#"kitti"
#"freiburg"
"wilddash"
#"--num_classes 30 --mode convert --orig_dataset cityscapes"
)

datasetssets=(
#"full"
#"full"
"test"
)

xp=rev-ablation
arg=$1


for d in ${!datasets[@]}
  do
  dataset=${datasets[$d]}
  mkdir -p results/$dataset/${xp}-txt
  for i in ${!checkpoints[@]}
  do

        txtoutput="results/${dataset}/${xp}-txt/${checkpoints[$i]}-lw.txt"
        echo "${txtoutput}"
        isInFile=$(cat "$txtoutput" | grep -c "DATALOADER:0 TEST RESULTS")
        if [ ! -f "$txtoutput" ] || [ $isInFile -eq 0 ] ; then
          echo "-> saving predictions"
          python3 lightning.py --bs 1 --dataset $dataset --test_checkpoint "lightning_logs/${checkpoints[$i]}" ${commands[$i]} --gpus 1 --test_set ${datasetssets[$dataset]} --loss_weight > "$txtoutput" 2>&1
          # python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${checkpoint}_affordances" --rgb
        fi
        # python3 overlay_imgs.py --xp $xp

  done
done
