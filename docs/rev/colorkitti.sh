checkpoints=(
"2021-11-23 13-43-combo-c30-sord-1,2,3-a1-logl2-rgb-epoch=7-val_loss=0.0077.ckpt"
)

commands=(
"--num_classes 3 --mode affordances --orig_dataset cityscapes"
#"--num_classes 30 --mode convert --orig_dataset cityscapes"
)

xp=rev-xp
dataset=kittiobj
arg=$1

mkdir -p results/$dataset/$xp/txt
for i in ${!checkpoints[@]}
do

      txtoutput="results/${dataset}/txt/${checkpoints[$i]}.txt"
      echo "${checkpoints[$i]}"
      isInFile=$(cat "$txtoutput" | grep -c "DATALOADER:0 TEST RESULTS")
        echo "-> saving predictions"
        python3 lightning.py --bs 1 --dataset $dataset --test_checkpoint "lightning_logs/${checkpoints[$i]}" ${commands[$i]} --save --gpus 1 --save_xp $xp --test_set full --dataset_seq tracking
        # python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${checkpoint}_affordances" --rgb

      # python3 overlay_imgs.py --xp $xp

done
