checkpoints=(
"2021-11-23 13-43-combo-c30-sord-1,2,3-a1-logl2-rgb-epoch=72-val_loss=0.0063.ckpt"
)

commands=(
"--num_classes 3 --mode affordances --orig_dataset cityscapes --init_channels 3"
#"--num_classes 30 --mode convert --orig_dataset cityscapes"
)

xp=colorkitti
dataset=kittiobj
arg=$1

for i in ${!checkpoints[@]}
do

      txtoutput="results/${dataset}/txt/${checkpoints[$i]}.txt"
      echo "${checkpoints[$i]}"
        echo "-> saving predictions"
        python3 lightning.py --bs 1 --dataset $dataset --test_checkpoint "lightning_logs/${checkpoints[$i]}" ${commands[$i]} --save --gpus 1 --save_xp $xp --test_set full --test_samples 10 --dataset_seq tracking
        # python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${checkpoint}_affordances" --rgb

      # python3 overlay_imgs.py --xp $xp

done
