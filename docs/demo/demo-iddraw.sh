checkpoints=(
"2021-05-12 11-24-combo-c30-sord-1,2,3-a1-logl2-rgb-epoch=42-val_loss=0.0060"
"2021-05-12 14-34-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=49-val_loss=0.0023"
)

dataset=cityscapesraw
arg=$1
xp=demo

# seq = 2016-02-22-12-32-18  2016-02-22-12-47-00  2016-02-26-14-51-16  2016-02-26-15-26-10  2016-03-01-11-54-41  2016-03-01-12-11-45 2016-02-22-12-37-11  2016-02-26-14-39-14
# 2016-02-26-15-05-05  2016-03-01-11-44-50  2016-03-01-11-57-14  2016-03-01-12-19-07 2016-02-22-12-42-53  2016-02-26-14-47-29  2016-02-26-15-20-48  2016-03-01-11-50-45  2016-03-01-12-05-22  2016-03-01-12-40-50

mkdir -p results/$dataset/$xp/txt
for checkpoint in "${checkpoints[@]}"
do
  for seq in frontpage
  do
      txtoutput="results/${dataset}/${xp}/txt/demo-${seq}-${checkpoint}.txt"
      echo "demo-${seq}-${checkpoint}"
      isInFile=$(cat "$txtoutput" | grep -c "DATALOADER:0 TEST RESULTS")
      if [ ! -f "$txtoutput" ] || [ $isInFile -eq 0 ] ; then
        echo "-> saving predictions"
        python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset $dataset --test_checkpoint "lightning_logs/${checkpoint}.ckpt" --save --save_xp demo-$seq --dataset_seq $seq --gpus 0 > "$txtoutput" 2>&1
        python3 overlay_imgs.py --dataset $dataset --xp demo-$seq --model "${checkpoint}_affordances" --rgb
      fi
  done
done
