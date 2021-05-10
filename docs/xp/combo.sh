checkpoint = "2021-05-10 19-34-combo-c30-sord-1,2,3-a1-logl2-rgb-epoch=36-val_loss=0.0057.ckpt"
xp = combo

for dataset in freiburg cityscapes thermalvoc synthia kitti multispectralseg freiburgthermal lostfound
do
  mkdir -p results/$dataset/$xp/txt
  python3 lightning.py --dataset $dataset --bs 1 --save --save_xp $xp --save --dataset_combo_ntrain 180 --test_checkpoint "lightning_logs/${checkpoint}" --loss_weight > "results/${dataset}/${xp}/txt/${checkpoint}.txt" 2>&1

  python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${checkpoint}_affordances" --rgb --gt
  python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${checkpoint}_affordances" --rgb
done
