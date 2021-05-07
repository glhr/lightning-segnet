set=07; v=V001

python3 fusion-test.py --fusion custom --modalities rgb,ir --save --bs 1 --save_xp demo-s$set-$v --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-03 22-03-freiburgthermal-c3-sord-1,2,3-a1-logl2-lw-rgb,ir-epoch=41-val_loss=0.0016.ckpt" --dataset kaistped --dataset_seq set$set/$v --gpus 0
python3 lightning.py --bs 1 --dataset kaistped --test_checkpoint "lightning_logs/2021-05-06 13-48-freiburgthermal-c13-sord-1,2,3-a1-logl2-rgb-epoch=23-val_loss=0.0037.ckpt" --save --save_xp demo-s$set-$v --modalities rgb

python3 overlay_imgs.py --dataset kaistped --xp demo-s$set-$v --rgb --ir --model "fusionfusion-custom16rll-multi-2021-05-03 22-03-freiburgthermal-c3-sord-1,2,3-a1-logl2-lw-rgb,ir-epoch=41-val_loss=0.0016_affordances"

ffmpeg -r 40 -f image2 -s 1920x1080 -i results/kaistped/demo-s$set-$v/overlay/kaistped%05d-demo-s$set-$v-pred_overlay.png -vcodec libx264 -crf 25  -pix_fmt yuv420p results/kaistped/kaistped-s$set-$v.mp4
