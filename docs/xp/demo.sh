python3 fusion-test.py --fusion custom --modalities rgb,ir --save --bs 1 --save_xp demo-s07-V001 --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-03 22-03-freiburgthermal-c3-sord-1,2,3-a1-logl2-lw-rgb,ir-epoch=41-val_loss=0.0016.ckpt" --dataset kaistped --dataset_seq set07/V001 --gpus 0

python3 overlay_imgs.py --dataset kaistped --xp demo-s07-V001 --rgb --ir --model "fusionfusion-custom16rll-multi-2021-05-03 22-03-freiburgthermal-c3-sord-1,2,3-a1-logl2-lw-rgb,ir-epoch=41-val_loss=0.0016_affordances"
