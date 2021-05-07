python3 fusion-test.py --fusion custom --dataset freiburgthermal --modalities rgb,ir --save --bs 1 --save_xp mishmash --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-03 11-30-freiburgthermal-c3-sord-1,2,3-a1-logl2-rgb,ir-epoch=39-val_loss=0.0038.ckpt" --loss_weight


python3 fusion-test.py --fusion custom --modalities rgb,ir --save --bs 1 --save_xp mishmash --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-03 22-03-freiburgthermal-c3-sord-1,2,3-a1-logl2-lw-rgb,ir-epoch=41-val_loss=0.0016.ckpt" --dataset thermalvoc

python3 lightning.py --bs 1 --dataset thermalvoc --test_checkpoint "lightning_logs/2021-05-06 13-48-freiburgthermal-c13-sord-1,2,3-a1-logl2-rgb-epoch=23-val_loss=0.0037.ckpt" --save --save_xp mishmash --modalities rgb


python3 fusion-test.py --fusion custom --modalities rgb,ir --save --bs 1 --save_xp mishmash --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-03 22-03-freiburgthermal-c3-sord-1,2,3-a1-logl2-lw-rgb,ir-epoch=41-val_loss=0.0016.ckpt" --dataset kaistpedann --gpus 0
