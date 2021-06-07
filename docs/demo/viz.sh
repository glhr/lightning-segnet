python3 overlay_imgs.py --dataset freiburg --xp lw --model2 "2021-04-07 11-31-freiburg-c6-kl-0,1,2-rgb-epoch=43-val_loss=0.0771_affordances" --model "2021-04-08 13-31-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474_affordances" --gt

python3 overlay_imgs.py --dataset cityscapes --xp lw --model2 "2021-04-09 10-00-cityscapes-c30-kl-rgb-epoch=6-val_loss=0.0283_affordances" --model "2021-04-09 03-40-cityscapes-c30-kl-rgb-epoch=18-val_loss=0.0918_affordances" --gt

python3 overay_imgs.py --dataset kaistped --xp mishmash --model "fusionfusion-custom16rll-multi-2021-05-03 22-03-freiburgthermal-c3-sord-1,2,3-a1-logl2-lw-rgb,ir-epoch=41-val_loss=0.0016_affordances" --ir

python3 overlay_imgs.py --dataset thermalvoc --xp mishmash --model "fusionfusion-custom16rll-multi-2021-05-03 11-30-freiburgthermal-c3-sord-1,2,3-a1-logl2-rgb,ir-epoch=39-val_loss=0.0038_affordances" --rgb --ir --model2 "fusionfusion-custom16rll-multi-2021-05-03 22-03-freiburgthermal-c3-sord-1,2,3-a1-logl2-lw-rgb,ir-epoch=41-val_loss=0.0016_affordances"

python3 overlay_imgs.py --dataset thermalvoc --xp mishmash --model "2021-05-06 13-48-freiburgthermal-c13-sord-1,2,3-a1-logl2-rgb-epoch=23-val_loss=0.0037_affordances" --rgb --gt
