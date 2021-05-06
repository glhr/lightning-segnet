## Freiburg

## Objects
python3 lightning.py --test_checkpoint "lightning_logs/2021-03-31 08-51-freiburg-c6-kl-rgb-epoch=673-val_loss=0.2363.ckpt" --num_classes 6 --bs 1 --mode convert --dataset freiburg --workers 10 --save --save_xp driv # done

python3 lightning.py --test_checkpoint "lightning_logs/2021-03-31 08-51-freiburg-c6-kl-rgb-epoch=673-val_loss=0.2363.ckpt" --num_classes 6 --bs 1 --mode convert --dataset kitti --orig_dataset freiburg --workers 10 --save --save_xp driv --test_set full

python3 lightning.py --test_checkpoint "lightning_logs/2021-03-31 08-51-freiburg-c6-kl-rgb-epoch=673-val_loss=0.2363.ckpt" --num_classes 6 --bs 1 --mode convert --dataset kitti --orig_dataset freiburg --workers 10 --save --save_xp driv --test_set full

## Driv

python3 lightning.py --test_checkpoint "lightning_logs/2021-04-01 00-16-freiburg-c3-kl-rgb-epoch=686-val_loss=0.1479.ckpt" --num_classes 3 --bs 1 --mode affordances --dataset freiburg --workers 10 --save --save_xp driv # done

python3 lightning.py --test_checkpoint "lightning_logs/2021-04-01 00-16-freiburg-c3-kl-rgb-epoch=686-val_loss=0.1479.ckpt" --num_classes 3 --bs 1 --mode affordances --dataset kitti --orig_dataset freiburg --workers 10 --save --save_xp driv --test_set full

python3 lightning.py --test_checkpoint "lightning_logs/2021-04-01 00-16-freiburg-c3-kl-rgb-epoch=686-val_loss=0.1479.ckpt" --num_classes 3 --bs 1 --mode affordances --dataset cityscapes --orig_dataset freiburg --workers 10 --save --save_xp driv --test_set full

## Transfer learning

python3 lightning.py --test_checkpoint "lightning_logs/2021-04-08 13-31-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474.ckpt" --num_classes 3 --bs 1 --mode affordances --dataset freiburg --workers 10 --save --save_xp driv # done

## Visualize

python3 overlay_imgs.py --dataset freiburg --xp driv --model "2021-03-31 08-51-freiburg-c6-kl-rgb-epoch=673-val_loss=0.2363_convert" --rgb --model2 "2021-04-01 00-16-freiburg-c3-kl-rgb-epoch=686-val_loss=0.1479_affordances" --model3 "2021-04-08 13-31-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474_affordances" --gt

python3 overlay_imgs.py --dataset kitti --xp driv --model "2021-03-31 08-51-freiburg-c6-kl-rgb-epoch=673-val_loss=0.2363_convert" --rgb --model2 "2021-04-01 00-16-freiburg-c3-kl-rgb-epoch=686-val_loss=0.1479_affordances" --model3 "2021-04-08 13-31-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474_affordances" --gt

## Cityscapes

## OBjects

python3 lightning.py --test_checkpoint "lightning_logs/2021-03-29 09-16-cityscapes-c30-kl-rgb-epoch=190-val_loss=0.4310.ckpt" --num_classes 30 --bs 1 --mode convert --dataset cityscapes --orig_dataset cityscapes --workers 10 --save --save_xp driv # done

## Driv

python3 lightning.py --test_checkpoint "lightning_logs/2021-03-27 14-54-cityscapes-c3-kl-rgb-epoch=191-val_loss=0.0958.ckpt" --num_classes 3 --bs 1 --mode affordances --dataset cityscapes --orig_dataset cityscapes --workers 10 --save --save_xp driv # done

## Transfer learning

python3 lightning.py --test_checkpoint "lightning_logs/2021-04-09 03-40-cityscapes-c30-kl-rgb-epoch=18-val_loss=0.0918.ckpt" --num_classes 3 --bs 1 --mode affordances --dataset cityscapes --orig_dataset cityscapes --workers 10 --save --save_xp driv # done

## Visualize

python3 overlay_imgs.py --dataset cityscapes --xp driv --model "2021-03-29 09-16-cityscapes-c30-kl-rgb-epoch=190-val_loss=0.4310_convert" --rgb --model2 "2021-03-27 14-54-cityscapes-c3-kl-rgb-epoch=191-val_loss=0.0958_affordances" --model3 "2021-04-09 03-40-cityscapes-c30-kl-rgb-epoch=18-val_loss=0.0918_affordances" --gt
