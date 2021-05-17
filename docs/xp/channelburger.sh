# Freiburg
## Baselines
python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset freiburg --loss kl --loss_weight --test_checkpoint "lightning_logs/2021-04-08 13-31-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474.ckpt" --save --save_xp burger --modalities rgb
python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset freiburg --loss kl --loss_weight --test_checkpoint "lightning_logs/2021-04-17 14-18-freiburg-c6-kl-depth-epoch=149-val_loss=0.3106.ckpt" --save --save_xp burger --modalities depth
python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset freiburg --loss kl --loss_weight --test_checkpoint "lightning_logs/2021-04-17 13-16-freiburg-c6-kl-ir-epoch=128-val_loss=0.1708.ckpt" --save --save_xp burger --modalities ir

## Channel stacking
python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset freiburg --loss kl --loss_weight --test_checkpoint "lightning_logs/2021-04-17 18-57-freiburg-c6-kl-rgb,depth-epoch=87-val_loss=0.1472.ckpt" --save --save_xp burger --modalities rgb,depth
python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset freiburg --loss kl --loss_weight --test_checkpoint "lightning_logs/2021-04-17 19-40-freiburg-c6-kl-rgb,ir-epoch=149-val_loss=0.1349.ckpt" --save --save_xp burger --modalities rgb,ir
python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset freiburg --loss kl --loss_weight --test_checkpoint "lightning_logs/2021-04-17 20-25-freiburg-c6-kl-rgb,depth,ir-epoch=81-val_loss=0.1352.ckpt" --save --save_xp burger --modalities rgb,depth,ir

# Cityscapes
## Baselines
python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset cityscapes --loss kl --loss_weight --test_checkpoint "lightning_logs/2021-04-09 03-40-cityscapes-c30-kl-rgb-epoch=18-val_loss=0.0918.ckpt" --save --save_xp burger --modalities rgb
python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset cityscapes --loss kl --loss_weight --test_checkpoint "lightning_logs/2021-04-18 13-12-cityscapes-c30-kl-depthraw-epoch=22-val_loss=0.1251.ckpt" --save --save_xp burger-rgb,draw --modalities depthraw
python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset cityscapes --loss kl --loss_weight --test_checkpoint "lightning_logs/2021-04-17 23-19-cityscapes-c30-kl-depth-epoch=23-val_loss=0.1222.ckpt" --save --save_xp burger-rgb,d --modalities depth

## Channel stacking
python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset cityscapes --loss kl --loss_weight --test_checkpoint "lightning_logs/2021-04-18 10-12-cityscapes-c30-kl-rgb,depthraw-epoch=23-val_loss=0.1024.ckpt" --save --save_xp burger-rgb,draw --modalities rgb,depthraw
python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset cityscapes --loss kl --loss_weight --test_checkpoint "lightning_logs/2021-04-18 00-48-cityscapes-c30-kl-rgb,depth-epoch=23-val_loss=0.0999.ckpt" --save --save_xp burger-rgb,d --modalities rgb,depth
