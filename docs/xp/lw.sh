## Freiburg

python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset freiburg --test_checkpoint "lightning_logs/2021-04-08 13-31-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474.ckpt" --save --save_xp lw --modalities rgb --loss_weight
python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset freiburg --test_checkpoint "lightning_logs/2021-04-07 11-31-freiburg-c6-kl-0,1,2-rgb-epoch=43-val_loss=0.0771.ckpt" --save --save_xp lw --modalities rgb --loss_weight

## Cityscapes

python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset cityscapes --test_checkpoint "lightning_logs/2021-04-09 03-40-cityscapes-c30-kl-rgb-epoch=18-val_loss=0.0918.ckpt" --save --save_xp lw --modalities rgb --gpus 0
python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset cityscapes --test_checkpoint "lightning_logs/2021-04-09 10-00-cityscapes-c30-kl-rgb-epoch=6-val_loss=0.0283.ckpt" --save --save_xp lw --modalities rgb --gpus 0
