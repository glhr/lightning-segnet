parameters
```bash
channels=3    # 3 for RGB, 1 for grayscale
xp=vap_gray   # name of folder to save result pics in
```

# cityscapes

## train SegNet from scratch on Cityscapes object classes:

```bash
python3 lightning.py --test_samples 0 --train --bs 8 --lr 0.001 --optim adam --loss kl --num_classes 30 --mode convert --dataset cityscapes --max_epochs 300 --augment --workers 4 --gpu 1 --loss kl --init_channels $channels
```
test:
```bash
python3 lightning.py --test_checkpoint "$checkpoint" --num_classes 30 --bs 16 --mode convert --dataset $dataset --orig_dataset cityscapes --workers 4 --gpu 1 --save_xp $xp --test_set full --init_channels $channels --save
```

checkpoints:

with color aug: "lightning_logs/2021-08-13 15-27-cityscapes-c30-kl-rgb-epoch=197-val_loss=0.4043.ckpt"

## transfer learning to driveability with SORD:

```bash
objcheckpoint="lightning_logs/2021-08-12 15-52-cityscapes-c30-kl-rgb-epoch=161-val_loss=0.4119.ckpt"
lightning.py --test_samples 0 --train --bs 8 --lr 0.0001 --optim adam --loss sord --num_classes 30 --mode affordances --dataset cityscapes --max_epochs 25 --augment --orig_dataset cityscapes --workers 4 --gpu 1 --train_checkpoint "$objcheckpoint" --update_output_layer --ranks 1,2,3 --dist logl2 --dist_alpha 1 --init_channels $channels
```
test:
```bash
python3 lightning.py --test_checkpoint "$checkpoint" --num_classes 3 --bs 16 --mode affordances --dataset $dataset --orig_dataset cityscapes --workers 4 --gpu 1 --save --save_xp $xp --init_channels $channels --test_set full
```
## transfer learning to driveability with one-hot:

```bash
lightning.py --test_samples 0 --train --bs 8 --lr 0.0001 --optim adam --loss sord --num_classes 30 --mode affordances --dataset cityscapes --max_epochs 25 --augment --orig_dataset cityscapes --workers 4 --gpu 1 --train_checkpoint "$objcheckpoint" --update_output_layer --dist_alpha 1 --init_channels $channels
```

# combo

```bash
objcheckpoint="lightning_logs/2021-08-12 15-52-cityscapes-c30-kl-rgb-epoch=161-val_loss=0.4119.ckpt"
lightning.py --test_samples 0 --train --bs 8 --lr 0.0001 --optim adam --num_classes 30 --mode affordances --dataset combo --max_epochs 200 --augment --workers 4 --train_checkpoint "$objcheckpoint" --update_output_layer --gpu 1 --loss sord --dist logl2 --dist_alpha 1


COMBO RGB
objcheckpoint="lightning_logs/2021-08-13 15-27-cityscapes-c30-kl-rgb-epoch=197-val_loss=0.4043.ckpt"
python3 lightning-combo.py --test_samples 0 --train --bs 8 --lr 0.0001 --optim adam --num_classes 30 --mode affordances --dataset combo --max_epochs 90 --augment --workers 4 --train_checkpoint "$objcheckpoint" --update_output_layer --gpu 1 --loss kl --init_channels 3
```

transfer learning SORD

python3 lightning-combo.py --test_samples 0 --train --bs 8 --lr 0.0001 --optim adam --num_classes 30 --mode affordances --dataset combo --max_epochs 90 --augment --workers 4 --train_checkpoint "lightning_logs/2021-03-29 09-16-cityscapes-c30-kl-rgb-epoch=190-val_loss=0.4310.ckpt" --update_output_layer --gpu 1 --loss sord --ranks 1,2,3 --dist logl2 --dist_alpha 1 --init_channels 1

LW step:

python3 lightning-combo.py --test_samples 0 --train --bs 8 --lr 0.0001 --optim adam --loss sord --num_classes 3 --mode affordances --dataset combo --max_epochs 200 --augment --workers 4 --train_checkpoint "lightning_logs/2021-08-16 09-02-combo-c30-sord-1,2,3-a1-logl2-rgb-epoch=78-val_loss=0.0062.ckpt" --ranks 1,2,3 --dist logl2 --dist_alpha 1 --loss_weight --lwmap_range 0,10 --init_channels 1 --gpu 1
