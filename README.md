# Results

trained on Freiburg

```bash
## trained from scratch - object classes
# c7
python3 lightning.py --gpus 0 --test_checkpoint lightning_logs/2021-03-22\ 12-46-freiburg-c7-kl-epoch\=512-val_loss\=0.1937.ckpt --num_classes 7 --bs 2 --mode convert --loss kl --dataset freiburg

# c6
python3 lightning.py --gpus 0 --test_checkpoint "lightning_logs/2021-03-31 08-51-freiburg-c6-kl-rgb-epoch=673-val_loss=0.2363.ckpt" --num_classes 6 --bs 2 --mode convert --dataset freiburg --loss kl

## trained from scratch - driveability

python3 lightning.py --gpus 0 --test_checkpoint lightning_logs/2021-03-22\ 20-58-freiburg-c3-kl-epoch\=676-val_loss\=0.1252.ckpt --num_classes 3 --bs 2 --mode affordances --dataset cityscapes --loss kl

python3 lightning.py --gpus 0 --test_checkpoint "lightning_logs/2021-04-01 00-16-freiburg-c3-kl-rgb-epoch=686-val_loss=0.1479.ckpt" --num_classes 3 --bs 2 --mode affordances --dataset cityscapes --loss kl

## transfer learning command
lightning.py --test_samples 0 --train --bs 8 --lr 0.0001 --optim adam --loss kl --num_classes 6 --mode affordances --dataset freiburg --max_epochs 100 --augment --orig_dataset freiburg --workers 10 --train_checkpoint "lightning_logs/2021-03-31 08-51-freiburg-c6-kl-rgb-epoch=673-val_loss=0.2363.ckpt" --update_output_layer

python3 lightning.py --gpus 0 --test_checkpoint "lightning_logs/2021-04-01 11-41-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474.ckpt" --num_classes 3 --bs 2 --mode affordances --dataset cityscapes --loss kl


## sord
python3 lightning.py --gpus 0 --test_checkpoint lightning_logs/2021-03-22\ 11-22-freiburg-c3-sord-epoch\=689-val_loss\=0.0164.ckpt --num_classes 3 --bs 2 --mode affordances --loss sord --dataset freiburg
```

trained on Cityscapes

```bash
## trained from scratch - object classes
python3 lightning.py --gpus 0 --test_checkpoint "lightning_logs/2021-03-29 09-16-cityscapes-c30-kl-rgb-epoch=190-val_loss=0.4310.ckpt" --num_classes 30 --bs 2 --mode convert --dataset cityscapes --orig_dataset cityscapes --loss kl --debug

## trained from scratch - driveability
python3 lightning.py --gpus 0 --test_checkpoint "lightning_logs/2021-03-27 14-54-cityscapes-c3-kl-rgb-epoch=191-val_loss=0.0958.ckpt" --num_classes 3 --bs 2 --mode affordances --dataset cityscapes --orig_dataset cityscapes --loss kl --debug

## transfer learning from c30 to c3 with lr == 0.001
python3 lightning.py --gpus 0 --test_checkpoint "lightning_logs/2021-03-29 22-19-cityscapes-c3-kl-rgb-epoch=12-val_loss=0.0952.ckpt" --num_classes 3 --bs 2 --mode affordances --dataset cityscapes --orig_dataset cityscapes --loss kl --debug
or
python3 lightning.py --gpus 0 --test_checkpoint "lightning_logs/2021-03-29 22-19-cityscapes-c3-kl-rgb-epoch=14-val_loss=0.0925.ckpt" --num_classes 3 --bs 2 --mode affordances --dataset cityscapes --orig_dataset cityscapes --loss kl --debug

## transfer learning from c30 to c3 with lr == 0.0001
python3 lightning.py --gpus 0 --test_checkpoint "lightning_logs/2021-03-30 08-51-cityscapes-c3-kl-rgb-epoch=15-val_loss=0.0915.ckpt" --num_classes 3 --bs 2 --mode affordances --dataset cityscapes --orig_dataset cityscapes --loss kl --debug
python3 lightning.py --gpus 0 --test_checkpoint "lightning_logs/2021-03-30 06-26-cityscapes-c30-kl-rgb-epoch=13-val_loss=0.0921.ckpt" --num_classes 3 --bs 2 --mode affordances --dataset cityscapes --orig_dataset cityscapes --loss kl --debug

## transfer learning from c30 to sord c3 with lr == 0.0001
python3 lightning.py --gpus 0 --test_checkpoint "lightning_logs/2021-03-30 10-11-cityscapes-c30-sord-0,5,10-rgb-epoch=5-val_loss=0.0853.ckpt" --num_classes 3 --bs 2 --mode affordances --dataset cityscapes --orig_dataset cityscapes --loss sord --debug
python3 lightning.py --gpus 0 --test_checkpoint "lightning_logs/2021-03-30 10-11-cityscapes-c30-sord-0,5,10-rgb-epoch=6-val_loss=0.0836.ckpt" --num_classes 3 --bs 2 --mode affordances --dataset cityscapes --orig_dataset cityscapes --loss sord --debug
python3 lightning.py --gpus 0 --test_checkpoint "lightning_logs/2021-03-30 10-11-cityscapes-c30-sord-0,5,10-rgb-epoch=13-val_loss=0.0817.ckpt" --num_classes 3 --bs 2 --mode affordances --dataset cityscapes --orig_dataset cityscapes --loss sord --debug

python3 lightning.py --gpus 0 --test_checkpoint "lightning_logs/2021-03-30 11-56-cityscapes-c30-sord-0,1,2-rgb-epoch=2-val_loss=0.0154.ckpt" --num_classes 3 --bs 2 --mode affordances --dataset cityscapes --orig_dataset cityscapes --loss sord --debug

python3 lightning.py --gpus 0 --test_checkpoint "lightning_logs/2021-03-30 11-56-cityscapes-c30-sord-0,1,2-rgb-epoch=28-val_loss=0.0132.ckpt" --num_classes 3 --bs 2 --mode affordances --dataset cityscapes --orig_dataset cityscapes --loss sord --debug

python3 lightning.py --gpus 0 --test_checkpoint "lightning_logs/2021-03-30 11-56-cityscapes-c30-sord-0,1,2-rgb-epoch=30-val_loss=0.0132.ckpt" --num_classes 3 --bs 2 --mode affordances --dataset cityscapes --orig_dataset cityscapes --loss sord --debug
```
