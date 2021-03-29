# Results

trained on Freiburg

```bash
python3 lightning.py --gpus 0 --test_checkpoint lightning_logs/2021-03-22\ 20-58-freiburg-c3-kl-epoch\=676-val_loss\=0.1252.ckpt --num_classes 3 --bs 2 --mode affordances --dataset cityscapes --loss kl

python3 lightning.py --gpus 0 --test_checkpoint lightning_logs/2021-03-22\ 11-22-freiburg-c3-sord-epoch\=689-val_loss\=0.0164.ckpt --num_classes 3 --bs 2 --mode affordances --loss sord --dataset freiburg

python3 lightning.py --gpus 0 --test_checkpoint lightning_logs/2021-03-22\ 12-46-freiburg-c7-kl-epoch\=512-val_loss\=0.1937.ckpt --num_classes 7 --bs 2 --mode convert --loss kl --dataset freiburg
```

trained on Cityscapes

```bash
python3 lightning.py --gpus 0 --test_checkpoint "lightning_logs/2021-03-29 09-16-cityscapes-c30-kl-rgb-epoch=190-val_loss=0.4310.ckpt" --num_classes 30 --bs 2 --mode convert --dataset cityscapes --orig_dataset cityscapes --loss kl --test_samples 2

python3 lightning.py --gpus 0 --test_checkpoint "lightning_logs/2021-03-27 14-54-cityscapes-c3-kl-rgb-epoch=191-val_loss=0.0958.ckpt" --num_classes 3 --bs 2 --mode affordances --dataset cityscapes --orig_dataset cityscapes --loss kl --test_samples 2
```
