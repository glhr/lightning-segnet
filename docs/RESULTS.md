## Chapter 6

### Object classes vs driveability

#### Trained on Freiburg

Object baseline: 2021-03-31 08-51-freiburg-c6-kl-rgb-epoch=673-val_loss=0.2363.ckpt
```bash
# object classes - prediction on Freiburg test set
python3 lightning.py --test_checkpoint "lightning_logs/2021-03-31 08-51-freiburg-c6-kl-rgb-epoch=673-val_loss=0.2363.ckpt" --num_classes 6 --bs 16 --mode convert --dataset freiburg --loss kl --debug --workers 10 > "docs/results/2021-03-31 08-51-freiburg-c6-kl-rgb-epoch=673-val_loss=0.2363.ckpt-freiburg-test.txt" 2>&1

# object classes - prediction on full Cityscapes
python3 lightning.py --test_checkpoint "lightning_logs/2021-03-31 08-51-freiburg-c6-kl-rgb-epoch=673-val_loss=0.2363.ckpt" --num_classes 6 --bs 16 --mode convert --dataset cityscapes --full --loss kl --debug --workers 10 > "docs/results/2021-03-31 08-51-freiburg-c6-kl-rgb-epoch=673-val_loss=0.2363.ckpt-cityscapes-full.txt" 2>&1

# object classes - prediction on full Kitti
python3 lightning.py --test_checkpoint "lightning_logs/2021-03-31 08-51-freiburg-c6-kl-rgb-epoch=673-val_loss=0.2363.ckpt" --num_classes 6 --bs 16 --mode convert --dataset kitti --full --loss kl --debug --workers 10 > "docs/results/2021-03-31 08-51-freiburg-c6-kl-rgb-epoch=673-val_loss=0.2363.ckpt-kitti-full.txt" 2>&1
```

Driveability from scratch: 2021-04-01 00-16-freiburg-c3-kl-rgb-epoch=686-val_loss=0.1479.ckpt
```bash
# driveability - prediction on Freiburg test set
python3 lightning.py --test_checkpoint "lightning_logs/2021-04-01 00-16-freiburg-c3-kl-rgb-epoch=686-val_loss=0.1479.ckpt" --num_classes 3 --bs 16 --mode affordances --dataset freiburg --loss kl --debug --workers 10 > "docs/results/2021-04-01 00-16-freiburg-c3-kl-rgb-epoch=686-val_loss=0.1479.ckpt-freiburg-test.txt" 2>&1

# driveability - prediction on full Cityscapes
python3 lightning.py --test_checkpoint "lightning_logs/2021-04-01 00-16-freiburg-c3-kl-rgb-epoch=686-val_loss=0.1479.ckpt" --num_classes 3 --bs 16 --mode affordances --dataset cityscapes --full --loss kl --debug --workers 10 > "docs/results/2021-04-01 00-16-freiburg-c3-kl-rgb-epoch=686-val_loss=0.1479.ckpt-cityscapes-full.txt" 2>&1

# driveability - prediction on full Kitti
python3 lightning.py --test_checkpoint "lightning_logs/2021-04-01 00-16-freiburg-c3-kl-rgb-epoch=686-val_loss=0.1479.ckpt" --num_classes 3 --bs 16 --mode affordances --dataset kitti --full --loss kl --debug --workers 10 > "docs/results/2021-04-01 00-16-freiburg-c3-kl-rgb-epoch=686-val_loss=0.1479.ckpt-kitti-full.txt" 2>&1
```

Transfer learning from object baseline: 2021-04-01 11-41-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474.ckpt
```bash
# driveability TL - prediction on Freiburg test set
python3 lightning.py --test_checkpoint "lightning_logs/2021-04-01 11-41-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474.ckpt" --num_classes 3 --bs 16 --mode affordances --dataset freiburg --loss kl --debug --workers 10 > "docs/results/2021-04-01 11-41-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474.ckpt-freiburg-test.txt" 2>&1

# driveability TL - prediction on full Cityscapes
python3 lightning.py --test_checkpoint "lightning_logs/2021-04-01 11-41-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474.ckpt" --num_classes 3 --bs 16 --mode affordances --dataset cityscapes --full --loss kl --debug --workers 10 > "docs/results/2021-04-01 11-41-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474.ckpt-cityscapes-full.txt" 2>&1

# driveability TL - prediction on full Kitti
python3 lightning.py --test_checkpoint "lightning_logs/2021-04-01 11-41-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474.ckpt" --num_classes 3 --bs 16 --mode affordances --dataset kitti --full --loss kl --debug --workers 10 > "docs/results/2021-04-01 11-41-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474.ckpt-kitti-full.txt" 2>&1
```

### Loss weighting

#### Trained on Freiburg

Transfer learning from object baseline: 2021-04-05 09-26-freiburg-c3-kl-0,1,2-rgb-epoch=94-val_loss=0.0758.ckpt
```bash
# driveability TL - prediction on Freiburg test set
python3 lightning.py --test_checkpoint "lightning_logs/2021-04-05 09-26-freiburg-c3-kl-0,1,2-rgb-epoch=94-val_loss=0.0758.ckpt" --num_classes 3 --bs 16 --mode affordances --dataset freiburg --loss kl --debug --workers 10 > "docs/results/2021-04-05 09-26-freiburg-c3-kl-0,1,2-rgb-epoch=94-val_loss=0.0758.ckpt-freiburg-test.txt" 2>&1

# driveability TL - prediction on full Cityscapes
python3 lightning.py --test_checkpoint "lightning_logs/2021-04-05 09-26-freiburg-c3-kl-0,1,2-rgb-epoch=94-val_loss=0.0758.ckpt" --num_classes 3 --bs 16 --mode affordances --dataset cityscapes --full --loss kl --debug --workers 10 > "docs/results/2021-04-05 09-26-freiburg-c3-kl-0,1,2-rgb-epoch=94-val_loss=0.0758.ckpt-cityscapes-full.txt" 2>&1

# driveability TL - prediction on full Kitti
python3 lightning.py --test_checkpoint "lightning_logs/2021-04-05 09-26-freiburg-c3-kl-0,1,2-rgb-epoch=94-val_loss=0.0758.ckpt" --num_classes 3 --bs 16 --mode affordances --dataset kitti --full --loss kl --debug --workers 10 > "docs/results/2021-04-05 09-26-freiburg-c3-kl-0,1,2-rgb-epoch=94-val_loss=0.0758.ckpt-kitti-full.txt" 2>&1
```
