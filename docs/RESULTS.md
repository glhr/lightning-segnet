## Chapter 6

### Object classes vs driveability

#### Trained on Freiburg

pred on Freiburg test set
```bash
# object classes
python3 lightning.py --gpus 1 --test_checkpoint "lightning_logs/2021-03-31 08-51-freiburg-c6-kl-rgb-epoch=673-val_loss=0.2363.ckpt" --num_classes 6 --bs 16 --mode convert --dataset freiburg --loss kl --debug --workers 10 > "docs/results/2021-03-31 08-51-freiburg-c6-kl-rgb-epoch=673-val_loss=0.2363.ckpt-freiburg-test"
```
