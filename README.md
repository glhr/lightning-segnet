# Visualization

### Data samples

will show image before and after pre-processing/data aug

python3 lightning.py --dataset kaistped --viz --modalities rgb,ir
python3 lightning.py --dataset kaistpedann --viz --modalities rgb,ir
python3 lightning.py --dataset freiburg --viz --modalities rgb,depth,ir
python3 lightning.py --dataset cityscapes --viz --modalities rgb,depthraw
python3 lightning.py --dataset cityscapes --viz --modalities rgb,depth
python3 lightning.py --dataset kitti --viz --modalities rgb,depthraw
python3 lightning.py --dataset kitti --viz --modalities rgb,depth
python3 lightning.py --dataset thermalvoc --viz --modalities rgb,ir

optionally, add --augment to see effect of augmentation on test samples
can also add --test_set train/val/test

### Loss weighting
```bash

 # visualize pixel loss on predictions
python3 lightning.py --gpus 0 --test_checkpoint "lightning_logs/2021-03-27 14-54-cityscapes-c3-kl-rgb-epoch=191-val_loss=0.0958.ckpt" --num_classes 3 --bs 1 --mode affordances --dataset cityscapes --loss compare --test_samples 10 --debug
python3 lightning.py --gpus 0 --test_checkpoint "lightning_logs/2021-04-01 00-16-freiburg-c3-kl-rgb-epoch=686-val_loss=0.1479.ckpt" --num_classes 3 --bs 1 --mode affordances --dataset freiburg --loss compare --test_samples 10 --debug

# visualize a loss weight map
python3 metrics.py --distmap
```

### Soft labels

```bash
python3 losses.py --dist l1 --alpha 2
```

### Dataset stats

will output the proportion of pixels in each class

python3 lightning.py --nopredict --test_set full --workers 10 --dataset freiburg
