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
python3 metrics.py --distmap --final --input results/freiburg/driv_freiburg/freiburg105
python3 metrics.py --distmap --final --input results/kitti/driv_freiburg/kitti156
python3 metrics.py --distmap --final --input results/freiburgthermal/thermo/freiburgthermal-1579080512_296238130
python3 metrics.py --distmap --final --input results/multispectralseg/thermo/multispectralseg-00081D
python3 metrics.py --distmap --final --input results/thermalvoc/thermo/thermalvoc-IMG_7270
python3 metrics.py --distmap --final --input results/kaistpedann/thermo/kaistpedann-I00630_labeled
python3 metrics.py --distmap --final --input results/synthia/test/synthia-Omni_B_000115
python3 metrics.py --distmap --final --input results/lostfound/test/lostfound-07_Festplatz_Flugfeld_000002_000410
```

### Soft labels

```bash
python3 losses.py --dist l1 --alpha 2
```

### Dataset stats

will output the proportion of pixels in each class

python3 lightning.py --nopredict --test_set full --workers 10 --dataset freiburg

# Results

## Benchmarking

python3 benchmark.py # CPU-only, will use all available threads
python3 benchmark.py --cuda --force # inference on GPU

this will load existing result pickle files from results/benchmarking/* and display them
to re-run the tests, add the --force argument (will overwrite existing results)
