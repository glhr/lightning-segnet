This is a sub-module of https://github.com/glhr/learning-driveability-heatmaps

# Content

```python
.
├── docs                    # bash scripts for getting experiment results + summaries of evaluation runs 
├── requirements            # separate pip requirements.txt files for setting this up on different platforms
├── results                 # raw output of evaluation runs
├── misc                    # miscellaneous utility scripts
├── dataloader.py           # interfaces with a bunch of datasets, handles mappings between different class definitions
├── segnet.py               # defines the single-modality/channel stacking SegNet model
├── fusion.py               # defines the fusion architecture
├── lightning.py            # training & evaluation for single-modality/channel stacking SegNet model
├── fusion-test.py          # training & evaluation for deep fusion models
├── losses.py               # KL, SORD & distance-based loss
├── metrics.py              # IoU & regression metrics & loss weight map computation
├── plotting.py             # self-explanatory
├── utils.py                # useful helper functions
└── Singularity             # for building container in CLAAUDIA
```

# Installation

```bash
# set up virtual environmen
python3 -m venv venv
source venv/bin/activate

# install requirements:

pip install --upgrade pip && pip install -r requirements/requirements.txt  # or requirements/requirements_claaudia.txt

# for Jetson Xavier (Jetpack 4.4):
pip3 install numpy==1.19.4
wget -O torch-1.4.0-cp36-cp36m-linux_aarch64.whl https://nvidia.box.com/shared/static/ncgzus5o23uck9i5oth2n8n06k340l6k.whl # see https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-7-0-now-available/72048
pip install torch-1.4.0-cp36-cp36m-linux_aarch64.whl
git clone --branch v0.5.0 https://github.com/pytorch/vision torchvision   # see below for version of torchvision to download
pip install -r requirements_jetson.txt
```


# Visualization

### Data samples

will show image before and after pre-processing/data aug

```bash
python3 lightning.py --dataset kaistped --viz --modalities rgb,ir
python3 lightning.py --dataset kaistpedann --viz --modalities rgb,ir
python3 lightning.py --dataset freiburg --viz --modalities rgb,depth,ir
python3 lightning.py --dataset cityscapes --viz --modalities rgb,depthraw
python3 lightning.py --dataset cityscapes --viz --modalities rgb,depth
python3 lightning.py --dataset kitti --viz --modalities rgb,depthraw
python3 lightning.py --dataset kitti --viz --modalities rgb,depth
python3 lightning.py --dataset thermalvoc --viz --modalities rgb,ir
```

optionally, add `--augment` to see effect of augmentation on test samples
can also add `--test_set` train/val/test

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
```
python3 lightning.py --nopredict --test_set full --workers 10 --dataset freiburg
```

# Results

## Evaluation scrips

these will spit out results and save predictions on the test sets
```
.
├── docs/xp/driv.sh          # Chapter 6 - object classes to driveability
├── docs/xp/sord.sh          # Chapter 6 - soft labels for ordinal segmentation
├── docs/xp/lw.sh            # Chapter 6 - loss weighting
├── docs/xp/channelburger.sh # Chapter 8 - channel stacking
├── docs/xp/deep-fusion.sh   # Chapter 8 - cooler fusion
├── docs/xp/combo.sh         # Chapter 9 - cross-dataset
├── docs/xp/thermo.sh        # Chapter 9 - thermal fusion
```

## Benchmarking

```
python3 benchmark.py # CPU-only, will use all available threads
python3 benchmark.py --cuda --force # inference on GPU

this will load existing result pickle files from results/benchmarking/* and display them
to re-run the tests, add the --force argument (will overwrite existing results)
```
