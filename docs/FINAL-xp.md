## FReiburg thermal

### Single modality

python3 lightning.py --bs 1 --dataset freiburgthermal --test_checkpoint "lightning_logs/2021-05-06 13-48-freiburgthermal-c13-sord-1,2,3-a1-logl2-rgb-epoch=23-val_loss=0.0037.ckpt" --save --save_xp mishmash --modalities rgb --loss_weight
```bash
[INFO] CM IoU - tensor([96.9616, 61.0726, 90.2493])
[INFO] precision tensor([98.1684, 73.6862, 96.1385], dtype=torch.float64) (89.33102728174495) | recall tensor([98.7480, 78.1074, 93.6439], dtype=torch.float64) (90.1664395770831)
Testing: 100%|██████████████████████████████| 1115/1115 [23:08<00:00,  1.25s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9560766816139221,
 'test_acc_w': 0.959296703338623,
 'test_dist_l1': 0.050574589520692825,
 'test_dist_l2': 0.06387706100940704,
 'test_dist_logl2': 0.018044644966721535,
 'test_dist_mistake_severity': 0.1514282375574112,
 'test_iou': 0.9185949563980103}
```

### Fusion

RGB encoder initialized with obj model from cityscapes
IR encoder trained on obj classes from freiburg thermal
then put into fusion architecture, with output layer = 3 to learn driveability

python3 fusion-test.py  --bs 1 --fusion custom --dataset freiburgthermal --modalities rgb,ir --save --bs 1 --save_xp mishmash --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-03 11-30-freiburgthermal-c3-sord-1,2,3-a1-logl2-rgb,ir-epoch=39-val_loss=0.0038.ckpt" --loss_weight --fusion_activ softmax
```bash
[INFO] CM IoU - tensor([96.8845, 61.2904, 90.1812])
[INFO] precision tensor([97.9077, 74.6868, 96.2584], dtype=torch.float64) (89.61764044109266) | recall tensor([98.9328, 77.3603, 93.4572], dtype=torch.float64) (89.91677552326452)
Testing: 100%|█████████████████████████████████████████████████████████| 1115/1115 [28:33<00:00,  1.54s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9560661315917969,
 'test_acc_w': 0.9596188068389893,
 'test_dist_l1': 0.051289044320583344,
 'test_dist_l2': 0.06599943339824677,
 'test_dist_logl2': 0.018726205453276634,
 'test_dist_mistake_severity': 0.16741521656513214,
 'test_iou': 0.918484628200531}
```

with loss weighting

python3 fusion-test.py  --bs 1 --fusion custom --dataset freiburgthermal --modalities rgb,ir --save --bs 1 --save_xp mishmash --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-03 22-03-freiburgthermal-c3-sord-1,2,3-a1-logl2-lw-rgb,ir-epoch=41-val_loss=0.0016.ckpt" --loss_weight --fusion_activ softmax
```bash
[INFO] CM IoU - tensor([96.9692, 61.5706, 90.4489])
[INFO] precision tensor([97.9649, 76.0129, 96.0158], dtype=torch.float64) (89.99787428072003) | recall tensor([98.9627, 76.4184, 93.9760], dtype=torch.float64) (89.78571661945155)
Testing: 100%|█████████████████████████████████████████████████████████| 1115/1115 [28:17<00:00,  1.52s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9571654200553894,
 'test_acc_w': 0.9612782001495361,
 'test_dist_l1': 0.04996063560247421,
 'test_dist_l2': 0.06421279907226562,
 'test_dist_logl2': 0.018208062276244164,
 'test_dist_mistake_severity': 0.166362926363945,
 'test_iou': 0.9204030632972717}
```

RGB encoder trained on obj classes from freiburg thermal
IR encoder trained on obj classes from freiburg thermal
then put into fusion architecture, with output layer = 3 to learn driveability

python3 fusion-test.py  --bs 1 --fusion custom --dataset freiburgthermal --modalities rgb,ir --save --bs 1 --save_xp mishmash --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-07 11-08-freiburgthermal-c3-sord-1,2,3-a1-logl2-rgb,ir-epoch=25-val_loss=0.0036.ckpt" --loss_weight --fusion_activ softmax
```bash

```

## ThermalVOC

### Single modality

python3 lightning.py --bs 1 --dataset thermalvoc --test_checkpoint "lightning_logs/2021-05-06 13-48-freiburgthermal-c13-sord-1,2,3-a1-logl2-rgb-epoch=23-val_loss=0.0037.ckpt" --save --save_xp mishmash --modalities rgb --loss_weight
```bash
[INFO] CM IoU - tensor([88.7696,  0.0000,  0.0000])
/usr/local/lib/python3.8/dist-packages/torch/tensor.py:521: RuntimeWarning: invalid value encountered in multiply
  return self.to(torch.get_default_dtype()).reciprocal() * other
[INFO] precision tensor([100.0000,   0.0000,   0.0000], dtype=torch.float64) (33.33333370561293) | recall tensor([88.7696,     nan,     nan], dtype=torch.float64) (nan)
Testing: 100%|██████████████████████████████| 1659/1659 [33:09<00:00,  1.20s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.8876955509185791,
 'test_acc_w': 0.8374802470207214,
 'test_dist_l1': 0.14250725507736206,
 'test_dist_l2': 0.20291279256343842,
 'test_dist_logl2': 0.07589922100305557,
 'test_dist_mistake_severity': 0.2689364552497864,
 'test_iou': 0.8158679604530334}
```

### Fusion

python3 fusion-test.py  --bs 1 --fusion custom --dataset thermalvoc --modalities rgb,ir --save --bs 1 --save_xp mishmash --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-03 11-30-freiburgthermal-c3-sord-1,2,3-a1-logl2-rgb,ir-epoch=39-val_loss=0.0038.ckpt" --loss_weight --fusion_activ softmax
```bash

```

python3 fusion-test.py  --bs 1 --fusion custom --dataset thermalvoc --modalities rgb,ir --save --bs 1 --save_xp mishmash --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-03 22-03-freiburgthermal-c3-sord-1,2,3-a1-logl2-lw-rgb,ir-epoch=41-val_loss=0.0016.ckpt" --loss_weight --fusion_activ softmax

```bash

```

python3 fusion-test.py  --bs 1 --fusion custom --dataset thermalvoc --modalities rgb,ir --save --bs 1 --save_xp mishmash --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-07 11-08-freiburgthermal-c3-sord-1,2,3-a1-logl2-rgb,ir-epoch=25-val_loss=0.0036.ckpt" --loss_weight --fusion_activ softmax
