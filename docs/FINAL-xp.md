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

python3 fusion-test.py  --bs 1 --fusion custom --dataset freiburgthermal --modalities rgb,ir --save --bs 1 --save_xp mishmash --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-03 11-30-freiburgthermal-c3-sord-1,2,3-a1-logl2-rgb,ir-epoch=39-val_loss=0.0038.ckpt" --loss_weight
```bash
[INFO] CM IoU - tensor([96.9657, 58.9908, 89.8577])
[INFO] precision tensor([97.8827, 76.5533, 95.0873], dtype=torch.float64) (89.84110661041362) | recall tensor([99.0431, 71.9995, 94.2324], dtype=torch.float64) (88.42499848473793)
Testing: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1115/1115 [28:37<00:00,  1.54s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9551512002944946,
 'test_acc_w': 0.9585757255554199,
 'test_dist_l1': 0.05222538113594055,
 'test_dist_l2': 0.06697860360145569,
 'test_dist_logl2': 0.018733210861682892,
 'test_dist_mistake_severity': 0.16447745263576508,
 'test_iou': 0.9167538285255432}
```

with loss weighting

python3 fusion-test.py  --bs 1 --fusion custom --dataset freiburgthermal --modalities rgb,ir --save --bs 1 --save_xp mishmash --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-03 22-03-freiburgthermal-c3-sord-1,2,3-a1-logl2-lw-rgb,ir-epoch=41-val_loss=0.0016.ckpt" --loss_weight
```bash
[INFO] CM IoU - tensor([97.0061, 59.2552, 90.0164])
[INFO] precision tensor([97.9392, 76.9238, 95.0633], dtype=torch.float64) (89.97539493065374) | recall tensor([99.0275, 72.0655, 94.4307], dtype=torch.float64) (88.50790058537645)
Testing: 100%|████████████████████████████████████████████████| 1115/1115 [29:05<00:00,  1.57s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9557244777679443,
 'test_acc_w': 0.9596750736236572,
 'test_dist_l1': 0.05145256221294403,
 'test_dist_l2': 0.06580664962530136,
 'test_dist_logl2': 0.018410688266158104,
 'test_dist_mistake_severity': 0.16209959983825684,
 'test_iou': 0.9177820682525635}
```

RGB encoder trained on obj classes from freiburg thermal
IR encoder trained on obj classes from freiburg thermal
then put into fusion architecture, with output layer = 3 to learn driveability

python3 fusion-test.py  --bs 1 --fusion custom --dataset freiburgthermal --modalities rgb,ir --save --bs 1 --save_xp mishmash --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-07 11-08-freiburgthermal-c3-sord-1,2,3-a1-logl2-rgb,ir-epoch=25-val_loss=0.0036.ckpt" --loss_weight
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

python3 fusion-test.py  --bs 1 --fusion custom --dataset thermalvoc --modalities rgb,ir --save --bs 1 --save_xp mishmash --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-03 11-30-freiburgthermal-c3-sord-1,2,3-a1-logl2-rgb,ir-epoch=39-val_loss=0.0038.ckpt" --loss_weight
```bash
[INFO] CM IoU - tensor([96.1628,  0.0000,  0.0000])
/usr/local/lib/python3.8/dist-packages/torch/tensor.py:521: RuntimeWarning: invalid value encountered in multiply
  return self.to(torch.get_default_dtype()).reciprocal() * other
[INFO] precision tensor([100.0000,   0.0000,   0.0000], dtype=torch.float64) (33.333333298420875) | recall tensor([96.1628,     nan,     nan], dtype=torch.float64) (nan)
Testing: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1659/1659 [43:49<00:00,  1.58s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9616280794143677,
 'test_acc_w': 0.9513047337532043,
 'test_dist_l1': 0.046021562069654465,
 'test_dist_l2': 0.06132075935602188,
 'test_dist_logl2': 0.02399332821369171,
 'test_dist_mistake_severity': 0.19935385882854462,
 'test_iou': 0.9399399161338806}
```

python3 fusion-test.py  --bs 1 --fusion custom --dataset thermalvoc --modalities rgb,ir --save --bs 1 --save_xp mishmash --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-03 22-03-freiburgthermal-c3-sord-1,2,3-a1-logl2-lw-rgb,ir-epoch=41-val_loss=0.0016.ckpt" --loss_weight

```bash
[INFO] CM IoU - tensor([96.0871,  0.0000,  0.0000])
/usr/local/lib/python3.8/dist-packages/torch/tensor.py:521: RuntimeWarning: invalid value encountered in multiply
  return self.to(torch.get_default_dtype()).reciprocal() * other
[INFO] precision tensor([100.0000,   0.0000,   0.0000], dtype=torch.float64) (33.33333320796612) | recall tensor([96.0871,     nan,     nan], dtype=torch.float64) (nan)
Testing: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1659/1659 [43:33<00:00,  1.58s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9608707427978516,
 'test_acc_w': 0.9478398561477661,
 'test_dist_l1': 0.04797082021832466,
 'test_dist_l2': 0.06565381586551666,
 'test_dist_logl2': 0.02522311359643936,
 'test_dist_mistake_severity': 0.22595582902431488,
 'test_iou': 0.9385401606559753}
```

python3 fusion-test.py  --bs 1 --fusion custom --dataset thermalvoc --modalities rgb,ir --save --bs 1 --save_xp mishmash --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-07 11-08-freiburgthermal-c3-sord-1,2,3-a1-logl2-rgb,ir-epoch=25-val_loss=0.0036.ckpt" --loss_weight
