# Freiburg

## SSMA

python3 fusion-test.py  --bs 1 --fusion ssma --dataset freiburg --modalities rgb,depth --save --bs 1 --save_xp fusion --decoders single --test_checkpoint "lightning_logs/fusionfusion-ssma16-single-2021-04-20 15-20-freiburg-c3-kl-rgb,depth-epoch=83-val_loss=0.1917.ckpt" --loss_weight

```bash
[INFO] CM IoU - tensor([93.2676, 77.7040, 70.9775])
[INFO] precision tensor([94.3460, 88.7383, 98.3211], dtype=torch.float64) (93.80180696726748) | recall tensor([98.7893, 86.2049, 71.8483], dtype=torch.float64) (85.6141741852646)
Testing: 100%|██████████████████████████████████████████████████████████████████████████████████████| 136/136 [04:44<00:00,  2.09s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9313876628875732,
 'test_acc_w': 0.9230747222900391,
 'test_dist_l1': 0.07119351625442505,
 'test_dist_l2': 0.07635588943958282,
 'test_dist_logl2': 0.02775973454117775,
 'test_dist_mistake_severity': 0.037619899958372116,
 'test_iou': 0.8745157122612}
```

python3 fusion-test.py  --bs 1 --fusion ssma --dataset freiburg --modalities rgb,depth --save --bs 1 --save_xp fusion --decoders multi --test_checkpoint "lightning_logs/fusionfusion-ssma16-multi-2021-04-20 16-36-freiburg-c3-kl-rgb,depth-epoch=29-val_loss=0.1780.ckpt" --loss_weight

```bash
[INFO] CM IoU - tensor([92.5521, 76.4705, 77.5039])
[INFO] precision tensor([93.9237, 90.0253, 94.5016], dtype=torch.float64) (92.81686166812597) | recall tensor([98.4467, 83.5495, 81.1640], dtype=torch.float64) (87.72004312228674)
Testing: 100%|██████████████████████████████████████████████████████████████████████████████████████| 136/136 [05:31<00:00,  2.44s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9300007820129395,
 'test_acc_w': 0.9278292655944824,
 'test_dist_l1': 0.0713713988661766,
 'test_dist_l2': 0.07411573082208633,
 'test_dist_logl2': 0.028729936107993126,
 'test_dist_mistake_severity': 0.019602587446570396,
 'test_iou': 0.8710793852806091}
```

## Custom

python3 fusion-test.py  --bs 1 --fusion custom --dataset freiburg --modalities rgb,depth --save --bs 1 --save_xp fusion --decoders single --test_checkpoint "lightning_logs/fusionfusion-custom16-single-2021-04-20 18-31-freiburg-c3-kl-rgb,depth-epoch=43-val_loss=0.1429.ckpt" --loss_weight

```bash
[INFO] CM IoU - tensor([94.3308, 81.6555, 80.6822])
[INFO] precision tensor([96.0335, 90.1197, 97.3849], dtype=torch.float64) (94.51272763804161) | recall tensor([98.1550, 89.6843, 82.4689], dtype=torch.float64) (90.10276913391309)
Testing: 100%|██████████████████████████████████████████████████████████████████████████████████████| 136/136 [05:01<00:00,  2.22s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9455978274345398,
 'test_acc_w': 0.9393054246902466,
 'test_dist_l1': 0.05501844733953476,
 'test_dist_l2': 0.056250955909490585,
 'test_dist_logl2': 0.02147301286458969,
 'test_dist_mistake_severity': 0.01132777240127325,
 'test_iou': 0.8989060521125793}
```

python3 fusion-test.py  --bs 1 --fusion custom --dataset freiburg --modalities rgb,depth --save --bs 1 --save_xp fusion --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16-multi-2021-04-20 19-24-freiburg-c3-kl-rgb,depth-epoch=141-val_loss=0.1369.ckpt" --loss_weight
```bash
[INFO] CM IoU - tensor([94.7388, 82.9838, 81.4231])
[INFO] precision tensor([96.7505, 89.8015, 97.6981], dtype=torch.float64) (94.75003337848898) | recall tensor([97.8524, 91.6181, 83.0157], dtype=torch.float64) (90.82873263703965)
Testing: 100%|█████████████████████████████████████████████████████████████████| 136/136 [03:15<00:00,  1.44s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9492639899253845,
 'test_acc_w': 0.9446205496788025,
 'test_dist_l1': 0.05132059380412102,
 'test_dist_l2': 0.052489787340164185,
 'test_dist_logl2': 0.019895028322935104,
 'test_dist_mistake_severity': 0.011522334069013596,
 'test_iou': 0.9053769707679749}
```

# Cityscapes

## V + Draw

### SSMA

python3 fusion-test.py  --bs 1 --fusion ssma --dataset cityscapes --modalities rgb,depthraw --save --bs 1 --save_xp fusion --decoders multi --test_checkpoint "lightning_logs/fusionfusion-ssma16-multi-2021-04-20 21-01-cityscapes-c3-kl-rgb,depthraw-epoch=23-val_loss=0.0875.ckpt" --loss_weight
```bash
[INFO] CM IoU - tensor([98.2037, 70.0274, 94.4774])
[INFO] precision tensor([98.8912, 77.8948, 98.5673], dtype=torch.float64) (91.78443923244592) | recall tensor([99.2970, 87.3950, 95.7929], dtype=torch.float64) (94.16163919241663)
Testing: 100%|█████████████████████████████████████████████████████████████████| 233/233 [06:34<00:00,  1.69s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9722700715065002,
 'test_acc_w': 0.9760338664054871,
 'test_dist_l1': 0.030870771035552025,
 'test_dist_l2': 0.037152476608753204,
 'test_dist_logl2': 0.010094355791807175,
 'test_dist_mistake_severity': 0.11326582729816437,
 'test_iou': 0.9476955533027649}
```

python3 fusion-test.py  --bs 1 --fusion ssma --dataset cityscapes --modalities rgb,depthraw --save --bs 1 --save_xp fusion --decoders single --test_checkpoint "lightning_logs/fusionfusion-ssma16-single-2021-04-20 22-45-cityscapes-c3-kl-rgb,depthraw-epoch=11-val_loss=0.0910.ckpt" --loss_weight
```bash
[INFO] CM IoU - tensor([98.0832, 67.8026, 93.6397])
[INFO] precision tensor([98.6050, 75.0245, 98.9060], dtype=torch.float64) (90.84515053435638) | recall tensor([99.4634, 87.5678, 94.6197], dtype=torch.float64) (93.88364056002749)
Testing: 100%|█████████████████████████████████████████████████████████████████| 233/233 [06:12<00:00,  1.60s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9690155386924744,
 'test_acc_w': 0.971576988697052,
 'test_dist_l1': 0.03463395684957504,
 'test_dist_l2': 0.04193303734064102,
 'test_dist_logl2': 0.011227413080632687,
 'test_dist_mistake_severity': 0.11778627336025238,
 'test_iou': 0.9422069191932678}
```

### Custom

python3 fusion-test.py  --bs 1 --fusion custom --dataset cityscapes --modalities rgb,depthraw --save --bs 1 --save_xp fusion --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16-multi-2021-04-21 00-22-cityscapes-c3-kl-rgb,depthraw-epoch=23-val_loss=0.0873.ckpt" --loss_weight
```bash
[INFO] CM IoU - tensor([98.2418, 68.4958, 93.8710])
[INFO] precision tensor([98.9885, 75.3589, 98.5700], dtype=torch.float64) (90.97247962948174) | recall tensor([99.2380, 88.2644, 95.1670], dtype=torch.float64) (94.2231498945612)
Testing: 100%|█████████████████████████████████████████████████████████████████| 233/233 [06:23<00:00,  1.64s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9702072739601135,
 'test_acc_w': 0.9722357988357544,
 'test_dist_l1': 0.0328991636633873,
 'test_dist_l2': 0.03911207616329193,
 'test_dist_logl2': 0.010336345992982388,
 'test_dist_mistake_severity': 0.10426907986402512,
 'test_iou': 0.9445460438728333}
```

python3 fusion-test.py  --bs 1 --fusion custom --dataset cityscapes --modalities rgb,depthraw --save --bs 1 --save_xp fusion --decoders single --test_checkpoint "lightning_logs/fusionfusion-custom16-single-2021-04-21 07-16-cityscapes-c3-kl-rgb,depthraw-epoch=2-val_loss=0.0920.ckpt" --loss_weight
```bash
[INFO] CM IoU - tensor([98.0910, 70.2256, 94.5367])
[INFO] precision tensor([98.8265, 78.6386, 98.4418], dtype=torch.float64) (91.96899800930018) | recall tensor([99.2469, 86.7798, 95.9728], dtype=torch.float64) (93.99984525112663)
Testing: 100%|█████████████████████████████████████████████████████████████████| 233/233 [06:18<00:00,  1.63s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9722428321838379,
 'test_acc_w': 0.9770115613937378,
 'test_dist_l1': 0.03132832050323486,
 'test_dist_l2': 0.03847062960267067,
 'test_dist_logl2': 0.010617847554385662,
 'test_dist_mistake_severity': 0.12865705788135529,
 'test_iou': 0.9475404024124146}
```

## V + Dcomp

### SSMA

python3 fusion-test.py  --bs 1 --fusion ssma --dataset cityscapes --modalities rgb,depth --save --bs 1 --save_xp fusion --decoders multi --test_checkpoint "lightning_logs/fusionfusion-ssma16-multi-2021-04-21 08-54-cityscapes-c3-kl-rgb,depth-epoch=15-val_loss=0.0889.ckpt" --loss_weight
```bash
[INFO] CM IoU - tensor([98.0225, 71.4644, 94.8282])
[INFO] precision tensor([98.6587, 80.7297, 98.4806], dtype=torch.float64) (92.6230179582395) | recall tensor([99.3465, 86.1625, 96.2362], dtype=torch.float64) (93.91504455554318)
Testing: 100%|█████████████████████████████████████████████████████████████████| 233/233 [06:25<00:00,  1.66s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9733688831329346,
 'test_acc_w': 0.9787277579307556,
 'test_dist_l1': 0.030646201223134995,
 'test_dist_l2': 0.03867644444108009,
 'test_dist_logl2': 0.01088542677462101,
 'test_dist_mistake_severity': 0.1507682353258133,
 'test_iou': 0.9495238065719604}
```

python3 fusion-test.py  --bs 1 --fusion ssma --dataset cityscapes --modalities rgb,depth --save --bs 1 --save_xp fusion --decoders single --test_checkpoint "lightning_logs/fusionfusion-ssma16-single-2021-04-21 10-37-cityscapes-c3-kl-rgb,depth-epoch=11-val_loss=0.0894.ckpt" --loss_weight
```bash
[INFO] CM IoU - tensor([97.9988, 67.7790, 93.6209])
[INFO] precision tensor([98.5011, 75.4689, 98.8534], dtype=torch.float64) (90.94115081022119) | recall tensor([99.4823, 86.9311, 94.6487], dtype=torch.float64) (93.68738428837182)
Testing: 100%|█████████████████████████████████████████████████████████████████| 233/233 [06:31<00:00,  1.68s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9688108563423157,
 'test_acc_w': 0.9710447788238525,
 'test_dist_l1': 0.035212695598602295,
 'test_dist_l2': 0.043259840458631516,
 'test_dist_logl2': 0.011689901351928711,
 'test_dist_mistake_severity': 0.1290055811405182,
 'test_iou': 0.9418937563896179}
```

## Custom

python3 fusion-test.py  --bs 1 --fusion custom --dataset cityscapes --modalities rgb,depth --save --bs 1 --save_xp fusion --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16-multi-2021-04-21 14-31-cityscapes-c3-kl-rgb,depth-epoch=5-val_loss=0.0876.ckpt" --loss_weight
```bash
[INFO] CM IoU - tensor([98.1930, 68.3725, 93.7500])
[INFO] precision tensor([98.9313, 75.0252, 98.6373], dtype=torch.float64) (90.8646233259951) | recall tensor([99.2457, 88.5198, 94.9801], dtype=torch.float64) (94.2485333117285)
Testing: 100%|█████████████████████████████████████████████████████████████████| 233/233 [06:07<00:00,  1.58s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9697316884994507,
 'test_acc_w': 0.9717258810997009,
 'test_dist_l1': 0.033619899302721024,
 'test_dist_l2': 0.04032310098409653,
 'test_dist_logl2': 0.010682874359190464,
 'test_dist_mistake_severity': 0.11072973161935806,
 'test_iou': 0.9436619877815247}
```

python3 fusion-test.py  --bs 1 --fusion custom --dataset cityscapes --modalities rgb,depth --save --bs 1 --save_xp fusion --decoders single --test_checkpoint "lightning_logs/fusionfusion-custom16-single-2021-04-21 16-17-cityscapes-c3-kl-rgb,depth-epoch=17-val_loss=0.0901.ckpt" --loss_weight
```bash
[INFO] CM IoU - tensor([98.1752, 69.3024, 94.2064])
[INFO] precision tensor([98.7852, 76.9381, 98.7087], dtype=torch.float64) (91.47733158909452) | recall tensor([99.3749, 87.4733, 95.3818], dtype=torch.float64) (94.0767082411734)
Testing: 100%|█████████████████████████████████████████████████████████████████| 233/233 [06:32<00:00,  1.68s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9712527990341187,
 'test_acc_w': 0.9751651287078857,
 'test_dist_l1': 0.03202430158853531,
 'test_dist_l2': 0.038578473031520844,
 'test_dist_logl2': 0.010415713302791119,
 'test_dist_mistake_severity': 0.11399666219949722,
 'test_iou': 0.9459006786346436}
```
