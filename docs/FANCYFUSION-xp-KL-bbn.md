# Freiburg

## V + D

### SSMA

python3 fusion-test.py  --bs 1 --fusion ssma --dataset freiburg --modalities rgb,depth --save --bs 1 --save_xp fusion-rgb,d --decoders single --test_checkpoint "lightning_logs/fusionfusion-ssma16-single-2021-04-26 06-40-freiburg-c3-kl-rgb,depth-epoch=75-val_loss=0.1338.ckpt" --loss_weight
```bash
[INFO] CM IoU - tensor([94.5639, 82.5814, 82.1213])
[INFO] precision tensor([95.8063, 91.9477, 96.6818], dtype=torch.float64) (94.81193395523326) | recall tensor([98.6471, 89.0193, 84.5031], dtype=torch.float64) (90.72317911016725)
Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 136/136 [02:55<00:00,  1.29s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9487413763999939,
 'test_acc_w': 0.9447363615036011,
 'test_dist_l1': 0.05239295959472656,
 'test_dist_l2': 0.05466165021061897,
 'test_dist_logl2': 0.02086523175239563,
 'test_dist_mistake_severity': 0.022129828110337257,
 'test_iou': 0.9044176936149597}

```

python3 fusion-test.py  --bs 1 --fusion ssma --dataset freiburg --modalities rgb,depth --save --bs 1 --save_xp fusion-rgb,d --decoders multi --test_checkpoint "lightning_logs/fusionfusion-ssma16-multi-2021-04-26 07-35-freiburg-c3-kl-rgb,depth-epoch=102-val_loss=0.1500.ckpt" --loss_weight
```bash
[INFO] CM IoU - tensor([94.8032, 81.9579, 74.4610])
[INFO] precision tensor([97.4417, 86.8016, 98.0871], dtype=torch.float64) (94.1101421206215) | recall tensor([97.2232, 93.6254, 75.5582], dtype=torch.float64) (88.80225076644615)
Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 136/136 [02:48<00:00,  1.24s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9442029595375061,
 'test_acc_w': 0.9388664960861206,
 'test_dist_l1': 0.056573860347270966,
 'test_dist_l2': 0.05812755227088928,
 'test_dist_logl2': 0.020652301609516144,
 'test_dist_mistake_severity': 0.01392271462827921,
 'test_iou': 0.8963174223899841}
```


## V + NIR

### SSMA

python3 fusion-test.py  --bs 1 --fusion ssma --dataset freiburg --modalities rgb,ir --save --bs 1 --save_xp fusion-rgb,ir --decoders single --test_checkpoint "lightning_logs/fusionfusion-ssma16-single-2021-04-26 09-56-freiburg-c3-kl-rgb,ir-epoch=79-val_loss=0.1066.ckpt" --loss_weight
```bash
[INFO] CM IoU - tensor([95.8745, 86.6787, 86.4235])
[INFO] precision tensor([97.4460, 92.7421, 96.6727], dtype=torch.float64) (95.6202870414908) | recall tensor([98.3457, 92.9863, 89.0730], dtype=torch.float64) (93.46833175367237)
Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 136/136 [02:55<00:00,  1.29s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9612641334533691,
 'test_acc_w': 0.9630003571510315,
 'test_dist_l1': 0.03932240605354309,
 'test_dist_l2': 0.040495555847883224,
 'test_dist_logl2': 0.015461365692317486,
 'test_dist_mistake_severity': 0.015142977237701416,
 'test_iou': 0.9268472790718079}
```

python3 fusion-test.py  --bs 1 --fusion ssma --dataset freiburg --modalities rgb,ir --save --bs 1 --save_xp fusion-rgb,ir --decoders multi --test_checkpoint "lightning_logs/fusionfusion-ssma16-multi-2021-04-26 08-44-freiburg-c3-kl-rgb,ir-epoch=80-val_loss=0.1199.ckpt" --loss_weight
```bash
[INFO] CM IoU - tensor([95.0592, 84.4485, 83.6180])
[INFO] precision tensor([96.4897, 92.7008, 95.0489], dtype=torch.float64) (94.74645325727002) | recall tensor([98.4644, 90.4639, 87.4259], dtype=torch.float64) (92.11806233919594)
Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 136/136 [03:44<00:00,  1.65s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9538992047309875,
 'test_acc_w': 0.9557709693908691,
 'test_dist_l1': 0.04772939532995224,
 'test_dist_l2': 0.050986647605895996,
 'test_dist_logl2': 0.019243542104959488,
 'test_dist_mistake_severity': 0.03532750904560089,
 'test_iou': 0.9133612513542175}
```


## V + D + IR

### SSMA

python3 fusion-test.py --bs 1 --fusion ssma --dataset freiburg --modalities rgb,depth,ir --save --bs 1 --save_xp fusion-rgb,d,ir --decoders single --test_checkpoint "lightning_logs/fusionfusion-ssma16-single-2021-04-25 22-30-freiburg-c3-kl-rgb,depth,ir-epoch=129-val_loss=0.1145.ckpt" --loss_weight
```bash
[INFO] CM IoU - tensor([95.3587, 84.9844, 85.1340])
[INFO] precision tensor([96.5571, 92.9543, 97.0092], dtype=torch.float64) (95.50687559970447) | recall tensor([98.7152, 90.8356, 87.4287], dtype=torch.float64) (92.32649032669093)
Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 136/136 [03:04<00:00,  1.36s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9565188884735107,
 'test_acc_w': 0.9547152519226074,
 'test_dist_l1': 0.04411790147423744,
 'test_dist_l2': 0.045391518622636795,
 'test_dist_logl2': 0.017451180145144463,
 'test_dist_mistake_severity': 0.014645632356405258,
 'test_iou': 0.9182520508766174}
```

python3 fusion-test.py  --bs 1 --fusion ssma --dataset freiburg --modalities rgb,depth,ir --save --bs 1 --save_xp fusion-rgb,d,ir --decoders multi --test_checkpoint "lightning_logs/fusionfusion-ssma16-multi-2021-04-25 20-49-freiburg-c3-kl-rgb,depth,ir-epoch=125-val_loss=0.1195.ckpt" --loss_weight
```bash
[INFO] CM IoU - tensor([95.5162, 85.0616, 83.1088])
[INFO] precision tensor([96.7636, 92.1231, 97.9637], dtype=torch.float64) (95.61677675165754) | recall tensor([98.6684, 91.7335, 84.5698], dtype=torch.float64) (91.65723435755999)
Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 136/136 [03:02<00:00,  1.34s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9561812877655029,
 'test_acc_w': 0.9546265006065369,
 'test_dist_l1': 0.044631268829107285,
 'test_dist_l2': 0.04625644534826279,
 'test_dist_logl2': 0.017280209809541702,
 'test_dist_mistake_severity': 0.018544360995292664,
 'test_iou': 0.9179295897483826}
```


# Cityscapes


## V + Draw

### SSMA

python3 fusion-test.py  --bs 1 --fusion ssma --dataset cityscapes --modalities rgb,depthraw --save --bs 1 --save_xp fusion-rgb,draw --decoders multi --test_checkpoint "lightning_logs/fusionfusion-ssma16-multi-2021-05-01 15-44-cityscapes-c3-kl-rgb,depthraw-epoch=24-val_loss=0.0876.ckpt" --loss_weight
```bash
[INFO] CM IoU - tensor([98.2089, 66.0249, 92.9753])
[INFO] precision tensor([98.8923, 72.3197, 98.6848], dtype=torch.float64) (89.96561346700412) | recall tensor([99.3012, 88.3524, 94.1418], dtype=torch.float64) (93.93178858544596)
Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 233/233 [05:26<00:00,  1.40s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9668559432029724,
 'test_acc_w': 0.9668952822685242,
 'test_dist_l1': 0.03640196472406387,
 'test_dist_l2': 0.042917754501104355,
 'test_dist_logl2': 0.011060087010264397,
 'test_dist_mistake_severity': 0.09829497337341309,
 'test_iou': 0.9395483732223511}
```

python3 fusion-test.py  --bs 1 --fusion ssma --dataset cityscapes --modalities rgb,depthraw --save --bs 1 --save_xp fusion-rgb,draw --decoders single --test_checkpoint "lightning_logs/fusionfusion-ssma16-single-2021-05-01 09-20-cityscapes-c3-kl-rgb,depthraw-epoch=17-val_loss=0.0901.ckpt" --loss_weight
```bash
[INFO] CM IoU - tensor([98.2149, 68.3403, 93.7995])
[INFO] precision tensor([98.8333, 75.4030, 98.7169], dtype=torch.float64) (90.98441453785972) | recall tensor([99.3669, 87.9462, 94.9572], dtype=torch.float64) (94.09010916960389)
Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 233/233 [05:50<00:00,  1.51s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.969957709312439,
 'test_acc_w': 0.9725385308265686,
 'test_dist_l1': 0.03330245614051819,
 'test_dist_l2': 0.03982274606823921,
 'test_dist_logl2': 0.010542850010097027,
 'test_dist_mistake_severity': 0.10851848870515823,
 'test_iou': 0.9438738226890564}
```

## V + Dcomp

### SSMA

python3 fusion-test.py  --bs 1 --fusion ssma --dataset cityscapes --modalities rgb,depth --save --bs 1 --save_xp fusion-rgb,d --decoders multi --test_checkpoint "lightning_logs/" --loss_weight
```bash

```

python3 fusion-test.py  --bs 1 --fusion ssma --dataset cityscapes --modalities rgb,depth --save --bs 1 --save_xp fusion-rgb,d --decoders single --test_checkpoint "lightning_logs/fusionfusion-ssma16-single-2021-04-26 10-54-cityscapes-c3-kl-rgb,depth-epoch=24-val_loss=0.0884.ckpt" --loss_weight
```bash
INFO] CM IoU - tensor([98.1897, 68.3453, 93.7863])
[INFO] precision tensor([98.7626, 75.1746, 98.8789], dtype=torch.float64) (90.93870253397776) | recall tensor([99.4127, 88.2674, 94.7943], dtype=torch.float64) (94.15811084903018)
Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 233/233 [05:11<00:00,  1.34s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9698296189308167,
 'test_acc_w': 0.9729503989219666,
 'test_dist_l1': 0.03346700221300125,
 'test_dist_l2': 0.040060266852378845,
 'test_dist_logl2': 0.010638359934091568,
 'test_dist_mistake_severity': 0.10926724225282669,
 'test_iou': 0.9436922073364258}
```
