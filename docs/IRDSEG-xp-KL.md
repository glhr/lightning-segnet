# Freiburg

## Baseline (RGB)

2021-04-08 13-31-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474.ckpt
 ```bash
 [INFO] CM IoU - tensor([93.8467, 80.6959, 81.4506])
 [INFO] precision tensor([95.2272, 91.0170, 97.3936], dtype=torch.float64) (94.54591993392611) | recall tensor([98.4788, 87.6790, 83.2656], dtype=torch.float64) (89.80781070313739)
 Testing: 100%|███████████████████████████████████████████████████████████████████████████████████| 136/136 [05:09<00:00,  2.28s/it]
 --------------------------------------------------------------------------------
 DATALOADER:0 TEST RESULTS
 {'cm': 0.0,
  'test_acc': 0.9430200457572937,
  'test_acc_w': 0.9367597103118896,
  'test_dist_l1': 0.05796791985630989,
  'test_dist_l2': 0.059943895787000656,
  'test_dist_logl2': 0.02330946922302246,
  'test_dist_mistake_severity': 0.017339220270514488,
  'test_iou': 0.894274115562439}
 ```

## IR

### geom_transform

2021-04-16 22-39-freiburg-c6-kl-ir-epoch=22-val_loss=0.1947.ckpt
```bash
[INFO] CM IoU - tensor([93.9832, 77.3056, 61.0496])
[INFO] precision tensor([95.5579, 85.5379, 95.4517], dtype=torch.float64) (92.18252076035469) | recall tensor([98.2769, 88.9289, 62.8788], dtype=torch.float64) (83.3615010620207)
Testing: 100%|██████████████████████████████████████████████████████████████████████| 136/136 [03:51<00:00,  1.70s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9277106523513794,
 'test_acc_w': 0.9210140109062195,
 'test_dist_l1': 0.07488689571619034,
 'test_dist_l2': 0.08008195459842682,
 'test_dist_logl2': 0.02665707655251026,
 'test_dist_mistake_severity': 0.03593237325549126,
 'test_iou': 0.868135929107666}
```

### geom_photo_transform

2021-04-17 13-16-freiburg-c6-kl-ir-epoch=128-val_loss=0.1708.ckpt

```bash
[INFO] CM IoU - tensor([94.5588, 78.4743, 62.8379])
[INFO] precision tensor([96.0090, 85.9882, 96.8560], dtype=torch.float64) (92.95103680309418) | recall tensor([98.4278, 89.9805, 64.1464], dtype=torch.float64) (84.18488305503828)
Testing: 100%|██████████████████████████████████████████████████████████████████████| 136/136 [03:39<00:00,  1.61s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9325739145278931,
 'test_acc_w': 0.9258043169975281,
 'test_dist_l1': 0.0689634382724762,
 'test_dist_l2': 0.07203814387321472,
 'test_dist_logl2': 0.02380155585706234,
 'test_dist_mistake_severity': 0.022800549864768982,
 'test_iou': 0.8762935400009155}
```

## Depth

### geom_transform

2021-04-17 14-18-freiburg-c6-kl-depth-epoch=149-val_loss=0.3106.ckpt
```bash
[INFO] CM IoU - tensor([91.2048, 66.4805, 33.4057])
[INFO] precision tensor([93.5841, 76.0219, 93.8194], dtype=torch.float64) (87.80848158468473) | recall tensor([97.2880, 84.1190, 34.1574], dtype=torch.float64) (71.85482238405734)
Testing: 100%|██████████████████████████████████████████████████████████████████████| 136/136 [04:05<00:00,  1.81s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.8840377330780029,
 'test_acc_w': 0.8500466346740723,
 'test_dist_l1': 0.11870251595973969,
 'test_dist_l2': 0.12418300658464432,
 'test_dist_logl2': 0.04026950150728226,
 'test_dist_mistake_severity': 0.023630507290363312,
 'test_iou': 0.7956401109695435}

```

### geom_photo_transform

2021-04-17 12-17-freiburg-c6-kl-depth-epoch=149-val_loss=0.3213.ckpt
```bash
[INFO] CM IoU - tensor([91.2113, 65.1968, 27.3008])
[INFO] precision tensor([93.0690, 75.3414, 94.4732], dtype=torch.float64) (87.62787329754289) | recall tensor([97.8585, 82.8826, 27.7439], dtype=torch.float64) (69.49501067586135)
Testing: 100%|██████████████████████████████████████████████████████████████████████| 136/136 [03:49<00:00,  1.69s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.8789797425270081,
 'test_acc_w': 0.84084153175354,
 'test_dist_l1': 0.12392999231815338,
 'test_dist_l2': 0.12974941730499268,
 'test_dist_logl2': 0.04132106900215149,
 'test_dist_mistake_severity': 0.02404315583407879,
 'test_iou': 0.7878786325454712}
```
