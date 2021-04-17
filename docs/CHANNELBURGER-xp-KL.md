# Freiburg

## Baseline (RGB)

2021-04-08 13-31-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474.ckpt
 ```bash

 ```

## Stacking

python3 lightning.py --gpus 0 --num_classes 3 --bs 1 --mode affordances --dataset freiburg --loss kl --debug --loss_weight --test_checkpoint "lightning_logs/2021-04-16 21-07-freiburg-c6-kl-rgb,depth-epoch=87-val_loss=0.1475.ckpt"

2021-04-17 18-57-freiburg-c6-kl-rgb,depth-epoch=87-val_loss=0.1472.ckpt
```bash
[INFO] CM IoU - tensor([94.0667, 81.2238, 80.7965])
[INFO] precision tensor([95.8536, 90.0039, 97.2533], dtype=torch.float64) (94.37023134269111) | recall tensor([98.0568, 89.2776, 82.6833], dtype=torch.float64) (90.00589052239056)
Testing: 100%|███████████████████████████████████████████████████████████████| 136/136 [03:46<00:00,  1.67s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9440577626228333,
 'test_acc_w': 0.9383087754249573,
 'test_dist_l1': 0.056791193783283234,
 'test_dist_l2': 0.05848913639783859,
 'test_dist_logl2': 0.022480720654129982,
 'test_dist_mistake_severity': 0.015175855718553066,
 'test_iou': 0.8961920142173767}
```

2021-04-17 19-40-freiburg-c6-kl-rgb,ir-epoch=149-val_loss=0.1349.ckpt
```bash
[INFO] CM IoU - tensor([94.8675, 83.3967, 81.9301])
[INFO] precision tensor([96.5262, 90.9891, 96.8485], dtype=torch.float64) (94.78792707432575) | recall tensor([98.2209, 90.9045, 84.1742], dtype=torch.float64) (91.09990481659031)
Testing: 100%|███████████████████████████████████████████████████████████████| 136/136 [03:45<00:00,  1.65s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9507324695587158,
 'test_acc_w': 0.9523745775222778,
 'test_dist_l1': 0.050222184509038925,
 'test_dist_l2': 0.052131522446870804,
 'test_dist_logl2': 0.019676214084029198,
 'test_dist_mistake_severity': 0.019377263262867928,
 'test_iou': 0.9079180955886841}
```


```bash

```



```bash

```
