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

2021-04-17 20-25-freiburg-c6-kl-rgb,depth,ir-epoch=81-val_loss=0.1352.ckpt
```bash
[INFO] CM IoU - tensor([94.6175, 82.4653, 81.6896])
[INFO] precision tensor([96.3075, 91.3736, 93.8972], dtype=torch.float64) (93.85944602321032) | recall tensor([98.1791, 89.4276, 86.2700], dtype=torch.float64) (91.2922539374295)
Testing: 100%|███████████████████████████████████████████████████████████████| 136/136 [04:50<00:00,  2.14s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9483013153076172,
 'test_acc_w': 0.9492813348770142,
 'test_dist_l1': 0.05263639986515045,
 'test_dist_l2': 0.05451178178191185,
 'test_dist_logl2': 0.02061903290450573,
 'test_dist_mistake_severity': 0.018137618899345398,
 'test_iou': 0.9034363031387329}
```


2021-04-17 21-32-freiburg-c6-kl-depth,ir-epoch=140-val_loss=0.2201.ckpt
```bash
[INFO] CM IoU - tensor([92.7442, 73.3879, 55.1055])
[INFO] precision tensor([93.7675, 84.5334, 96.8628], dtype=torch.float64) (91.72122186230331) | recall tensor([98.8370, 84.7704, 56.1069], dtype=torch.float64) (79.90474964244285)
Testing: 100%|███████████████████████████████████████████████████████████████| 136/136 [05:06<00:00,  2.25s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9144801497459412,
 'test_acc_w': 0.9010012149810791,
 'test_dist_l1': 0.08897984027862549,
 'test_dist_l2': 0.09589977562427521,
 'test_dist_logl2': 0.03240959346294403,
 'test_dist_mistake_severity': 0.04045804962515831,
 'test_iou': 0.8473340272903442}
```


# Cityscapes

2021-04-18 10-12-cityscapes-c30-kl-rgb,depthraw-epoch=23-val_loss=0.1024.ckpt
```bash
[INFO] CM IoU - tensor([97.6072, 65.2163, 93.3347])
[INFO] precision tensor([98.4425, 74.6435, 98.1889], dtype=torch.float64) (90.42497555006625) | recall tensor([99.1381, 83.7762, 94.9697], dtype=torch.float64) (92.62798504728949)
Testing: 100%|███████████████████████████████████████████████████████████████| 233/233 [07:07<00:00,  1.83s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9659654498100281,
 'test_acc_w': 0.9699937701225281,
 'test_dist_l1': 0.038696739822626114,
 'test_dist_l2': 0.048021167516708374,
 'test_dist_logl2': 0.013336172327399254,
 'test_dist_mistake_severity': 0.13698484003543854,
 'test_iou': 0.936249852180481}
```

2021-04-18 00-48-cityscapes-c30-kl-rgb,depth-epoch=23-val_loss=0.0999.ckpt
```bash
[INFO] CM IoU - tensor([97.6695, 66.3780, 93.5943])
[INFO] precision tensor([98.4600, 76.1403, 98.1543], dtype=torch.float64) (90.91819993332658) | recall tensor([99.1846, 83.8112, 95.2710], dtype=torch.float64) (92.75562262335959)
Testing: 100%|███████████████████████████████████████████████████████████████| 233/233 [07:09<00:00,  1.84s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9673596024513245,
 'test_acc_w': 0.9713712930679321,
 'test_dist_l1': 0.037374597042798996,
 'test_dist_l2': 0.04684307798743248,
 'test_dist_logl2': 0.013045149855315685,
 'test_dist_mistake_severity': 0.1450425535440445,
 'test_iou': 0.9387784600257874}
```
