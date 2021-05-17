## Freiburg

### Hot labels

NO WEIGHT 2021-04-08 13-31-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474.ckpt
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

WEIGHT 2021-04-07 11-31-freiburg-c6-kl-0,1,2-rgb-epoch=43-val_loss=0.0771.ckpt
```bash
[INFO] CM IoU - tensor([93.9054, 80.7990, 80.8461])
[INFO] precision tensor([95.4123, 90.5784, 97.4412], dtype=torch.float64) (94.47727137113824) | recall tensor([98.3460, 88.2128, 82.5997], dtype=torch.float64) (89.71948606564362)
Testing: 100%|███████████████████████████████████████████████████████████████████████████████████| 136/136 [05:11<00:00,  2.29s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9430180788040161,
 'test_acc_w': 0.9379153847694397,
 'test_dist_l1': 0.05800328031182289,
 'test_dist_l2': 0.06004602089524269,
 'test_dist_logl2': 0.023182164877653122,
 'test_dist_mistake_severity': 0.01792445033788681,
 'test_iou': 0.8942697048187256}
```

### SORD

NO WEIGHT 2021-04-08 13-59-freiburg-c6-sord-1,2,3-a1-l2-rgb-epoch=66-val_loss=0.0278.ckpt
```bash
# Train set
 'test_acc': 0.9601759314537048,
 'test_acc_w': 0.9604872465133667,
 'test_dist_l1': 0.040252719074487686,
 'test_dist_l2': 0.04111005365848541,
 'test_dist_logl2': 0.01525642815977335,
 'test_dist_mistake_severity': 0.010764060541987419,
 'test_iou': 0.924487829208374
# Test set
 'test_acc': 0.9447071552276611,
 'test_acc_w': 0.939323902130127,
 'test_dist_l1': 0.0558588020503521,
 'test_dist_l2': 0.056990720331668854,
 'test_dist_logl2': 0.022109368816018105,
 'test_dist_mistake_severity': 0.010235673747956753,
 'test_iou': 0.8972585201263428
```

## Cityscapes


NO WEIGHT 2021-04-09 03-40-cityscapes-c30-kl-rgb-epoch=18-val_loss=0.0918.ckpt
```bash
INFO] CM IoU - tensor([98.0204, 67.9424, 93.7633])
[INFO] precision tensor([98.8983, 75.2056, 98.4099], dtype=torch.float64) (90.83792570294355) | recall tensor([99.1025, 87.5544, 95.2057], dtype=torch.float64) (93.95421391434033)
Testing: 100%|███████████████████████████████████████████████████████████████████████████████████| 233/233 [09:54<00:00,  2.55s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9691147208213806,
 'test_acc_w': 0.9723863005638123,
 'test_dist_l1': 0.03461460769176483,
 'test_dist_l2': 0.042073216289281845,
 'test_dist_logl2': 0.011372342705726624,
 'test_dist_mistake_severity': 0.1207469254732132,
 'test_iou': 0.942446231842041}
```

python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset cityscapes --test_checkpoint "lightning_logs/2021-04-09 10-00-cityscapes-c30-kl-rgb-epoch=6-val_loss=0.0283.ckpt" --save --save_xp lw --modalities rgb --gpus 0
WEIGHT
```bash
[INFO] CM IoU - tensor([97.9321, 66.8624, 93.5493])
[INFO] precision tensor([98.7867, 74.6277, 98.3586], dtype=torch.float64) (90.59098521932918) | recall tensor([99.1244, 86.5334, 95.0329], dtype=torch.float64) (93.56355974125469)
Testing: 100%|███████████████████████████████████████████████████████████████████████████████████| 233/233 [09:15<00:00,  2.39s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.967932403087616,
 'test_acc_w': 0.971306324005127,
 'test_dist_l1': 0.03594338521361351,
 'test_dist_l2': 0.043695077300071716,
 'test_dist_logl2': 0.011837306432425976,
 'test_dist_mistake_severity': 0.12086509168148041,
 'test_iou': 0.9402391910552979}
```

python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset cityscapes --test_checkpoint "lightning_logs/2021-05-09 10-53-cityscapes-c3-kl-lw-rgb-epoch=23-val_loss=0.0284.ckpt" --save --save_xp lw --modalities rgb --gpus 0
```bash
[INFO] CM IoU - tensor([98.0286, 66.5581, 93.3087])
[INFO] precision tensor([98.8716, 73.4529, 98.4820], dtype=torch.float64) (90.26883589461838) | recall tensor([99.1378, 87.6400, 94.6703], dtype=torch.float64) (93.81604108830288)
Testing: 100%|███████████████████████████████████████████████████████████| 233/233 [03:54<00:00,  1.01s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9674034118652344,
 'test_acc_w': 0.9674034118652344,
 'test_dist_l1': 0.036246929317712784,
 'test_dist_l2': 0.04354766756296158,
 'test_dist_logl2': 0.01158237550407648,
 'test_dist_mistake_severity': 0.11198629438877106,
 'test_iou': 0.9397504329681396}
```
