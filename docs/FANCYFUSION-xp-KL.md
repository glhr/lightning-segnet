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

python3 fusion-test.py  --bs 1 --fusion custom --dataset freiburg --modalities rgb,depth --save --bs 1 --save_xp fusion --decoders multi --test_checkpoint "lightning_logsfusionfusion-custom16-multi-2021-04-20 19-24-freiburg-c3-kl-rgb,depth-epoch=141-val_loss=0.1369.ckpt" --loss_weight
```bash
```
