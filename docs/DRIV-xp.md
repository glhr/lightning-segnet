## Freiburg

### OBJECTS

2021-03-31 08-51-freiburg-c6-kl-rgb-epoch=673-val_loss=0.2363.ckpt

python3 lightning.py --test_checkpoint "lightning_logs/2021-03-31 08-51-freiburg-c6-kl-rgb-epoch=673-val_loss=0.2363.ckpt" --num_classes 6 --bs 16 --mode convert --dataset freiburg --workers 10

FREIBURG test
```bash
[INFO] CM IoU - tensor([94.1782, 81.1333, 78.4960])
[INFO] precision tensor([95.7691, 89.7014, 98.3032], dtype=torch.float64) (94.59123180944621) | recall tensor([98.2667, 89.4670, 79.5741], dtype=torch.float64) (89.10263241334371)
Testing: 100%|███████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:23<00:00,  2.63s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9432798027992249,
 'test_acc_w': 0.9432798624038696,
 'test_dist_l1': 0.057901859283447266,
 'test_dist_l2': 0.06026526540517807,
 'test_dist_logl2': 0.022624187171459198,
 'test_dist_mistake_severity': 0.02083394303917885,
 'test_iou': 0.8928890824317932}
```

KITTI full
```bash
[INFO] CM IoU - tensor([85.3325, 29.8693, 27.5883])
[INFO] precision tensor([88.3378, 36.4094, 89.2799], dtype=torch.float64) (71.34236366287703) | recall tensor([96.1661, 62.4462, 28.5335], dtype=torch.float64) (62.3819198890601)
Testing: 100%|█████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:33<00:00,  2.61s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.7565035223960876,
 'test_acc_w': 0.7565034627914429,
 'test_dist_l1': 0.27697834372520447,
 'test_dist_l2': 0.3439421057701111,
 'test_dist_logl2': 0.096998430788517,
 'test_dist_mistake_severity': 0.13750462234020233,
 'test_iou': 0.608755350112915}
```

CITYSCAPES full
```bash
[INFO] CM IoU - tensor([88.4214, 14.4090, 25.4242])
[INFO] precision tensor([90.1356, 15.7994, 96.9491], dtype=torch.float64) (67.62802488339248) | recall tensor([97.8945, 62.0827, 25.6292], dtype=torch.float64) (61.8687932335607)
Testing: 100%|███████████████████████████████████████████████████████████████████████████████████| 313/313 [09:47<00:00,  1.88s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.6858367323875427,
 'test_acc_w': 0.685836672782898,
 'test_dist_l1': 0.34786534309387207,
 'test_dist_l2': 0.41526931524276733,
 'test_dist_logl2': 0.09866121411323547,
 'test_dist_mistake_severity': 0.10727542638778687,
 'test_iou': 0.3653223514556885}
```

### DRIVEABILITY

2021-04-01 00-16-freiburg-c3-kl-rgb-epoch=686-val_loss=0.1479.ckpt

FREIBURG test
```bash
[INFO] CM IoU - tensor([94.9978, 82.4633, 77.8264])
[INFO] precision tensor([96.3975, 90.0800, 97.6220], dtype=torch.float64) (94.6998226526413) | recall tensor([98.4946, 90.7000, 79.3303], dtype=torch.float64) (89.50829563299627)
Testing: 100%|███████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:23<00:00,  2.63s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9478403329849243,
 'test_acc_w': 0.9478403329849243,
 'test_dist_l1': 0.05282871052622795,
 'test_dist_l2': 0.05416679382324219,
 'test_dist_logl2': 0.019682863727211952,
 'test_dist_mistake_severity': 0.012826787307858467,
 'test_iou': 0.9009587168693542}
```

KITTI full
```bash
[INFO] CM IoU - tensor([84.6296, 26.2260,  6.9008])
[INFO] precision tensor([87.5073, 30.8834, 88.7370], dtype=torch.float64) (69.04255053873194) | recall tensor([96.2595, 63.4911,  6.9618], dtype=torch.float64) (55.570816931777735)
Testing: 100%|█████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:33<00:00,  2.61s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.7085537314414978,
 'test_acc_w': 0.708553671836853,
 'test_dist_l1': 0.32706689834594727,
 'test_dist_l2': 0.39830806851387024,
 'test_dist_logl2': 0.10831732302904129,
 'test_dist_mistake_severity': 0.12222011387348175,
 'test_iou': 0.5488705635070801}
```

CITYSCAPES full
```bash
[INFO] CM IoU - tensor([87.8080, 11.0355,  2.7261])
[INFO] precision tensor([89.0959, 11.9509, 96.7226], dtype=torch.float64) (65.92313412236437) | recall tensor([98.3805, 59.0288,  2.7286], dtype=torch.float64) (53.37928655390315)
Testing: 100%|███████████████████████████████████████████████████████████████████████████████████| 313/313 [09:52<00:00,  1.89s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.6019625067710876,
 'test_acc_w': 0.6019625067710876,
 'test_dist_l1': 0.43416741490364075,
 'test_dist_l2': 0.5064271688461304,
 'test_dist_logl2': 0.11569175124168396,
 'test_dist_mistake_severity': 0.09077001363039017,
 'test_iou': 0.30110037326812744}
```

### TRANSFER LEARNING

2021-04-08 13-31-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474.ckpt

python3 lightning.py --test_checkpoint "lightning_logs/2021-04-08 13-31-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474.ckpt" --num_classes 3 --bs 16 --mode affordances --dataset freiburg --workers 10

FREIBURG test
```bash
[INFO] CM IoU - tensor([93.8466, 80.6957, 81.4507])
[INFO] precision tensor([95.2271, 91.0170, 97.3936], dtype=torch.float64) (94.54588515214604) | recall tensor([98.4788, 87.6787, 83.2657], dtype=torch.float64) (89.80774336524982)
Testing: 100%|███████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:23<00:00,  2.63s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9430193901062012,
 'test_acc_w': 0.943019449710846,
 'test_dist_l1': 0.057968493551015854,
 'test_dist_l2': 0.05994434282183647,
 'test_dist_logl2': 0.023309728130698204,
 'test_dist_mistake_severity': 0.01733790524303913,
 'test_iou': 0.8924306035041809}
```

KITTI full
```bash
[INFO] CM IoU - tensor([82.9753, 26.9090, 20.1698])
[INFO] precision tensor([84.8499, 34.3795, 91.3661], dtype=torch.float64) (70.1984836346142) | recall tensor([97.4065, 55.3243, 20.5617], dtype=torch.float64) (57.764185344493534)
Testing: 100%|█████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:33<00:00,  2.60s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.7355641722679138,
 'test_acc_w': 0.735564112663269,
 'test_dist_l1': 0.3136218786239624,
 'test_dist_l2': 0.41199398040771484,
 'test_dist_logl2': 0.11867493391036987,
 'test_dist_mistake_severity': 0.18600371479988098,
 'test_iou': 0.5820549726486206}
```

CITYSCAPES full
```bash
[INFO] CM IoU - tensor([87.6511, 12.1181, 12.5038])
[INFO] precision tensor([88.8688, 13.2521, 97.0440], dtype=torch.float64) (66.38831053206061) | recall tensor([98.4607, 58.6117, 12.5516], dtype=torch.float64) (56.54133309026261)
Testing: 100%|███████████████████████████████████████████████████████████████████████████████████| 313/313 [09:52<00:00,  1.89s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.6382311582565308,
 'test_acc_w': 0.6382311582565308,
 'test_dist_l1': 0.4002727270126343,
 'test_dist_l2': 0.47728031873703003,
 'test_dist_logl2': 0.11182627826929092,
 'test_dist_mistake_severity': 0.10643207281827927,
 'test_iou': 0.3277997076511383}
```

## Cityscapes

### OBJECTS

2021-03-29 09-16-cityscapes-c30-kl-rgb-epoch=190-val_loss=0.4310.ckpt

python3 lightning.py --test_checkpoint "lightning_logs/2021-03-29 09-16-cityscapes-c30-kl-rgb-epoch=190-val_loss=0.4310.ckpt" --num_classes 30 --bs 16 --mode convert --dataset cityscapes --orig_dataset cityscapes --workers 10

CITYSCAPES test
```bash
[INFO] CM IoU - tensor([97.9642, 67.3182, 93.7300])
[INFO] precision tensor([98.9995, 74.5845, 98.2479], dtype=torch.float64) (90.61064855713354) | recall tensor([98.9438, 87.3575, 95.3234], dtype=torch.float64) (93.8748619444842)
Testing: 100%|█████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:43<00:00,  2.89s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9685177803039551,
 'test_acc_w': 0.9685177206993103,
 'test_dist_l1': 0.03508581966161728,
 'test_dist_l2': 0.04229297116398811,
 'test_dist_logl2': 0.011477082967758179,
 'test_dist_mistake_severity': 0.11446370929479599,
 'test_iou': 0.9396739602088928}
```

KITTI full
```bash
[INFO] CM IoU - tensor([94.2493, 48.2244, 76.8402])
[INFO] precision tensor([96.1396, 83.3671, 80.5457], dtype=torch.float64) (86.68413154960604) | recall tensor([97.9565, 53.3582, 94.3511], dtype=torch.float64) (81.88860482397425)
Testing: 100%|█████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:33<00:00,  2.59s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9073212146759033,
 'test_acc_w': 0.9073211550712585,
 'test_dist_l1': 0.1032998338341713,
 'test_dist_l2': 0.12454189360141754,
 'test_dist_logl2': 0.034755829721689224,
 'test_dist_mistake_severity': 0.11460043489933014,
 'test_iou': 0.8308783173561096}
```

```bash
FREIBURG full
[INFO] CM IoU - tensor([83.8809,  4.2177, 27.0202])
[INFO] precision tensor([89.9695, 55.0758, 27.4489], dtype=torch.float64) (57.498030175326186) | recall tensor([92.5345,  4.3679, 94.5353], dtype=torch.float64) (63.81255653559871)
Testing: 100%|█████████████████████████████████████████████████████████████████████████████████████| 23/23 [01:01<00:00,  2.69s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.6952874064445496,
 'test_acc_w': 0.6952874064445496,
 'test_dist_l1': 0.34853020310401917,
 'test_dist_l2': 0.436165452003479,
 'test_dist_logl2': 0.11820249259471893,
 'test_dist_mistake_severity': 0.1437998265028,
 'test_iou': 0.5334489345550537}
```

### DRIVEABILITY

2021-03-27 14-54-cityscapes-c3-kl-rgb-epoch=191-val_loss=0.0958.ckpt

python3 lightning.py --test_checkpoint "lightning_logs/2021-03-27 14-54-cityscapes-c3-kl-rgb-epoch=191-val_loss=0.0958.ckpt" --num_classes 3 --bs 16 --mode affordances --dataset cityscapes --orig_dataset cityscapes --workers 10

CITYSCAPES test
```bash
[INFO] CM IoU - tensor([97.8280, 69.7919, 94.9561])
[INFO] precision tensor([98.4389, 85.9788, 97.3912], dtype=torch.float64) (93.93629476089629) | recall tensor([99.3697, 78.7554, 97.4344], dtype=torch.float64) (91.85316919720492)
Testing: 100%|█████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:44<00:00,  2.95s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9730347394943237,
 'test_acc_w': 0.973034679889679,
 'test_dist_l1': 0.03152279928326607,
 'test_dist_l2': 0.04063786193728447,
 'test_dist_logl2': 0.011696329340338707,
 'test_dist_mistake_severity': 0.1690148413181305,
 'test_iou': 0.9477444887161255}
```

KITTI full
```bash
[INFO] CM IoU - tensor([92.5024, 22.9389, 70.5896])
[INFO] precision tensor([93.8420, 85.9558, 72.8902], dtype=torch.float64) (84.2293283672935) | recall tensor([98.4803, 23.8321, 95.7200], dtype=torch.float64) (72.67745747721128)
Testing: 100%|█████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:34<00:00,  2.63s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.871475875377655,
 'test_acc_w': 0.871475875377655,
 'test_dist_l1': 0.14236870408058167,
 'test_dist_l2': 0.1700579822063446,
 'test_dist_logl2': 0.046950701624155045,
 'test_dist_mistake_severity': 0.10772020369768143,
 'test_iou': 0.7730973958969116}
```
FREIBURG full
```bash
[INFO] CM IoU - tensor([79.3794,  0.1977, 21.8401])
[INFO] precision tensor([92.5938, 71.4867, 21.9431], dtype=torch.float64) (62.00789424618378) | recall tensor([84.7610,  0.1979, 97.8954], dtype=torch.float64) (60.95143565875034)
Testing: 100%|█████████████████████████████████████████████████████████████████████████████████████| 23/23 [01:01<00:00,  2.66s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.6372231841087341,
 'test_acc_w': 0.6372232437133789,
 'test_dist_l1': 0.46285393834114075,
 'test_dist_l2': 0.6630083918571472,
 'test_dist_logl2': 0.17726299166679382,
 'test_dist_mistake_severity': 0.2758643925189972,
 'test_iou': 0.4681866765022278}
```

### TRANSFER LEARNING

2021-04-09 03-40-cityscapes-c30-kl-rgb-epoch=18-val_loss=0.0918.ckpt

python3 lightning.py --test_checkpoint "lightning_logs/2021-04-09 03-40-cityscapes-c30-kl-rgb-epoch=18-val_loss=0.0918.ckpt" --num_classes 3 --bs 16 --mode affordances --dataset cityscapes --orig_dataset cityscapes --workers 10

CITYSCAPES test
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

KITTI full
```bash
[INFO] CM IoU - tensor([93.8625, 51.9899, 80.0525])
[INFO] precision tensor([95.3059, 85.0376, 84.3645], dtype=torch.float64) (88.23601387911233) | recall tensor([98.4121, 57.2246, 93.9983], dtype=torch.float64) (83.21166323231495)
Testing: 100%|█████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:34<00:00,  2.63s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9148878455162048,
 'test_acc_w': 0.9148878455162048,
 'test_dist_l1': 0.09453120827674866,
 'test_dist_l2': 0.11336925625801086,
 'test_dist_logl2': 0.03354334086179733,
 'test_dist_mistake_severity': 0.11066600680351257,
 'test_iou': 0.843567967414856}
```

FREIBURG full
```bash
[INFO] CM IoU - tensor([82.1477,  3.6380, 28.1015])
[INFO] precision tensor([87.7839, 58.9734, 28.6002], dtype=torch.float64) (58.452524877781066) | recall tensor([92.7508,  3.7324, 94.1567], dtype=torch.float64) (63.54661270429531)
Testing: 100%|█████████████████████████████████████████████████████████████████████████████████████| 23/23 [01:01<00:00,  2.68s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.6946653127670288,
 'test_acc_w': 0.6946653723716736,
 'test_dist_l1': 0.3506442606449127,
 'test_dist_l2': 0.4412635564804077,
 'test_dist_logl2': 0.12423243373632431,
 'test_dist_mistake_severity': 0.14839334785938263,
 'test_iou': 0.5329012870788574}
```