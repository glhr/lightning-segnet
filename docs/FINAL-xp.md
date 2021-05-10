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
[INFO] CM IoU - tensor([97.0429, 62.8041, 90.3248])
[INFO] precision tensor([97.9790, 74.8229, 96.7056], dtype=torch.float64) (89.83584262440912) | recall tensor([99.0251, 79.6327, 93.1923], dtype=torch.float64) (90.61669930707464)
Testing: 100%|█████████████████████████████████████████████████████████| 1115/1115 [24:50<00:00,  1.34s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9575048089027405,
 'test_acc_w': 0.9607035517692566,
 'test_dist_l1': 0.04968160018324852,
 'test_dist_l2': 0.06405442208051682,
 'test_dist_logl2': 0.018049761652946472,
 'test_dist_mistake_severity': 0.16911114752292633,
 'test_iou': 0.9212123155593872}
```

## MIRMultispectral

### Single modality

python3 lightning.py --bs 1 --dataset multispectralseg --test_checkpoint "lightning_logs/2021-05-06 13-48-freiburgthermal-c13-sord-1,2,3-a1-logl2-rgb-epoch=23-val_loss=0.0037.ckpt" --save --save_xp mishmash --modalities rgb --loss_weight --test_set full
```bash
[INFO] CM IoU - tensor([90.3042,  3.4718,  0.0000])
/usr/local/lib/python3.8/dist-packages/torch/tensor.py:521: RuntimeWarning: invalid value encountered in multiply
  return self.to(torch.get_default_dtype()).reciprocal() * other
[INFO] precision tensor([99.9985,  5.4381,  0.0000], dtype=torch.float64) (35.14553745476974) | recall tensor([90.3054,  8.7606,     nan], dtype=torch.float64) (nan)
Testing: 100%|███████████████████████████████████████████████████████████| 820/820 [14:49<00:00,  1.08s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.8772016763687134,
 'test_acc_w': 0.8461166620254517,
 'test_dist_l1': 0.16837453842163086,
 'test_dist_l2': 0.25952693819999695,
 'test_dist_logl2': 0.08297184109687805,
 'test_dist_mistake_severity': 0.37114670872688293,
 'test_iou': 0.7548574805259705}
```

### Fusion

python3 fusion-test.py  --bs 1 --fusion custom --dataset multispectralseg --modalities rgb,ir --save --bs 1 --save_xp mishmash --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-03 11-30-freiburgthermal-c3-sord-1,2,3-a1-logl2-rgb,ir-epoch=39-val_loss=0.0038.ckpt" --loss_weight --fusion_activ softmax --test_set full
```bash
[INFO] CM IoU - tensor([92.0664,  3.4945,  0.0000])
/usr/local/lib/python3.8/dist-packages/torch/tensor.py:521: RuntimeWarning: invalid value encountered in multiply
  return self.to(torch.get_default_dtype()).reciprocal() * other
[INFO] precision tensor([99.9972,  5.4682,  0.0000], dtype=torch.float64) (35.15514374884201) | recall tensor([92.0688,  8.8267,     nan], dtype=torch.float64) (nan)
Testing: 100%|███████████████████████████████████████████████████████████| 820/820 [17:16<00:00,  1.26s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.8942967653274536,
 'test_acc_w': 0.8704556822776794,
 'test_dist_l1': 0.13412420451641083,
 'test_dist_l2': 0.19096612930297852,
 'test_dist_logl2': 0.062305547297000885,
 'test_dist_mistake_severity': 0.2688751518726349,
 'test_iou': 0.7741038203239441}
```

python3 fusion-test.py  --bs 1 --fusion custom --dataset multispectralseg --modalities rgb,ir --save --bs 1 --save_xp mishmash --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-03 22-03-freiburgthermal-c3-sord-1,2,3-a1-logl2-lw-rgb,ir-epoch=41-val_loss=0.0016.ckpt" --loss_weight --fusion_activ softmax --test_set full
```bash
[INFO] CM IoU - tensor([92.1655,  2.0089,  0.0000])
/usr/local/lib/python3.8/dist-packages/torch/tensor.py:521: RuntimeWarning: invalid value encountered in multiply
  return self.to(torch.get_default_dtype()).reciprocal() * other
[INFO] precision tensor([99.9965,  3.3312,  0.0000], dtype=torch.float64) (34.44258612048448) | recall tensor([92.1685,  4.8170,     nan], dtype=torch.float64) (nan)
Testing: 100%|███████████████████████████████████████████████████████████| 820/820 [17:13<00:00,  1.26s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.8939910531044006,
 'test_acc_w': 0.8689529895782471,
 'test_dist_l1': 0.13752460479736328,
 'test_dist_l2': 0.20055583119392395,
 'test_dist_logl2': 0.0643007680773735,
 'test_dist_mistake_severity': 0.29729196429252625,
 'test_iou': 0.7754229307174683}
```

python3 fusion-test.py  --bs 1 --fusion custom --dataset multispectralseg --modalities rgb,ir --save --bs 1 --save_xp mishmash --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-07 11-08-freiburgthermal-c3-sord-1,2,3-a1-logl2-rgb,ir-epoch=25-val_loss=0.0036.ckpt" --loss_weight --fusion_activ softmax --test_set full
```bash
[INFO] CM IoU - tensor([92.2981,  5.1472,  0.0000])
/usr/local/lib/python3.8/dist-packages/torch/tensor.py:521: RuntimeWarning: invalid value encountered in multiply
  return self.to(torch.get_default_dtype()).reciprocal() * other
[INFO] precision tensor([99.9984,  8.5744,  0.0000], dtype=torch.float64) (36.19094958016073) | recall tensor([92.2994, 11.4083,     nan], dtype=torch.float64) (nan)
Testing: 100%|███████████████████████████████████████████████████████████| 820/820 [16:43<00:00,  1.22s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.8973487019538879,
 'test_acc_w': 0.8743297457695007,
 'test_dist_l1': 0.13865084946155548,
 'test_dist_l2': 0.21064996719360352,
 'test_dist_logl2': 0.06660020351409912,
 'test_dist_mistake_severity': 0.3506975769996643,
 'test_iou': 0.7787538170814514}
```

## ThermalVOC

### Single modality

python3 lightning.py --bs 1 --dataset thermalvoc --test_checkpoint "lightning_logs/2021-05-06 13-48-freiburgthermal-c13-sord-1,2,3-a1-logl2-rgb-epoch=23-val_loss=0.0037.ckpt" --save --save_xp mishmash --modalities rgb --test_set full
```bash
 [INFO] CM IoU - tensor([90.3832,  0.0000,  0.0000])
/home/robotlab/rob10/learning-driveability-heatmaps/models/pytorch-unet-segnet/env/lib/python3.6/site-packages/torch/tensor.py:521: RuntimeWarning: invalid value encountered in multiply
  return self.to(torch.get_default_dtype()).reciprocal() * other
[INFO] precision tensor([100.0000,   0.0000,   0.0000], dtype=torch.float64) (33.33333492210979) | recall tensor([90.3832,     nan,     nan], dtype=torch.float64) (nan)
Testing: 100%|█████████████████████████████████████████████████████████| 1466/1466 [20:03<00:00,  1.22it/s]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.903832197189331,
 'test_acc_w': 0.903832197189331,
 'test_dist_l1': 0.12273531407117844,
 'test_dist_l2': 0.17587031424045563,
 'test_dist_logl2': 0.06550530344247818,
 'test_dist_mistake_severity': 0.2762618362903595,
 'test_iou': 0.8403701782226562}
```

### Fusion

python3 fusion-test.py  --bs 1 --fusion custom --dataset thermalvoc --modalities rgb,ir --save --bs 1 --save_xp mishmash --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-03 11-30-freiburgthermal-c3-sord-1,2,3-a1-logl2-rgb,ir-epoch=39-val_loss=0.0038.ckpt" --loss_weight --fusion_activ softmax --test_set full
```bash
[INFO] CM IoU - tensor([95.6144,  0.0000,  0.0000])
/usr/local/lib/python3.8/dist-packages/torch/tensor.py:521: RuntimeWarning: invalid value encountered in multiply
  return self.to(torch.get_default_dtype()).reciprocal() * other
[INFO] precision tensor([100.0000,   0.0000,   0.0000], dtype=torch.float64) (33.333332154151016) | recall tensor([95.6144,     nan,     nan], dtype=torch.float64) (nan)
Testing: 100%|█████████████████████████████████████████████████████████| 1466/1466 [31:18<00:00,  1.28s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9561437368392944,
 'test_acc_w': 0.943773627281189,
 'test_dist_l1': 0.05482174828648567,
 'test_dist_l2': 0.076752670109272,
 'test_dist_logl2': 0.029037248343229294,
 'test_dist_mistake_severity': 0.25003165006637573,
 'test_iou': 0.9304367303848267}
```

python3 fusion-test.py  --bs 1 --fusion custom --dataset thermalvoc --modalities rgb,ir --save --bs 1 --save_xp mishmash --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-03 22-03-freiburgthermal-c3-sord-1,2,3-a1-logl2-lw-rgb,ir-epoch=41-val_loss=0.0016.ckpt" --loss_weight --fusion_activ softmax --test_set full
```bash
[INFO] CM IoU - tensor([95.6481,  0.0000,  0.0000])
/usr/local/lib/python3.8/dist-packages/torch/tensor.py:521: RuntimeWarning: invalid value encountered in multiply
  return self.to(torch.get_default_dtype()).reciprocal() * other
[INFO] precision tensor([100.0000,   0.0000,   0.0000], dtype=torch.float64) (33.33333348007648) | recall tensor([95.6481,     nan,     nan], dtype=torch.float64) (nan)
Testing: 100%|█████████████████████████████████████████████████████████| 1466/1466 [26:33<00:00,  1.09s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.956481397151947,
 'test_acc_w': 0.9419324994087219,
 'test_dist_l1': 0.05644182488322258,
 'test_dist_l2': 0.08228830993175507,
 'test_dist_logl2': 0.03029731847345829,
 'test_dist_mistake_severity': 0.29695916175842285,
 'test_iou': 0.9313881397247314}
```

python3 fusion-test.py  --bs 1 --fusion custom --dataset thermalvoc --modalities rgb,ir --save --bs 1 --save_xp mishmash --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-07 11-08-freiburgthermal-c3-sord-1,2,3-a1-logl2-rgb,ir-epoch=25-val_loss=0.0036.ckpt" --loss_weight --fusion_activ softmax --test_set full
```bash
[INFO] CM IoU - tensor([96.7692,  0.0000,  0.0000])
/usr/local/lib/python3.8/dist-packages/torch/tensor.py:521: RuntimeWarning: invalid value encountered in multiply
  return self.to(torch.get_default_dtype()).reciprocal() * other
[INFO] precision tensor([100.0000,   0.0000,   0.0000], dtype=torch.float64) (33.33333561034439) | recall tensor([96.7692,     nan,     nan], dtype=torch.float64) (nan)
Testing: 100%|█████████████████████████████████████████████████████████| 1466/1466 [27:09<00:00,  1.11s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9676920175552368,
 'test_acc_w': 0.9553976058959961,
 'test_dist_l1': 0.03911534324288368,
 'test_dist_l2': 0.05273018404841423,
 'test_dist_logl2': 0.020468004047870636,
 'test_dist_mistake_severity': 0.21070434153079987,
 'test_iou': 0.9424524307250977}
 ```
