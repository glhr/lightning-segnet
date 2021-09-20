dataset=wilddash

## objects

python3 lightning.py --test_checkpoint "lightning_logs/2021-03-29 09-16-cityscapes-c30-kl-rgb-epoch=190-val_loss=0.4310.ckpt" --num_classes 30 --bs 16 --mode convert --dataset $dataset --orig_dataset cityscapes --workers 4 --gpu 1 --save --save_xp vap_driv

```
[INFO] CM IoU - tensor([90.0804, 24.2166, 69.9937])
[INFO] precision tensor([96.5079, 33.8952, 80.5489], dtype=torch.float64) (70.31731901416384) | recall tensor([93.1155, 45.8901, 84.2305], dtype=torch.float64) (74.41204301612126)
Testing: 100%|██████████████████████████████████| 13/13 [00:37<00:00,  2.91s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.8906076550483704,
 'test_acc_w': 0.8906076550483704,
 'test_dist_l1': 0.1571289449930191,
 'test_dist_l2': 0.25260230898857117,
 'test_dist_logl2': 0.07628671824932098,
 'test_dist_mistake_severity': 0.4363807141780853,
 'test_iou': 0.8032903075218201}

```

## transfer LEARNING

python3 lightning.py --test_checkpoint "lightning_logs/2021-04-09 03-40-cityscapes-c30-kl-rgb-epoch=18-val_loss=0.0918.ckpt" --num_classes 3 --bs 16 --mode affordances --dataset $dataset --orig_dataset cityscapes --workers 4 --gpu 1 --save_xp vap_driv

```
[INFO] CM IoU - tensor([90.0662, 26.1507, 69.5855])
[INFO] precision tensor([96.4034, 37.8558, 79.5020], dtype=torch.float64) (71.25372996260921) | recall tensor([93.1978, 45.8213, 84.7997], dtype=torch.float64) (74.60626422062546)
Testing: 100%|██████████████████████████████████| 13/13 [00:39<00:00,  3.04s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.8924764394760132,
 'test_acc_w': 0.8924764394760132,
 'test_dist_l1': 0.15949195623397827,
 'test_dist_l2': 0.2634287178516388,
 'test_dist_logl2': 0.07911224663257599,
 'test_dist_mistake_severity': 0.48332083225250244,
 'test_iou': 0.8063532114028931}
```

## transfer learning - SORD

python3 lightning.py --test_checkpoint "lightning_logs/2021-04-08 21-07-cityscapes-c30-sord-1,2,3-a1-logl2-rgb-epoch=23-val_loss=0.0034.ckpt" --num_classes 3 --bs 16 --mode affordances --dataset $dataset --orig_dataset cityscapes --workers 10 --gpu 1 --test_samples 200 --save --save_xp vap_driv

```
[INFO] CM IoU - tensor([92.3901, 23.1639, 71.3595])
[INFO] precision tensor([96.0790, 28.6846, 91.6856], dtype=torch.float64) (72.14969926243859) | recall tensor([96.0102, 54.6191, 76.2968], dtype=torch.float64) (75.6420184356846)
Testing: 100%|██████████████████████████████████| 13/13 [00:38<00:00,  2.96s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.8973527550697327,
 'test_acc_w': 0.8973527550697327,
 'test_dist_l1': 0.12751026451587677,
 'test_dist_l2': 0.17723636329174042,
 'test_dist_logl2': 0.05315626785159111,
 'test_dist_mistake_severity': 0.2422184944152832,
 'test_iou': 0.8139743208885193}
```

## transfer learning - LW

python3 lightning.py --num_classes 3 --bs 16 --mode affordances --test_checkpoint "lightning_logs/2021-04-09 10-00-cityscapes-c30-kl-rgb-epoch=6-val_loss=0.0283.ckpt" --modalities rgb --gpus 1 --dataset $dataset --orig_dataset cityscapes  --test_samples 200 --save --save_xp vap_driv

## transfer learning - SORD + LW (0,10)

python3 lightning.py --num_classes 3 --bs 16 --mode affordances --test_checkpoint "lightning_logs/2021-06-11 13-18-cityscapes-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=32-val_loss=0.0044.ckpt" --modalities rgb --gpus 1 --dataset $dataset --orig_dataset cityscapes  --test_samples 200 --save --save_xp vap_driv

```
[INFO] CM IoU - tensor([92.3883, 22.8753, 72.1534])
[INFO] precision tensor([96.0972, 28.6951, 91.5629], dtype=torch.float64) (72.11837435108279) | recall tensor([95.9901, 53.0054, 77.2923], dtype=torch.float64) (75.42923148521842)
Testing: 100%|██████████████████████████████████| 13/13 [00:59<00:00,  4.56s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.8987846970558167,
 'test_acc_w': 0.8987846970558167,
 'test_dist_l1': 0.12570515275001526,
 'test_dist_l2': 0.17468492686748505,
 'test_dist_logl2': 0.052650514990091324,
 'test_dist_mistake_severity': 0.24195845425128937,
 'test_iou': 0.8163917660713196}
```

## COMBO SORD

python3 lightning.py --num_classes 3 --bs 16 --mode affordances --test_checkpoint "lightning_logs/2021-05-12 11-24-combo-c30-sord-1,2,3-a1-logl2-rgb-epoch=42-val_loss=0.0060.ckpt" --modalities rgb --gpus 1 --dataset $dataset --orig_dataset cityscapes  --test_samples 200 --save --save_xp vap_driv

## COMBO SORD + LW

python3 lightning.py --num_classes 3 --bs 16 --mode affordances --test_checkpoint "lightning_logs/2021-05-12 14-34-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=49-val_loss=0.0023.ckpt" --modalities rgb --gpus 1 --dataset $dataset --orig_dataset cityscapes  --test_samples 200 --save --save_xp vap_driv

## COMBO SORD + LW (0,10)

python3 lightning.py --num_classes 3 --bs 16 --mode affordances --test_checkpoint "lightning_logs/2021-06-08 11-34-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=51-val_loss=0.0192.ckpt" --modalities rgb --gpus 1 --dataset $dataset --orig_dataset cityscapes  --test_samples 200 --save --save_xp vap_driv








[INFO] CM IoU - tensor([92.8371, 23.2144, 69.8034])
[INFO] precision tensor([95.7590, 29.7205, 92.7068], dtype=torch.float64) (72.72875785352349) | recall tensor([96.8179, 51.4669, 73.8593], dtype=torch.float64) (74.04804585015654)
[INFO] CM IoU - tensor([92.1637, 25.4816, 70.1350])
[INFO] precision tensor([95.8572, 31.7159, 91.3808], dtype=torch.float64) (72.98465894672242) | recall tensor([95.9870, 56.4523, 75.1032], dtype=torch.float64) (75.84753408434986)
Testing: 100%|████████████████████████████████| 250/250 [00:53<00:00,  4.71it/s]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.8928411602973938,
 'test_acc_w': 0.8928412199020386,
 'test_dist_l1': 0.1312778741121292,
 'test_dist_l2': 0.17951622605323792,
 'test_dist_logl2': 0.053813375532627106,
 'test_dist_mistake_severity': 0.2250789999961853,
 'test_iou': 0.8142262101173401}
----------------------------------
