## FReiburg thermal

RGB encoder initialized with obj model from cityscapes
IR encoder trained on obj classes from freiburg thermal
then put into fusion architecture, with output layer = 3 to learn driveability

python3 fusion-test.py  --bs 1 --fusion custom --dataset freiburgthermal --modalities rgb,ir --save --bs 1 --save_xp mishmash --decoders multi --test_checkpoint "lightning_logs/fusionfusion-custom16rll-multi-2021-05-03 11-30-freiburgthermal-c3-sord-1,2,3-a1-logl2-rgb,ir-epoch=39-val_loss=0.0038.ckpt" --loss_weight
```bash
[INFO] CM IoU - tensor([96.9657, 58.9908, 89.8577])
[INFO] precision tensor([97.8827, 76.5533, 95.0873], dtype=torch.float64) (89.84110661041362) | recall tensor([99.0431, 71.9995, 94.2324], dtype=torch.float64) (88.42499848473793)
Testing: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1115/1115 [28:37<00:00,  1.54s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'cm': 0.0,
 'test_acc': 0.9551512002944946,
 'test_acc_w': 0.9585757255554199,
 'test_dist_l1': 0.05222538113594055,
 'test_dist_l2': 0.06697860360145569,
 'test_dist_logl2': 0.018733210861682892,
 'test_dist_mistake_severity': 0.16447745263576508,
 'test_iou': 0.9167538285255432}
```
