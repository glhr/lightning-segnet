PAPER CHECKPOINTS

Cityscapes

obj, one-hot, rgb "lightning_logs/2021-08-13 15-27-cityscapes-c30-kl-rgb-epoch=197-val_loss=0.4043.ckpt"
obj, one-hot, gray "lightning_logs/2021-03-29 09-16-cityscapes-c30-kl-rgb-epoch=190-val_loss=0.4310.ckpt"
driv, one-hot, gray "lightning_logs/2021-04-09 03-40-cityscapes-c30-kl-rgb-epoch=18-val_loss=0.0918.ckpt"
obj, SORD, gray "lightning_logs/2021-04-08 21-07-cityscapes-c30-sord-1,2,3-a1-logl2-rgb-epoch=23-val_loss=0.0034.ckpt"
obj, SORD, gray + LW "lightning_logs/2021-06-11 13-18-cityscapes-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=32-val_loss=0.0044.ckpt"


Combo

driv, one-hot, rgb "lightning_logs/2021-08-14 18-55-combo-c30-kl-rgb-epoch=68-val_loss=0.1411.ckpt"
driv, one-hot, gray "lightning_logs/2021-08-15 11-58-combo-c30-kl-rgb-epoch=82-val_loss=0.1474.ckpt"
driv, SORD, gray "lightning_logs/2021-08-16 09-02-combo-c30-sord-1,2,3-a1-logl2-rgb-epoch=78-val_loss=0.0062.ckpt"
driv, SORD, gray + LW_0,10 "lightning_logs/2021-08-17 09-47-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=85-val_loss=0.0216.ckpt"
