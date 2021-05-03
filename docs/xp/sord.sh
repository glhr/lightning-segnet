## Freiburg

python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset freiburg --loss_weight --test_checkpoint "lightning_logs/2021-04-08 13-31-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474.ckpt" --save --save_xp sord --modalities rgb

python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset freiburg --loss_weight --test_checkpoint "lightning_logs/2021-04-08 14-55-freiburg-c6-sord-1,2,3-a1-l1-rgb-epoch=66-val_loss=0.0195.ckpt" --save --save_xp sord --modalities rgb
python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset freiburg --loss_weight --test_checkpoint "lightning_logs/2021-04-08 13-59-freiburg-c6-sord-1,2,3-a1-l2-rgb-epoch=66-val_loss=0.0278.ckpt" --save --save_xp sord --modalities rgb
python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset freiburg --loss_weight --test_checkpoint "lightning_logs/2021-04-08 14-28-freiburg-c6-sord-1,2,3-a1-logl2-rgb-epoch=74-val_loss=0.0061.ckpt" --save --save_xp sord --modalities rgb

python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset freiburg --loss_weight --test_checkpoint "lightning_logs/2021-04-08 16-53-freiburg-c6-sord-1,2,3-a2-l1-rgb-epoch=66-val_loss=0.0597.ckpt" --save --save_xp sord --modalities rgb
python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset freiburg --loss_weight --test_checkpoint "lightning_logs/2021-04-09 08-26-freiburg-c6-sord-1,2,3-a2-l2-rgb-epoch=43-val_loss=0.1179.ckpt" --save --save_xp sord --modalities rgb
python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset freiburg --loss_weight --test_checkpoint "lightning_logs/2021-04-09 09-04-freiburg-c6-sord-1,2,3-a2-logl2-rgb-epoch=74-val_loss=0.0498.ckpt" --save --save_xp sord --modalities rgb

## Cityscapes
python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset cityscapes --loss_weight --test_checkpoint "lightning_logs/2021-04-09 03-40-cityscapes-c30-kl-rgb-epoch=18-val_loss=0.0918.ckpt" --save --save_xp sord --modalities rgb

python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset cityscapes --loss_weight --test_checkpoint "lightning_logs/2021-04-09 05-09-cityscapes-c30-sord-1,2,3-a1-l1-rgb-epoch=23-val_loss=0.0134.ckpt" --save --save_xp sord --modalities rgb
python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset cityscapes --loss_weight --test_checkpoint "lightning_logs/2021-04-08 19-38-cityscapes-c30-sord-1,2,3-a1-l2-rgb-epoch=21-val_loss=0.0214.ckpt" --save --save_xp sord --modalities rgb
python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset cityscapes --loss_weight --test_checkpoint "lightning_logs/2021-04-08 21-07-cityscapes-c30-sord-1,2,3-a1-logl2-rgb-epoch=23-val_loss=0.0034.ckpt" --save --save_xp sord --modalities rgb

python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset cityscapes --loss_weight --test_checkpoint "lightning_logs/2021-04-08 23-08-cityscapes-c30-sord-1,2,3-a2-l1-rgb-epoch=23-val_loss=0.0391.ckpt" --save --save_xp sord --modalities rgb
python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset cityscapes --loss_weight --test_checkpoint "lightning_logs/2021-04-09 00-39-cityscapes-c30-sord-1,2,3-a2-l2-rgb-epoch=23-val_loss=0.0742.ckpt" --save --save_xp sord --modalities rgb
python3 lightning.py --num_classes 3 --bs 1 --mode affordances --dataset cityscapes --loss_weight --test_checkpoint "lightning_logs/2021-04-09 02-09-cityscapes-c30-sord-1,2,3-a2-logl2-rgb-epoch=23-val_loss=0.0236.ckpt" --save --save_xp sord --modalities rgb
