dataset=own
xp=combo

checkpoint1="2021-05-12 11-24-combo-c30-sord-1,2,3-a1-logl2-rgb-epoch=42-val_loss=0.0060"
checkpoint2="2021-05-12 14-34-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=49-val_loss=0.0023"

python3 lightning.py --dataset $dataset --bs 1 --save --save_xp $xp --save --test_checkpoint "lightning_logs/${checkpoint1}.ckpt"

python3 lightning.py --dataset $dataset --bs 1 --save --save_xp $xp --save --test_checkpoint "lightning_logs/${checkpoint2}.ckpt"

python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${checkpoint1}_affordances" --rgb

python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${checkpoint1}_affordances" --model2 "${checkpoint2}_affordances" --rgb
