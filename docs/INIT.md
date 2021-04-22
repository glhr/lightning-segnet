python3 fusion-test.py  --bs 1 --fusion ssma --dataset cityscapes --modalities rgb,depthraw --save --bs 1 --save_xp fusion-init --decoders multi
python3 fusion-test.py  --bs 1 --fusion ssma --dataset cityscapes --modalities rgb,depthraw --save --bs 1 --save_xp fusion-init --decoders single
python3 fusion-test.py  --bs 1 --fusion custom --dataset cityscapes --modalities rgb,depthraw --save --bs 1 --save_xp fusion-init --decoders multi --pretrained_last_layer
python3 fusion-test.py  --bs 1 --fusion custom --dataset cityscapes --modalities rgb,depthraw --save --bs 1 --save_xp fusion-init --decoders single
