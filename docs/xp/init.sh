python3 fusion-test.py --fusion ssma --dataset cityscapes --modalities rgb,depthraw --save --bs 1 --save_xp fusion-init --decoders multi
python3 fusion-test.py --fusion ssma --dataset cityscapes --modalities rgb,depthraw --save --bs 1 --save_xp fusion-init --decoders single
python3 fusion-test.py --fusion ssma --dataset cityscapes --modalities rgb,depthraw --save --bs 1 --save_xp fusion-init --decoders late

python3 fusion-test.py --fusion custom --dataset cityscapes --modalities rgb,depthraw --save --bs 1 --save_xp fusion-init --decoders multi --pretrained_last_layer --fusion_activ softmax
python3 fusion-test.py --fusion custom --dataset cityscapes --modalities rgb,depthraw --save --bs 1 --save_xp fusion-init --decoders multi --fusion_activ softmax
python3 fusion-test.py --fusion custom --dataset cityscapes --modalities rgb,depthraw --save --bs 1 --save_xp fusion-init --decoders single --fusion_activ softmax
python3 fusion-test.py --fusion custom --dataset cityscapes --modalities rgb,depthraw --save --bs 1 --save_xp fusion-init --decoders late --pretrained_last_layer --fusion_activ softmax
python3 fusion-test.py --fusion custom --dataset cityscapes --modalities rgb,depthraw --save --bs 1 --save_xp fusion-init --decoders late --fusion_activ softmax
