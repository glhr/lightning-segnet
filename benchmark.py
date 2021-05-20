import torch
import torch.utils.benchmark as benchmark
import torch.autograd.profiler as profiler

import pickle
import os
from pathlib import Path
from varname import varname, nameof


from segnet import SegNet
from fusion import FusionNet

from utils import logger, create_folder
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=False, action="store_true")
parser.add_argument('--force', default=False, action="store_true")
parser.add_argument('--threads', default=os.cpu_count(), type=int)
args = parser.parse_args()
logger.debug(args)


num_classes = 3

logger.info(f"... setting up models ...")
create_folder("results/benchmark")



model_dict = {
  "model_segnet_1": SegNet(num_classes=num_classes, n_init_features=1),
  "model_segnet_2": SegNet(num_classes=num_classes, n_init_features=2),
  "model_segnet_3": SegNet(num_classes=num_classes, n_init_features=3),

  # "mid_ssma_2": FusionNet(fusion="ssma", bottleneck=16, fusion_activ="sigmoid", num_classes=num_classes, decoders="single", branches=2),
  "mid_custom_2": FusionNet(fusion="custom", bottleneck=16, fusion_activ="softmax", num_classes=num_classes, decoders="single", branches=2),

  # "dual_ssma_2": FusionNet(fusion="ssma", bottleneck=16, fusion_activ="sigmoid", num_classes=num_classes, decoders="multi", branches=2),
  "dual_custom_2": FusionNet(fusion="custom", bottleneck=16, fusion_activ="softmax", num_classes=num_classes, decoders="multi", branches=2),

  # "late_ssma_2": FusionNet(fusion="ssma", bottleneck=16, fusion_activ="sigmoid", num_classes=num_classes, decoders="late", branches=2),
  "late_custom_2": FusionNet(fusion="custom", bottleneck=16, fusion_activ="softmax", num_classes=num_classes, decoders="late", branches=2),


  "mid_custom_3": FusionNet(fusion="custom", bottleneck=16, fusion_activ="softmax", num_classes=num_classes, decoders="single", branches=3),
  # "mid_ssma_3": FusionNet(fusion="ssma", bottleneck=16, fusion_activ="sigmoid", num_classes=num_classes, decoders="single", branches=3),

  "dual_custom_3": FusionNet(fusion="custom", bottleneck=16, fusion_activ="softmax", num_classes=num_classes, decoders="multi", branches=3),
  # "dual_ssma_3": FusionNet(fusion="ssma", bottleneck=16, fusion_activ="sigmoid", num_classes=num_classes, decoders="multi", branches=3),

  # "late_ssma_3": FusionNet(fusion="ssma", bottleneck=16, fusion_activ="sigmoid", num_classes=num_classes, decoders="late", branches=3),
  "late_custom_3": FusionNet(fusion="custom", bottleneck=16, fusion_activ="softmax", num_classes=num_classes, decoders="late", branches=3)

}

for model_str,m in model_dict.items():
    if args.cuda:
        m = m.cuda()
    m.eval()

def forward(model_str, x):
    with torch.no_grad():
        model_dict[f"{model_str}"](x)

results = []

num_threads = args.threads
device = "cuda" if args.cuda else f"cpu_{num_threads}threads"

for model_str in model_dict.keys():
    logger.warning(model_str)
    # label and sub_label are the rows
    # description is the column
    channels = int(model_str.split("_")[-1])
    label = f'input=(1, {channels}, 240, 480)'
    input_size = (1, channels, 240, 480)
    sub_label = model_str
    if args.cuda:
        x = 255*torch.rand(size=input_size, device='cuda')
    else:
        x = 255*torch.rand(size=input_size)

    with profiler.profile(record_shapes=False, profile_memory=True) as prof:
        with profiler.record_function("model_inference"):
            model_dict[f"{model_str}"](x)
    prof.export_chrome_trace(f'results/benchmark/{model_str} - {device}.json')

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1))
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=1))

    save = f'results/benchmark/{model_str} - {device}.pickle'
    if args.force or not Path(save).is_file():
        logger.info(f"-> running eval for {model_str}")
        model = model_dict[f"{model_str}"]
        result = benchmark.Timer(
            stmt=f'forward(model_str,x)',
            setup=f'from __main__ import forward',
            globals={'x': x, 'model_str': model_str},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='test',
        ).adaptive_autorange(threshold=0.015, max_run_time=300)
        with open(save, 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        logger.info(f"-> found eval for {model_str}, loading pickle")
        with open(save, 'rb') as handle:
            result = pickle.load(handle)

    results.append(result)


compare = benchmark.Compare(results)
compare.trim_significant_figures()
compare.colorize()
compare.print()
