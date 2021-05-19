import torch
import torch.utils.benchmark as benchmark
import torch.autograd.profiler as profiler

import pickle
import os
from pathlib import Path

from segnet import SegNet
from fusion import FusionNet

from utils import logger, create_folder

num_classes = 3

logger.info(f"... setting up models ...")
create_folder("results/benchmark")

model_segnet_1 = SegNet(num_classes=num_classes, n_init_features=1)
model_segnet_2 = SegNet(num_classes=num_classes, n_init_features=2)
model_segnet_3 = SegNet(num_classes=num_classes, n_init_features=3)

input_size = (1, 1, 240, 480)
x = 255*torch.rand(size=input_size)
with profiler.profile(record_shapes=True, profile_memory=True) as prof:
    with profiler.record_function("model_inference"):
        model_segnet_1(x)

# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1))
print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))


mid_ssma_2 = FusionNet(fusion="ssma", bottleneck=16, fusion_activ="sigmoid", num_classes=num_classes, decoders="single", branches=2)
mid_custom_2 = FusionNet(fusion="custom", bottleneck=16, fusion_activ="softmax", num_classes=num_classes, decoders="single", branches=2)

dual_ssma_2 = FusionNet(fusion="ssma", bottleneck=16, fusion_activ="sigmoid", num_classes=num_classes, decoders="multi", branches=2)
dual_custom_2 = FusionNet(fusion="custom", bottleneck=16, fusion_activ="softmax", num_classes=num_classes, decoders="multi", branches=2)

late_ssma_2 = FusionNet(fusion="ssma", bottleneck=16, fusion_activ="sigmoid", num_classes=num_classes, decoders="late", branches=2)
late_custom_2 = FusionNet(fusion="custom", bottleneck=16, fusion_activ="softmax", num_classes=num_classes, decoders="late", branches=2)



mid_custom_3 = FusionNet(fusion="custom", bottleneck=16, fusion_activ="softmax", num_classes=num_classes, decoders="single", branches=3)
mid_ssma_3 = FusionNet(fusion="ssma", bottleneck=16, fusion_activ="sigmoid", num_classes=num_classes, decoders="single", branches=3)

dual_custom_3 = FusionNet(fusion="custom", bottleneck=16, fusion_activ="softmax", num_classes=num_classes, decoders="multi", branches=3)
dual_ssma_3 = FusionNet(fusion="ssma", bottleneck=16, fusion_activ="sigmoid", num_classes=num_classes, decoders="multi", branches=3)

late_ssma_3 = FusionNet(fusion="ssma", bottleneck=16, fusion_activ="sigmoid", num_classes=num_classes, decoders="late", branches=3)
late_custom_3 = FusionNet(fusion="custom", bottleneck=16, fusion_activ="softmax", num_classes=num_classes, decoders="late", branches=3)


results = []

models = [
    "model_segnet_1",
    "model_segnet_2",
    "model_segnet_3",
    "dual_custom_2",
    "mid_custom_2",
    "late_custom_2",
    "dual_custom_3",
    "mid_custom_3",
    "late_custom_3"
    ]

num_threads = os.cpu_count()
min_run_time = 60

for model in models:
    # label and sub_label are the rows
    # description is the column
    channels = int(model.split("_")[-1])
    label = f'input=(1, {channels}, 240, 480)'
    input_size = (1, channels, 240, 480)
    sub_label = model
    x = 255*torch.rand(size=input_size)

    save = f'results/benchmark/{model} {num_threads}threads {min_run_time}s.pickle'
    if not Path(save).is_file():
        logger.info(f"-> running eval for {model}")
        result = benchmark.Timer(
            stmt=f'{model}(x)',
            setup=f'from __main__ import {model}',
            globals={'x': x},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='test',
        ).blocked_autorange(min_run_time=min_run_time)
        with open(save, 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        logger.info(f"-> found eval for {model}, loading pickle")
        with open(save, 'rb') as handle:
            result = pickle.load(handle)

    results.append(result)


compare = benchmark.Compare(results)
compare.trim_significant_figures()
compare.colorize()
compare.print()
