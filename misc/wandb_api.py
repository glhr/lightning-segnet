import wandb
api = wandb.Api()

# for r in ["glhr/segnet-freiburg/1d3btjei","glhr/segnet-freiburg/22b1q7eu","glhr/segnet-freiburg/2gdwcurt","glhr/segnet-freiburg/ysn0vh0m","glhr/segnet-freiburg/2wu0x71s"]:
#     run = api.run(r)
#     run.config["fusion"] = "custom-rll"
#     run.update()

## Batch norm = False
# "glhr/segnet-freiburg/2xa5mpmb" fusionfusion-ssma16-multi-2021-05-01 18-18-cityscapes-c3-kl-rgb,depth-bbn
# "glhr/segnet-freiburg/3sbhsxyj" fusionfusion-ssma16-multi-2021-05-01 15-44-cityscapes-c3-kl-rgb,depthraw-bbn
# "glhr/segnet-freiburg/1crr1ova" fusionfusion-ssma16-single-2021-05-01 09-20-cityscapes-c3-kl-rgb,depthraw-bnn
# "glhr/segnet-freiburg/ajae4d3c" fusionfusion-ssma16-single-2021-04-26 10-54-cityscapes-c3-kl-rgb,depth-bbn

# "glhr/segnet-freiburg/1m0ltly4" fusionfusion-ssma16-single-2021-04-26 06-40-freiburg-c3-kl-rgb,depth
# "glhr/segnet-freiburg/1ck9uxlr" fusionfusion-ssma16-single-2021-04-26 09-56-freiburg-c3-kl-rgb,ir
# "glhr/segnet-freiburg/1jtwq1zb" fusionfusion-ssma16-single-2021-04-25 22-30-freiburg-c3-kl-rgb,depth,ir
# "glhr/segnet-freiburg/mtsxgw1i" fusionfusion-ssma16-multi-2021-04-26 07-35-freiburg-c3-kl-rgb,depth
# "glhr/segnet-freiburg/2ua81i5y" fusionfusion-ssma16-multi-2021-04-26 08-44-freiburg-c3-kl-rgb,ir
# "glhr/segnet-freiburg/vx3x8n6d" fusionfusion-ssma16-multi-2021-04-25 20-49-freiburg-c3-kl-rgb,depth,ir

## Batch norm = True
# fusionfusion-ssma16-single-2021-04-21 10-37-cityscapes-c3-kl-rgb,depth
# fusionfusion-ssma16-single-2021-04-20 22-45-cityscapes-c3-kl-rgb,depthraw
# fusionfusion-ssma16-multi-2021-04-21 23-05-cityscapes-c3-kl-rgb,depthraw
# fusionfusion-ssma16-multi-2021-04-22 00-50-cityscapes-c3-kl-rgb,depth

# fusionfusion-ssma16-single-2021-04-20 15-20-freiburg-c3-kl-rgb,depth
# fusionfusion-ssma16-single-2021-04-22 18-56-freiburg-c3-kl-rgb,ir
# fusionfusion-ssma16-single-2021-04-24 19-43-freiburg-c3-kl-rgb,depth,ir
# fusionfusion-ssma16-multi-2021-04-22 02-33-freiburg-c3-kl-rgb,depth
# fusionfusion-ssma16-multi-2021-04-22 17-47-freiburg-c3-kl-rgb,ir
# fusionfusion-ssma16-multi-2021-04-24 15-05-freiburg-c3-kl-rgb,depth,ir



for r in [  "glhr/segnet-freiburg/2xa5mpmb",
            "glhr/segnet-freiburg/3sbhsxyj",
            "glhr/segnet-freiburg/1crr1ova",
            "glhr/segnet-freiburg/ajae4d3c",
            "glhr/segnet-freiburg/1m0ltly4",
            "glhr/segnet-freiburg/1ck9uxlr",
            "glhr/segnet-freiburg/1jtwq1zb",
            "glhr/segnet-freiburg/mtsxgw1i",
            "glhr/segnet-freiburg/2ua81i5y",
            "glhr/segnet-freiburg/vx3x8n6d"
        ]:
    run = api.run(r)
    run.config["ssma_batchnorm"] = False
    run.update()
