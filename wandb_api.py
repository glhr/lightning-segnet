import wandb
api = wandb.Api()

for r in ["glhr/segnet-freiburg/1d3btjei","glhr/segnet-freiburg/22b1q7eu","glhr/segnet-freiburg/2gdwcurt","glhr/segnet-freiburg/ysn0vh0m","glhr/segnet-freiburg/2wu0x71s"]:
    run = api.run(r)
    run.config["fusion"] = "custom-rll"
    run.update()
