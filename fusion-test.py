from lightning import *


class LitFusion(LitSegNet):
    def __init__(self, conf, fusion, bottleneck, fusion_activ, pretrained_last_layer, late_dilation, decoders, segnet_models=None, viz=False, save=False, test_set=None, test_checkpoint = None, test_max=None, **kwargs):
        super().__init__(conf, viz, save, test_set, test_checkpoint, test_max)
        if segnet_models is not None:
            self.model = FusionNet(segnet_models=segnet_models, fusion=fusion, bottleneck=bottleneck, decoders=decoders, pretrained_last_layer=pretrained_last_layer, late_dilation=late_dilation, fusion_activ=fusion_activ)
            # self.model.init_decoder()
        else:
            self.model = FusionNet(fusion=fusion, bottleneck=bottleneck, decoders=decoders, pretrained_last_layer=pretrained_last_layer, late_dilation=late_dilation, fusion_activ=fusion_activ)

        rll = "rll" if (not pretrained_last_layer and fusion=="custom" and decoders == "multi") else ""
        activ = "-sig" if (fusion_activ == "sigmoid" and fusion=="custom") else ""
        activ = "-softm" if (args.fusion_activ == "softmax" and args.fusion=="ssma") else activ

        self.hparams.save_prefix = f"fusion-{args.fusion}{args.bottleneck}{rll}{activ}-{args.decoders}-" + f"{timestamp}-{self.hparams.dataset}-c{self.hparams.num_classes}-{self.hparams.loss}"
        if self.hparams.loss == "sord":
            self.hparams.save_prefix += f'-{",".join([str(r) for r in self.hparams.ranks])}'
            self.hparams.save_prefix += f'-a{self.hparams.dist_alpha}-{self.hparams.dist}'
        if self.hparams.loss_weight:
            self.hparams.save_prefix += "-lw"
        self.hparams.save_prefix += f'-{",".join(self.hparams.modalities)}'
        logger.info(self.hparams.save_prefix)

parser.add_argument('--fusion', default="ssma")
parser.add_argument('--bottleneck', type=int, default=16)
parser.add_argument('--decoders', default="multi")
parser.add_argument('--fusion_activ', default="sigmoid")
parser.add_argument('--pretrained_last_layer', action="store_true", default=False)
parser.add_argument('--late_dilation', type=int, default=1)
parser = LitSegNet.add_model_specific_args(parser)
args = parser.parse_args()
if args.debug: enable_debug()

logger.debug(args)

logger.warning("Testing phase")

segnet_rgb = LitSegNet(conf=args, model_only=True)
segnet_d = LitSegNet(conf=args, model_only=True)

trainer = pl.Trainer.from_argparse_args(args)

dataset = segnet_rgb.hparams.dataset
raw_depth = "depthraw" in segnet_rgb.hparams.modalities

logger.info(segnet_rgb.hparams.modalities)

checkpoints = {
    "freiburg": {
        "rgb": "lightning_logs/2021-04-08 13-31-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474.ckpt",
        "depth": "lightning_logs/2021-04-17 14-18-freiburg-c6-kl-depth-epoch=149-val_loss=0.3106.ckpt",
        "ir": "lightning_logs/2021-04-17 13-16-freiburg-c6-kl-ir-epoch=128-val_loss=0.1708.ckpt"
    },
    "cityscapes": {
        "rgb": "lightning_logs/2021-04-09 03-40-cityscapes-c30-kl-rgb-epoch=18-val_loss=0.0918.ckpt",
        "depthraw": "lightning_logs/2021-04-18 13-12-cityscapes-c30-kl-depthraw-epoch=22-val_loss=0.1251.ckpt", "depth": "lightning_logs/2021-04-17 23-19-cityscapes-c30-kl-depth-epoch=23-val_loss=0.1222.ckpt"
    }
}

# logger.info(f'using {checkpoints[dataset]["d"]} for depth')

mod1 = segnet_rgb.hparams.modalities[0]
mod2 = segnet_rgb.hparams.modalities[1]

def parse_chkpt(checkpoint):
    rll = "rll" if (not args.pretrained_last_layer and args.fusion=="custom" and args.decoders == "multi") else ""
    activ = "-sig" if (args.fusion_activ == "sigmoid" and args.fusion=="custom") else ""
    activ = "-softm" if (args.fusion_activ == "softmax" and args.fusion=="ssma") else activ
    c = f"fusion-{args.fusion}{args.bottleneck}{rll}{activ}-{args.decoders}-" + f'{segnet_rgb.hparams.dataset}-{args.modalities}'
    # +checkpoint.split("/")[-1].replace(".ckpt", "")
    return c
#create_folder(f"{segnet_rgb.result_folder}/{chkpt}")

segnet_mod1 = segnet_rgb.load_from_checkpoint(checkpoint_path=checkpoints[dataset][mod1], modalities=mod1, conf=args)

segnet_mod2 = segnet_d.load_from_checkpoint(checkpoint_path=checkpoints[dataset][mod2], modalities=mod2, conf=args)

fusionnet = LitFusion(segnet_models=[segnet_mod1.model, segnet_mod2.model], conf=args, test_max = args.test_samples, test_checkpoint=parse_chkpt(checkpoints[dataset][mod1]), save=args.save, viz=args.viz, test_set=args.test_set, fusion=args.fusion, bottleneck=args.bottleneck, decoders=args.decoders, pretrained_last_layer=args.pretrained_last_layer, late_dilation=args.late_dilation, fusion_activ=args.fusion_activ)



# create_folder(f'{fusionnet.result_folder}/{parse_chkpt(checkpoints[dataset]["rgb"])}')


#trainer.test(fusionnet)

if args.prefix is None:
    args.prefix = "fusion"+fusionnet.hparams.save_prefix
logger.debug(args.prefix)

checkpoint_callback = ModelCheckpoint(
    dirpath='lightning_logs',
    filename=args.prefix+'-{epoch}-{val_loss:.4f}',
    verbose=True,
    monitor='val_loss',
    mode='min',
    save_last=True
)
checkpoint_callback.CHECKPOINT_NAME_LAST = f"{args.prefix}-last"

if args.train:
    logger.warning("Training phase")
    wandb_logger = WandbLogger(project='segnet-freiburg', log_model = False, name = args.prefix)
    wandb_logger.log_hyperparams(fusionnet.hparams)
    #wandb_logger.watch(segnet_model, log='parameters', log_freq=100)
    trainer = pl.Trainer.from_argparse_args(args,
        check_val_every_n_epoch=1,
        # ~ log_every_n_steps=10,
        logger=wandb_logger,
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=args.train_checkpoint)
    trainer.fit(fusionnet)
else:
    logger.warning("Testing phase")
    if args.test_checkpoint is not None:
        chkpt = args.test_checkpoint.split("/")[-1].replace(".ckpt", "")
        print(chkpt)
        create_folder(f"{fusionnet.result_folder}/{chkpt}")
        fusionnet = fusionnet.load_from_checkpoint(args.test_checkpoint, conf=args, test_max = args.test_samples, test_checkpoint=chkpt, save=args.save, viz=args.viz, test_set=args.test_set, fusion=args.fusion, bottleneck=args.bottleneck, strict=False, decoders=args.decoders, pretrained_last_layer=args.pretrained_last_layer, late_dilation=args.late_dilation, fusion_activ=args.fusion_activ)
    trainer.test(fusionnet)
