from lightning import *

class LitFusion(LitSegNet):
    def __init__(self, models, conf, pooling_fusion, viz=False, save=False, test_set=None, test_checkpoint = None, test_max=None, **kwargs):
        super().__init__(conf, viz, save, test_set, test_checkpoint, test_max)
        segnet_rgb = models["rgb"]
        segnet_d = models["d"]
        self.model = FusionNet(encoders=[segnet_rgb.encoders,segnet_d.encoders], decoder=segnet_rgb.decoders, classifier=segnet_rgb.classifier, filter_config=segnet_rgb.filter_config, pooling_fusion=pooling_fusion)
        # self.model.init_decoder()

parser.add_argument('--pooling_fusion', default="fuse")
parser = LitSegNet.add_model_specific_args(parser)
args = parser.parse_args()
if args.debug: enable_debug()

logger.debug(args)

logger.warning("Testing phase")

segnet_rgb = LitSegNet(conf=args, model_only=True)
segnet_d = LitSegNet(conf=args, model_only=True)

trainer = pl.Trainer.from_argparse_args(args)

checkpoints = {
    "freiburg": {
        "rgb": "lightning_logs/2021-04-08 13-31-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474.ckpt",
        "d": "lightning_logs/2021-04-17 14-18-freiburg-c6-kl-depth-epoch=149-val_loss=0.3106.ckpt"
    },
    "cityscapes": {
        "rgb": "lightning_logs/2021-04-09 03-40-cityscapes-c30-kl-rgb-epoch=18-val_loss=0.0918.ckpt",
        "d": "lightning_logs/2021-04-18 13-12-cityscapes-c30-kl-depthraw-epoch=22-val_loss=0.1251.ckpt"
    }
}

dataset = segnet_rgb.hparams.dataset

def parse_chkpt(checkpoint):
    c = "fusion"+checkpoint.split("/")[-1].replace(".ckpt", "")
    return c
#create_folder(f"{segnet_rgb.result_folder}/{chkpt}")

segnet_rgb = segnet_rgb.load_from_checkpoint(checkpoint_path=checkpoints[dataset]["rgb"], modalities="rgb", conf=args)

segnet_d = segnet_d.load_from_checkpoint(checkpoint_path=checkpoints[dataset]["d"], modalities="depth", conf=args)

models = {
    "rgb": segnet_rgb.model,
    "d": segnet_d.model
}

fusionnet = LitFusion(models = models, conf=args, pooling_fusion = args.pooling_fusion, test_max = args.test_samples, test_checkpoint=parse_chkpt(checkpoints[dataset]["rgb"]), save=args.save, viz=args.viz, test_set=args.test_set)

create_folder(f'{fusionnet.result_folder}/{parse_chkpt(checkpoints[dataset]["rgb"])}')


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
    trainer.test(fusionnet)
