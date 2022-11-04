import subprocess
from pathlib import Path

configs = [
# {
#     "dataset_name": "muad",
#     "nclasses": "21",
#     "modalities": "rgb,depth",
#     "test_set": "val",
#     "checkpoints": {
#         "rgb": "lightning_logs/2022-11-02 18-19-muad-c21-ce-rgb,depth-epoch=9-val_mIoU_obj=0.3525.ckpt",
#         "ssma_custom": "lightning_logs/2022-11-03 14-04-muad-c21-ce-rgb,depth-epoch=6-val_mIoU_obj=0.3768.ckpt",
#         "avg": "lightning_logs/2022-11-02 21-58-muad-c21-ce-rgb,depth-epoch=9-val_mIoU_obj=0.3715.ckpt"
#     }
# },
# {
#     "dataset_name": "freiburg",
#     "nclasses": "6",
#     "modalities": "rgb,depth,ir",
#     "test_set": "test",
#     "checkpoints": {
#         "rgb": "lightning_logs/2022-11-02 18-11-freiburg-c6-ce-rgb,depth,ir-epoch=38-val_mIoU_obj=0.6440.ckpt",
#         "ssma_custom": "lightning_logs/2022-11-02 17-24-freiburg-c6-ce-rgb,depth,ir-epoch=38-val_mIoU_obj=0.6551.ckpt",
#         "avg": "lightning_logs/2022-11-02 17-09-freiburg-c6-ce-rgb,depth,ir-epoch=37-val_mIoU_obj=0.6434.ckpt"
#     }
# },
# {
#     "dataset_name": "pst900",
#     "nclasses": "5",
#     "modalities": "rgb,depth,ir",
#     "test_set": "test",
#     "checkpoints": {
#         "rgb": "lightning_logs/2022-11-02 15-13-pst900-c5-ce-rgb,depth,ir-epoch=38-val_mIoU_obj=0.5844.ckpt",
#         "ssma_custom": "lightning_logs/2022-11-02 16-11-pst900-c5-ce-rgb,depth,ir-epoch=32-val_mIoU_obj=0.6062.ckpt",
#         "avg": "lightning_logs/2022-11-02 14-58-pst900-c5-ce-rgb,depth,ir-epoch=40-val_mIoU_obj=0.6216.ckpt"
#     },
#     "class_weights": "1.4536937170316602,44.24574279980519,31.665023906601593,46.40709900799151,30.139092091430634"
# },
{
    "dataset_name": "multispectralseg",
    "nclasses": "9",
    "modalities": "rgb,ir",
    "test_set": "test",
    "checkpoints": {
        "rgb":"lightning_logs/2022-11-04 08-01-multispectralseg-c9-ce-rgb,ir-epoch=22-val_mIoU_obj=0.2913.ckpt",
        "ssma_custom": "lightning_logs/2022-11-04 08-01-multispectralseg-c9-ce-rgb,ir-epoch=22-val_mIoU_obj=0.3188.ckpt",
        "avg": "lightning_logs/2022-11-04 07-33-multispectralseg-c9-ce-rgb,ir-epoch=22-val_mIoU_obj=0.3125.ckpt"
    }
}


]

for config in configs:
    for fusion_mode, checkpoint in config["checkpoints"].items():
        txt_file = f"{config['dataset_name']}-{fusion_mode}-{checkpoint.replace('lightning_logs/','')}.txt"

        if not Path(txt_file).is_file()or True:
            cmd = ["python3","lightning.py","--model","deeplabv3+mm",
            "--num_classes", config['nclasses'],
            "--dataset", config['dataset_name'],
            "--orig_dataset",config['dataset_name'],
            "--modalities",config['modalities'],
            "--fusion_mode",fusion_mode,
            "--test_checkpoint",checkpoint,
            "--save","--save_xp","rgb","--test_set","val","--gpu","1"]
            if "class_weights" in config:
                cmd.extend(["--class_weights",config["class_weights"]])
            subprocess.call(cmd,
            stdout=open(txt_file, 'wb'))
