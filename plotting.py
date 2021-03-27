import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from utils import create_folder, logger, enable_debug, RANDOM_SEED


def plot_confusion_matrix(array, labels=None, filename=None, folder="", vmax=0.9, cbar=False, cmap="Blues", annot=True, vmin=None):
    l = len(array)
    df_cm = pd.DataFrame(array, range(l), range(l))

    scale = len(labels)*2
    plt.figure(figsize=(scale,scale))

    sn.set(font_scale=1.4) # for label size

    p = sn.heatmap(df_cm, annot=annot, annot_kws={"size": 16}, cmap=cmap, square=True, cbar=cbar, vmax=vmax, vmin=vmin)

    if labels is not None:
        p.set_xticklabels(labels)
        p.set_yticklabels(labels, rotation=90, ha='center',rotation_mode='anchor')

    plt.ylabel("Ground truth")
    plt.xlabel("Prediction")
    plt.tight_layout()

    f = f"{filename}-cm.png" if filename is not None else "cm.png"
    f = f"{folder}/{f}" if len(folder) else f
    plt.savefig(f)


def visualize_data_aug(imgs, augmented):

    modalities = [mod for mod,im in imgs.items() if im is not None]
    indices = {mod:i for i,mod in enumerate(modalities)}
    logger.debug(indices)

    fig, axes = plt.subplots(   ncols=len(modalities),
                                nrows=2,
                                figsize=(15, 4))

    axes[0][indices["mask"]].imshow(imgs["mask"])
    axes[0][indices["mask"]].set_title('Ground truth')
    axes[0][indices["mask"]].axis('off')

    axes[0][indices["image"]].imshow(imgs["image"])
    axes[0][indices["image"]].set_title('RGB')
    axes[0][indices["image"]].axis('off')

    if imgs.get("depth") is not None:
        axes[0][indices["depth"]].imshow(imgs["depth"], cmap=plt.cm.gray, vmin=0, vmax=255)
        axes[0][indices["depth"]].set_title('Depth')
        axes[0][indices["depth"]].axis('off')
    if imgs.get("ir") is not None:
        axes[0][indices["ir"]].imshow(imgs["ir"], cmap=plt.cm.gray, vmin=0, vmax=255)
        axes[0][indices["ir"]].set_title('IR')
        axes[0][indices["ir"]].axis('off')

    axes[1][indices["mask"]].imshow(augmented["mask"])
    axes[1][indices["mask"]].axis('off')

    axes[1][indices["image"]].imshow(augmented["image"])
    axes[1][indices["image"]].axis('off')

    if imgs.get("depth") is not None:
        axes[1][indices["depth"]].imshow(augmented["depth"], cmap=plt.cm.gray, vmin=0, vmax=255)
        axes[1][indices["depth"]].axis('off')
    if imgs.get("ir") is not None:
        axes[1][indices["ir"]].imshow(augmented["ir"], cmap=plt.cm.gray, vmin=0, vmax=255)
        axes[1][indices["ir"]].axis('off')

    plt.tight_layout()
    plt.show()
