import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


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
    fig, axes = plt.subplots(   ncols=len([im for im in imgs.values() if im is not None]),
                                nrows=2,
                                figsize=(15, 4))

    axes[0][0].imshow(imgs["mask"])
    axes[0][0].set_title('Ground truth')
    axes[0][0].axis('off')

    axes[0][1].imshow(imgs["image"])
    axes[0][1].set_title('RGB')
    axes[0][1].axis('off')

    if imgs.get("depth") is not None:
        axes[0][2].imshow(imgs["depth"], cmap=plt.cm.gray)
        axes[0][2].set_title('Depth')
        axes[0][2].axis('off')
    if imgs.get("ir") is not None:
        axes[0][3].imshow(imgs["ir"], cmap=plt.cm.gray)
        axes[0][3].set_title('IR')
        axes[0][3].axis('off')

    axes[1][0].imshow(augmented["mask"])
    axes[1][0].axis('off')

    axes[1][1].imshow(augmented["image"])
    axes[1][1].axis('off')

    if imgs.get("depth") is not None:
        axes[1][2].imshow(augmented["depth"], cmap=plt.cm.gray)
        axes[1][2].axis('off')
    if imgs.get("ir") is not None:
        axes[1][3].imshow(augmented["ir"], cmap=plt.cm.gray)
        axes[1][3].axis('off')

    plt.tight_layout()
    plt.show()
