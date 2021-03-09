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
