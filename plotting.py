import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def plot_confusion_matrix(array, labels=None, filename=None):
    l = len(array)
    df_cm = pd.DataFrame(array, range(l), range(l))

    plt.figure(figsize=(7,5.6))

    sn.set(font_scale=1.4) # for label size

    p = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="YlGnBu", square=True)

    if labels is not None:
        p.set_xticklabels(labels)
        p.set_yticklabels(labels, rotation=90, ha='center',rotation_mode='anchor')

    plt.ylabel("Ground truth")
    plt.xlabel("Prediction")
    plt.tight_layout()

    f = f"lightning_logs/{filename}-cm.png" if filename is not None else "cm.png"
    plt.savefig(f)
