import numpy as np
from matplotlib import pyplot as plt

correct_pred = np.load(f"acdc-night-correct_pred.npy")
incorrect_pred = np.load(f"acdc-night-incorrect_pred.npy")
bins = np.load(f"acdc-night-bins.npy")

ax = plt.subplot(111)
counts_i, bins_i, _ = ax.hist(bins[:-1], bins, weights = correct_pred, alpha=0.5, color = "green")

counts_i, bins_i, _ = ax.hist(bins[:-1], bins, weights = incorrect_pred, alpha=0.5, color = "red")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(labelleft=False, left=False)


ax.xaxis.set_ticks_position('bottom')
ax.set_xticks([0,1])

plt.show()
