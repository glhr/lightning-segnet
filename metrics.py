import skimage
import cv2

import numpy as np
import matplotlib.pyplot as plt

from skimage import filters
from skimage.io import imread


def compute_distmap(image_gray):
    edge = filters.roberts(image_gray)
    print(np.min(edge), np.max(edge))
    edge = np.array(edge > 0,dtype=np.uint8)*255
    edge = 255-edge

    fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True,
                             figsize=(10, 4))

    distmap = cv2.distanceTransform(edge, cv2.DIST_L2, cv2.DIST_MASK_5)
    # distmap[distmap > 100] = 100
    distmap = np.sqrt(distmap)
    distmap = cv2.normalize(distmap, 0, 1)
    # print(np.unique(distmap))

    axes[0].imshow(image_orig)
    axes[0].set_title('Ground trut')

    axes[1].imshow(edge, cmap=plt.cm.gray)
    axes[1].set_title('Edges')

    axes[2].imshow(distmap, cmap=plt.cm.gray)
    axes[2].set_title('Distance map')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    gt_path = 'results/seg-metrics/gt/1-ref_affordances.png'
    pred_path = 'results/seg-metrics/1-cls-freiburg-sord-c4-epoch=668-val_loss=0.00_affordances.png'
    image_gray = imread(gt_path, as_gray=True)
    image_orig = imread(gt_path)
    compute_distmap(image_gray)
