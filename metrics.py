import skimage
import cv2

import numpy as np
import matplotlib.pyplot as plt

from skimage import filters
from skimage.io import imread
from skimage.exposure import rescale_intensity


def compute_distmap(image_orig, depth_map=None):
    image_gray = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
    edge = filters.roberts(image_gray)
    print(np.min(edge), np.max(edge))
    edge = np.array(edge > 0,dtype=np.uint8)*255
    edge = 255-edge

    fig, axes = plt.subplots(ncols=5, sharex=True, sharey=True,
                             figsize=(15, 4))

    distmap = cv2.distanceTransform(edge, cv2.DIST_L2, cv2.DIST_MASK_5)
    # distmap[distmap > 100] = 100
    distmap = np.sqrt(distmap)
    distmap = cv2.normalize(distmap, 0.1, 1, norm_type=cv2.NORM_MINMAX)
    # print(np.unique(distmap))

    if depth_map is None:
        depth_map = np.zeros_like(image_gray).astype(np.float32)
        for ix,iy in np.ndindex(depth_map.shape):
            depth_map[ix,iy] = ix
        #weight_map = np.array([[i for j in range(weight_map.shape[0])] for i in range(weight_map.shape[1])])
        #weight_map = np.power(weight_map,2)

    depth_map = np.power(depth_map,2)
    # print(np.min(depth_map),np.max(depth_map))
    depth_map = rescale_intensity(depth_map, out_range=(0.1, 1))
    # print(np.min(depth_map),np.max(depth_map))
    # print(np.unique(depth_map))

    combined_map = distmap * depth_map
    combined_map = cv2.normalize(combined_map, 0.1, 1, norm_type=cv2.NORM_MINMAX)

    axes[0].imshow(image_orig)
    axes[0].set_title('Ground truth')

    axes[1].imshow(edge, cmap=plt.cm.gray)
    axes[1].set_title('Edges')

    axes[2].imshow(distmap, cmap=plt.cm.gray, vmin=0, vmax=1)
    axes[2].set_title('Edge distance map')

    axes[3].imshow(depth_map, cmap=plt.cm.gray, vmin=0, vmax=1)
    axes[3].set_title('Depth map')

    axes[4].imshow(combined_map, cmap=plt.cm.gray, vmin=0, vmax=1)
    axes[4].set_title('Combined map')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    gt_path = '/home/robotlab/rob10/learning-driveability-heatmaps/datasets/freiburg-forest/freiburg_forest_multispectral_annotated/freiburg_forest_annotated/test/GT_color/b1-09517_Clipped.png'
    depth_path = '/home/robotlab/rob10/learning-driveability-heatmaps/datasets/freiburg-forest/freiburg_forest_multispectral_annotated/freiburg_forest_annotated/test/depth_gray/b1-09517_Clipped_redict_depth_gray.png'
    pred_path = gt_path

    depth_gray = imread(depth_path, as_gray=True).astype(np.float32)
    depth_gray = cv2.resize(depth_gray, dsize=(480,240))
    depth_gray = np.max(depth_gray) - depth_gray
    print(np.unique(depth_gray))

    image_orig = imread(gt_path)
    image_orig = cv2.resize(image_orig, dsize=(480,240))
    compute_distmap(image_orig)
