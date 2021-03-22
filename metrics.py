import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import filters
from skimage.io import imread
from skimage.exposure import rescale_intensity
from skimage.util import random_noise

import torch
import torch.nn as nn
from sklearn.metrics import jaccard_score

from utils import logger, enable_debug
from losses import flatten_tensors

class MaskedIoU(nn.Module):
    def __init__(self, labels):
        super().__init__()
        self.labels = list(labels)
        logger.info(f"IoU labels: {self.labels}")

    def forward(self, output, target, debug=False, already_flattened=False):

        if not already_flattened:
            output, target = flatten_tensors(output, target)
            output = torch.argmax(output, dim=-1)

        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        iou_micro = jaccard_score(target, output, labels=self.labels, average='micro', zero_division=0)

        if debug:
            iou_macro = jaccard_score(target, output, labels=self.labels, average='macro', zero_division=0)
            iou_cls = jaccard_score(target, output, labels=self.labels, average=None, zero_division=0)
            logger.debug(f"MaskedIoU inputs: target {target}, pred {output}")
            logger.debug(f"MaskedIoU micro {iou_micro} | macro {iou_macro}")
            logger.debug(f"MaskedIoU per class {iou_cls}")
        else:
            logger.debug(f"MaskedIoU micro {iou_micro}")

        return iou_micro

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

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--distmap', default=False, action="store_true")
    parser.add_argument('--depth', default=False, action="store_true")
    parser.add_argument('--iou', default=False, action="store_true")
    parser.add_argument('--debug', default=True, action="store_true")
    args = parser.parse_args()
    print(args)

    if args.debug: enable_debug()

    if args.distmap:
        gt_path = '/home/robotlab/rob10/learning-driveability-heatmaps/datasets/freiburg-forest/freiburg_forest_multispectral_annotated/freiburg_forest_annotated/test/GT_color/b1-09517_Clipped.png'
        depth_path = '/home/robotlab/rob10/learning-driveability-heatmaps/datasets/freiburg-forest/freiburg_forest_multispectral_annotated/freiburg_forest_annotated/test/depth_gray/b1-09517_Clipped_redict_depth_gray.png'
        pred_path = gt_path
        image_orig = imread(gt_path)
        image_orig = cv2.resize(image_orig, dsize=(480,240))

        if args.depth:
            depth_gray = imread(depth_path, as_gray=True).astype(np.float32)
            depth_gray = cv2.resize(depth_gray, dsize=(480,240))
            depth_gray = np.max(depth_gray) - depth_gray
            print(np.unique(depth_gray))
            compute_distmap(image_orig, depth_map=depth_gray)
        else:
            compute_distmap(image_orig)

    if args.iou:
        gt_path = '/home/robotlab/rob10/learning-driveability-heatmaps/report/diagrams/cato-iou-gt-2.png'
        seg_gt = imread(gt_path,as_gray=True).astype(np.float32)
        unique,cnts = np.unique(seg_gt,return_counts=True)
        minority_pixels = np.argmin(cnts)
        print(unique)
        seg_gt[seg_gt == unique[minority_pixels]] = unique[0]

        unique,cnts = np.unique(seg_gt,return_counts=True)

        for i,val in enumerate(unique):
            seg_gt[seg_gt == val] = i

        pred = np.copy(seg_gt)
        pred[pred==0] = random_noise(seg_gt,mode='salt',amount=0.5,clip=False)[pred==0]
        #print(np.unique(pred))

        target_tensor = torch.tensor([seg_gt])
        pred_tensor = torch.tensor([pred])
        target_tensor = target_tensor.view(-1,)
        pred_tensor = pred_tensor.view(-1,)
        IoU = MaskedIoU(labels=range(3))
        print(IoU(pred_tensor, target_tensor, debug=True, already_flattened=True))

        fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True,
                                 figsize=(15, 4))

        axes[0].imshow(seg_gt, cmap=plt.cm.gray)
        axes[0].set_title('Ground truth')

        axes[1].imshow(pred, cmap=plt.cm.gray)
        axes[1].set_title('Prediction')

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        plt.show()
