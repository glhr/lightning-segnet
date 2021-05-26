from varname import nameof
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import filters
from skimage.io import imread
from skimage.exposure import rescale_intensity
from skimage.util import random_noise

import torch
import torch.nn as nn
from sklearn.metrics import jaccard_score, confusion_matrix

from utils import logger, enable_debug, print_range, create_folder
import losses

def iou_from_confmat(
    confmat: torch.Tensor,
    num_classes: int,
    absent_score: float = 0.0
):

    logger.debug(confmat)

    intersection = torch.diag(confmat)
    union = confmat.sum(0) + confmat.sum(1) - intersection

    # If this class is absent in both target AND pred (union == 0), then use the absent_score for this class.
    scores = intersection.float() / union.float()
    scores[union == 0] = float('nan')

    logger.debug(scores)

    return scores

class Mistakes(nn.Module):
    def __init__(self, ranks, masking=True):
        super().__init__()
        self.l1 = nn.L1Loss(size_average=False, reduce=False, reduction='none')
        self.l2 = nn.MSELoss(size_average=False, reduce=False, reduction='none')
        self.logl1 = losses.Distance(dist="logl1")
        self.logl2 = losses.Distance(dist="logl2")
        self.masking = masking
        self.ranks = sorted(ranks)
        logger.info(f"Distance ranks {self.ranks}")
        self.mistake_max = max(self.ranks) - min(self.ranks)
        self.mistake_min = self.mistake_max
        distances = []
        for i in range(len(self.ranks)):
            if i > 0:
                distances.append(abs(self.ranks[i]-self.ranks[i-1]))
        self.mistake_min = min(distances)
        logger.info(f"Distances: mistakes from {self.mistake_min} to {self.mistake_max}")


    def forward(self, output, target, debug=False, already_flattened=False, weight_map=None):

        if not already_flattened:
            bs, output, target, weight_map = losses.prepare_sample(output, target, weight_map=weight_map, masking=self.masking)
            output = torch.argmax(output, dim=-1)

        if self.masking:
            mask = target.ge(0)
            # logger.debug(mask, mask.shape)
            # logger.debug(output.shape,target.shape)
            output = output[mask]
            target = target[mask]
            if weight_map is not None: weight_map = weight_map[mask]


        target_orig, output_orig = torch.clone(target).float(), torch.clone(output).float()
        for i,r in enumerate(self.ranks):
            target[target_orig==i] = r
            output[output_orig==i] = r

        incorrect = (target != output)
        correct = (target == output)

        if weight_map is not None:
            correct_w = correct.long() * weight_map
            logger.debug(f"correct_w {correct_w}, {weight_map}")
        else:
            correct_w = correct

        if weight_map is not None:
            samples_w = torch.sum(weight_map, dim=0, keepdim=False)
        else:
            samples_w = torch.sum(torch.ones_like(correct), dim=0, keepdim=False)

        target, output = target.float(), output.float()

        dist_l1 = self.l1(output, target)
        dist_l2 = self.l2(output, target)
        dist_logl2 = self.logl2(output, target)
        #dist_logl1 = self.logl1(output, target)

        mistake_severity = self.l1(output[incorrect], target[incorrect])
        logger.debug(f"L1 distance {dist_l1}")
        logger.debug(f"L1 distance {dist_l1}")

        result = {
            "dist_l1": dist_l1,
            "dist_l2": dist_l2,
            #"dist_logl1": dist_logl1,
            "dist_logl2": dist_logl2,
            "dist_mistake_severity": (mistake_severity - self.mistake_min)/(self.mistake_max - self.mistake_min),
            "correct": correct,
            "correct_w": correct_w,
            "samples_w": samples_w
        }

        return result


class ConfusionMatrix(nn.Module):
    def __init__(self, labels, masking=True):
        super().__init__()
        self.labels = list(labels)
        self.masking = masking
        logger.info(f"Confusion matrix labels: {self.labels}")

    def forward(self, output, target, debug=False, already_flattened=False):

        if not already_flattened:
            output, target = losses.flatten_tensors(output, target)
            output = torch.argmax(output, dim=-1)

        if self.masking:
            mask = target.ge(0)
            # logger.debug(mask, mask.shape)
            # logger.debug(output.shape,target.shape)
            output = output[mask]
            target = target[mask]

        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        cm = confusion_matrix(target, output, labels=self.labels, normalize=None)

        logger.debug(f"CM: {cm}")

        if not torch.is_tensor(cm):
            cm = torch.from_numpy(cm)

        return cm

class MaskedIoU(nn.Module):
    def __init__(self, labels, masking=True):
        super().__init__()
        self.labels = list(labels)
        self.masking = masking
        logger.info(f"IoU labels: {self.labels}")

    def forward(self, output, target, debug=True, already_flattened=False):

        if not already_flattened:
            output, target = losses.flatten_tensors(output, target)
            output = torch.argmax(output, dim=-1)

        if self.masking:
            mask = target.ge(0)
            # logger.debug(mask, mask.shape)
            # logger.debug(output.shape,target.shape)
            output = output[mask]
            target = target[mask]

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


def weight_from_target(target):

    # logger.debug(target.shape)
    distmap = torch.zeros_like(target).float()
    for i,sample in enumerate(target):
        map = np.array(compute_distmap(target[i].detach().cpu().numpy())["combined_map"],dtype=np.float32)
        # logger.debug("map",np.unique(map),map.dtype)
        distmap[i] = torch.from_numpy(map).float()
        # logger.debug("map", torch.unique(distmap[i]))
    return distmap

def compute_hmap(image_gray):
    depth_map = np.zeros_like(image_gray).astype(np.float32)
    for ix, iy in np.ndindex(depth_map.shape):
        depth_map[ix, iy] = ix
    return depth_map

hmap = None

def compute_distmap(image_orig, depth_map=None):
    global hmap
    # logger.debug("img shape",image_orig.shape)
    img_h, img_w = image_orig.shape[:2]
    if image_orig.shape[-1] == 3:
        image_gray = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image_orig
    edge = filters.roberts(image_gray)

    print_range(edge, nameof(edge))
    edge = np.array(edge > 0, dtype=np.uint8)*255
    edge = 255-edge

    distmap_linear = cv2.distanceTransform(edge, cv2.DIST_L2, cv2.DIST_MASK_5)
    # distmap_linear[distmap_linear > 50] = 50
    print_range(distmap_linear, nameof(distmap_linear))

    # logger.debug(np.unique(distmap))

    if depth_map is None:
        depth_map = compute_hmap(image_gray) if hmap is None else hmap
        hmap = depth_map
        # weight_map = np.array([[i for j in range(weight_map.shape[0])] for i in range(weight_map.shape[1])])
        # weight_map = np.power(weight_map,2)

    #
    # depth_map = cv2.blur(depth_map, (10, 10))
    # depth_map = rescale_intensity(depth_map, out_range=(0, 1))
    depth_map = depth_map / img_h
    # depth_map = np.power(depth_map, 2)
    print_range(depth_map, nameof(depth_map))

    distmap = np.copy(distmap_linear)
    for ix, iy in np.ndindex(distmap_linear.shape):
        # logger.debug(np.max(distmap))
        # pow = np.power(distmap_linear[ix, iy], 1-depth_map[ix, iy])
        pow = 1-np.exp(-(distmap_linear[ix, iy])/(1+30*(1-depth_map[ix, iy]**2)**2))
        # pow = 1-np.exp(-(distmap_linear[ix, iy]/(1+10**(1-depth_map[ix, iy]))))
        # pow = 1-np.exp(-(distmap_linear[ix, iy]**2/30*(1+(1-depth_map[ix, iy]))))
        # pow = pow/np.max(pow)
        # pow = min(30,pow)
        distmap[ix, iy] = pow
    print_range(distmap, nameof(distmap))

    # row_sums = distmap.sum(axis=1)
    # distmap = distmap / row_sums[:, np.newaxis]

    # print_range(distmap, nameof(distmap))

    # distmap = cv2.normalize(distmap, 0.1, 1, norm_type=cv2.NORM_MINMAX)
    # distmap_linear = cv2.normalize(distmap_linear, 0.1, 1, norm_type=cv2.NORM_MINMAX)
    # distmap[distmap > 0.7] = 0.7

    combined_map = distmap * depth_map
    print_range(combined_map, nameof(combined_map))
    combined_map = rescale_intensity(combined_map, out_range=(0.1, 1))
    # combined_map = distmap
    print_range(combined_map, nameof(combined_map))

    result = {
        "combined_map": combined_map,
        "image_orig": image_orig,
        "edge": edge,
        "distmap_linear": distmap_linear,
        "depth_map": depth_map
    }
    return result


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--distmap', default=False, action="store_true")
    parser.add_argument('--final', default=False, action="store_true")
    parser.add_argument('--depth', default=False, action="store_true")
    parser.add_argument('--iou', default=False, action="store_true")
    parser.add_argument('--debug', default=True, action="store_true")
    parser.add_argument('--input', default="results/cityscapes/test/cityscapes-munster_000061_000019")
    args = parser.parse_args()
    logger.debug(args)

    if args.debug: enable_debug()

    if args.distmap:
        create_folder("results/distmap/")
        gt_path = f'{args.input}-gt_affordances.png'
        vis_path = f'{args.input}-orig-rgb_affordances.png'
        depth_path = 'datasets/freiburg-forest/freiburg_forest_multispectral_annotated/freiburg_forest_annotated/test/depth_gray/b1-09517_Clipped_redict_depth_gray.png'
        pred_path = gt_path
        image_orig = imread(gt_path)
        image_orig = cv2.resize(image_orig, dsize=(480,240))

        if args.depth:
            depth_gray = imread(depth_path, as_gray=True).astype(np.float32)
            depth_gray = cv2.resize(depth_gray, dsize=(480,240))
            depth_gray = np.max(depth_gray) - depth_gray
            logger.debug(np.unique(depth_gray))
            result = compute_distmap(image_orig, depth_map=depth_gray)
        else:
            result = compute_distmap(image_orig)

        fig, axes = plt.subplots(ncols=5, sharex=True, sharey=True,
                                 figsize=(15, 4))

        axes[0].imshow(result["image_orig"])
        axes[0].set_title('Ground truth')
        cv2.imwrite("results/distmap/gt.png",cv2.cvtColor(result["image_orig"], cv2.COLOR_BGR2RGB))

        axes[1].imshow(result["edge"], cmap=plt.cm.gray)
        axes[1].set_title('Edges')

        cv2.imwrite("results/distmap/edges.png",result["edge"])

        axes[2].imshow(result["distmap_linear"], cmap=plt.cm.gray)
        axes[2].set_title('Edge distance map')

        cv2.imwrite("results/distmap/dmap.png", rescale_intensity(result["distmap_linear"], out_range=(0, 255)))

        axes[3].imshow(result["depth_map"], cmap=plt.cm.gray, vmin=0, vmax=1)
        axes[3].set_title('Depth map')

        cv2.imwrite("results/distmap/hmap.png",result["depth_map"]*255)

        im = axes[4].imshow(result["combined_map"], cmap=plt.cm.gray, vmin=0, vmax=1)
        axes[4].set_title('Combined map')

        cv2.imwrite("results/distmap/wmap.png", rescale_intensity(result["combined_map"], out_range=(0, 255)))

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        # plt.show()


        if args.final:

            filename = args.input.split("/")[-1]
            vis_img = imread(vis_path, as_gray=True)

            fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True,
                                     figsize=(24, 3.95))

            axes[1].imshow(result["image_orig"])
            axes[0].imshow(vis_img, cmap=plt.cm.gray)

            im = axes[2].imshow(result["combined_map"], cmap=plt.cm.jet, vmin=0.1)

            axes[0].axis('off')
            axes[1].axis('off')
            axes[2].axis('off')

            fig.colorbar(im, fraction=0.046, pad=0.04)

            plt.tight_layout()
            # plt.show()
            plt.savefig(f"results/distmap/wmap-cbar-{filename}.pdf")

    if args.iou:
        gt_path = '/home/robotlab/rob10/learning-driveability-heatmaps/report/diagrams/cato-iou-gt-2.png'
        seg_gt = imread(gt_path,as_gray=True).astype(np.float32)
        unique,cnts = np.unique(seg_gt,return_counts=True)
        minority_pixels = np.argmin(cnts)
        logger.debug(unique)
        seg_gt[seg_gt == unique[minority_pixels]] = unique[0]

        unique,cnts = np.unique(seg_gt,return_counts=True)

        for i,val in enumerate(unique):
            seg_gt[seg_gt == val] = i

        pred = np.copy(seg_gt)
        pred[pred==0] = random_noise(seg_gt,mode='salt',amount=0.5,clip=False)[pred==0]
        #logger.debug(np.unique(pred))

        target_tensor = torch.tensor([seg_gt])
        pred_tensor = torch.tensor([pred])
        target_tensor = target_tensor.view(-1,)
        pred_tensor = pred_tensor.view(-1,)
        IoU = MaskedIoU(labels=range(3))
        logger.debug(IoU(pred_tensor, target_tensor, debug=True, already_flattened=True))

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
