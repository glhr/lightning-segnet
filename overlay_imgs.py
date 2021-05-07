import cv2 as cv
import numpy as np
from argparse import ArgumentParser

from utils import create_folder

# fusionfusion-custom16rll-multi-2021-05-03 22-03-freiburgthermal-c3-sord-1,2,3-a1-logl2-lw-rgb,ir-epoch=41-val_loss=0.0016_affordances

parser = ArgumentParser()
parser.add_argument('--dataset', default="freiburgthermal")
parser.add_argument('--xp', default="mishmash")
parser.add_argument('--model', default="fusionfusion-custom16rll-multi-2021-05-03 11-30-freiburgthermal-c3-sord-1,2,3-a1-logl2-rgb,ir-epoch=39-val_loss=0.0038_affordances")
parser.add_argument('--model2', default=None)
parser.add_argument('--model3', default=None)
parser.add_argument('--ir', default=False, action="store_true")
parser.add_argument('--rgb', default=False, action="store_true")
parser.add_argument('--gt', default=False, action="store_true")
parser.add_argument('--alpha', type=float, default=0.4)
args = parser.parse_args()

alpha = args.alpha

save_folder = "overlay" if args.model2 is None else "overlay_modelcomp"
create_folder(f'results/{args.dataset}/{args.xp}/{save_folder}')

i = 1
while True:
    try:
        f_rgb = f"results/{args.dataset}/{args.xp}/{args.dataset}{i}-orig-rgb_affordances.png"
        f_gt = f"results/{args.dataset}/{args.xp}/{args.dataset}{i}-gt_affordances.png"
        f_ir = f"results/{args.dataset}/{args.xp}/{args.dataset}{i}-orig-ir_affordances.png"
        f_pred = f"results/{args.dataset}/{args.xp}/{args.dataset}{i}-cls-{args.model}.png"
        f_pred2 = f"results/{args.dataset}/{args.xp}/{args.dataset}{i}-cls-{args.model2}.png"
        f_pred3 = f"results/{args.dataset}/{args.xp}/{args.dataset}{i}-cls-{args.model3}.png"

        print(f_rgb)
        img_rgb = cv.imread(f_rgb)
        print(f_gt)
        img_gt = cv.imread(f_gt)

        print(f_pred)
        img_pred = cv.imread(f_pred)
        beta = (1.0 - alpha)

        dst = cv.addWeighted(img_rgb, alpha, img_pred, beta, 0.0)

        spacing = np.ones_like(img_rgb)[:,:10,:]*255

        stack = []
        if args.rgb:

            stack.append(img_rgb)
            stack.append(spacing)
        if args.ir:
            print(f_ir)
            img_ir = cv.imread(f_ir)
            stack.append(img_ir)
            stack.append(spacing)
        if args.gt:
            stack.append(img_gt)
            stack.append(spacing)

        stack.append(dst)
        if args.model2:
            stack.append(spacing)
            img_pred2 = cv.imread(f_pred2)
            dst2 = cv.addWeighted(img_rgb, alpha, img_pred2, beta, 0.0)
            stack.append(dst2)
        if args.model3:
            stack.append(spacing)
            img_pred3 = cv.imread(f_pred3)
            dst3 = cv.addWeighted(img_rgb, alpha, img_pred3, beta, 0.0)
            stack.append(dst3)

        out = np.hstack(stack)
        # cv.imshow('dst', out)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        i_str = str(i)
        i_str = "0"*(5-len(i_str)) + i_str
        cv.imwrite(f"results/{args.dataset}/{args.xp}/{save_folder}/{args.dataset}{i_str}-{args.xp}-pred_overlay.png",out)
        i += 1
    except Exception as e:
    	print(f"stopped at i={i}",e)
    	break


# ffmpeg -r 5 -f image2 -s 1920x1080 -i overlay/freiburgthermal%05d-pred_overlay.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4
