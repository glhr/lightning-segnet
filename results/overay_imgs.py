import cv2 as cv
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset', default="freiburgthermal")
parser.add_argument('--xp', default="mishmash")
parser.add_argument('--model', default="fusionfusion-custom16rll-multi-2021-05-03 11-30-freiburgthermal-c3-sord-1,2,3-a1-logl2-rgb,ir-epoch=39-val_loss=0.0038_affordances")
parser.add_argument('--model2', default=None)
parser.add_argument('--ir', default=False, action="store_true")
parser.add_argument('--rgb', default=False, action="store_true")
parser.add_argument('--gt', default=False, action="store_true")
args = parser.parse_args()

alpha = 0.4

for i in range(2000):
    try:
        f_rgb = f"{args.dataset}/{args.xp}/{args.dataset}{i}-orig-rgb_affordances.png"
        f_gt = f"{args.dataset}/{args.xp}/{args.dataset}{i}-gt_affordances.png"
        f_ir = f"{args.dataset}/{args.xp}/{args.dataset}{i}-orig-ir_affordances.png"
        f_pred = f"{args.dataset}/{args.xp}/{args.dataset}{i}-cls-{args.model}.png"
        f_pred2 = f"{args.dataset}/{args.xp}/{args.dataset}{i}-cls-{args.model2}.png"

        img_rgb = cv.imread(f_rgb)
        img_gt = cv.imread(f_gt)

        img_pred = cv.imread(f_pred)
        beta = (1.0 - alpha)

        dst = cv.addWeighted(img_rgb, alpha, img_pred, beta, 0.0)

        spacing = np.ones_like(img_gt)[:,:10,:]*255

        stack = []
        if args.rgb:
            stack.append(img_rgb)
        if args.ir:
            img_ir = cv.imread(f_ir)
            stack.append(img_ir)
        if args.gt:
            stack.append(img_gt)
            stack.append(spacing)
        stack.append(dst)
        if args.model2:
            stack.append(spacing)
            img_pred2 = cv.imread(f_pred2)
            dst2 = cv.addWeighted(img_rgb, alpha, img_pred2, beta, 0.0)
            stack.append(dst2)

        out = np.hstack(stack)
        # cv.imshow('dst', out)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        i_str = str(i)
        i_str = "0"*(5-len(i_str)) + i_str
        cv.imwrite(f"{args.dataset}/{args.xp}/overlay/{args.dataset}{i_str}-{args.xp}-pred_overlay.png",out)
    except Exception as e:
    	print(e)
    	continue


# ffmpeg -r 5 -f image2 -s 1920x1080 -i overlay/freiburgthermal%05d-pred_overlay.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4
