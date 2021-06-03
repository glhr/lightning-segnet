import cv2 as cv
import numpy as np
from argparse import ArgumentParser
import glob

from utils import create_folder

# fusionfusion-custom16rll-multi-2021-05-03 22-03-freiburgthermal-c3-sord-1,2,3-a1-logl2-lw-rgb,ir-epoch=41-val_loss=0.0016_affordances

parser = ArgumentParser()
parser.add_argument('--dataset', default="freiburgthermal")
parser.add_argument('--xp', default="mishmash")
parser.add_argument('--model', default="fusionfusion-custom16rll-multi-2021-05-03 11-30-freiburgthermal-c3-sord-1,2,3-a1-logl2-rgb,ir-epoch=39-val_loss=0.0038_affordances")
parser.add_argument('--model2', default=None)
parser.add_argument('--model3', default=None)
parser.add_argument('--model4', default=None)
parser.add_argument('--ir', default=False, action="store_true")
parser.add_argument('--nopred', default=False, action="store_true")
parser.add_argument('--depth', default=False, action="store_true")
parser.add_argument('--depthraw', default=False, action="store_true")
parser.add_argument('--rgb', default=False, action="store_true")
parser.add_argument('--gt', default=False, action="store_true")
parser.add_argument('--error', default=False, action="store_true")
parser.add_argument('--nospacing', default=False, action="store_true")
parser.add_argument('--prefix', default="")
parser.add_argument('--alpha', type=float, default=0.4)
args = parser.parse_args()

alpha = args.alpha

if len(args.prefix):
    args.prefix = "-" + args.prefix

save_folder = f"overlay_{args.model}" if args.model2 is None else f"overlay_modelcomp{args.prefix}_{args.model}"
if args.model2 is not None:
    save_folder += f"_{args.model2}"
if args.model3 is not None:
    save_folder += f"_{args.model3}"
if args.model4 is not None:
    save_folder += f"_{args.model4}"

description = ""
if args.gt: description += "Gt"
if args.rgb:  description += "Rgb"
if args.depth:  description += "D"
if args.depthraw:  description += "Draw"
if args.ir:  description += "Ir"
if len(description): description += "-"

save_folder = save_folder.replace("overlay",f"overlay{description}")

try:
    create_folder(f'results/{args.dataset}/{args.xp}/{save_folder}')
except OSError:
    save_folder = f"overlay_" if args.model2 is None else f"overlay_modelcomp{args.prefix}_"
    for model in [args.model, args.model2, args.model3, args.model4]:
        if model is not None:
            save_folder += model[-20:]
    create_folder(f'results/{args.dataset}/{args.xp}/{save_folder}')

filenames = []

for file in glob.glob(f"results/{args.dataset}/{args.xp}/{args.dataset}-*-cls-{args.model}.png"):
    file = file.replace(f"results/{args.dataset}/{args.xp}/{args.dataset}-","")
    file = file.replace(f"-cls-{args.model}.png","")
    # print("-->",file)
    filenames.append(file)

if not len(filenames):
    print(f"couldn't find anything matching results/{args.dataset}/{args.xp}/{args.dataset}-*-cls-{args.model}.png")

print(args)
i = 1
for i in filenames:
    try:
        f_rgb = f"results/{args.dataset}/{args.xp}/{args.dataset}-{i}-orig-rgb_affordances.png"
        f_gt = f"results/{args.dataset}/{args.xp}/{args.dataset}-{i}-gt_affordances.png"
        f_ir = f"results/{args.dataset}/{args.xp}/{args.dataset}-{i}-orig-ir_affordances.png"
        f_d = f"results/{args.dataset}/{args.xp}/{args.dataset}-{i}-orig-depth_affordances.png"
        f_draw = f"results/{args.dataset}/{args.xp}/{args.dataset}-{i}-orig-depthraw_affordances.png"
        f_pred = f"results/{args.dataset}/{args.xp}/{args.dataset}-{i}-cls-{args.model}.png"
        f_pred2 = f"results/{args.dataset}/{args.xp}/{args.dataset}-{i}-cls-{args.model2}.png"
        f_pred3 = f"results/{args.dataset}/{args.xp}/{args.dataset}-{i}-cls-{args.model3}.png"
        f_pred4 = f"results/{args.dataset}/{args.xp}/{args.dataset}-{i}-cls-{args.model4}.png"

        f_error1=f"results/{args.dataset}/{args.xp}/error/{args.dataset}-{i}-errorw-{args.model}.png"
        f_error2=f"results/{args.dataset}/{args.xp}/error/{args.dataset}-{i}-errorw-{args.model2}.png"
        f_error3=f"results/{args.dataset}/{args.xp}/error/{args.dataset}-{i}-errorw-{args.model3}.png"
        f_error4=f"results/{args.dataset}/{args.xp}/error/{args.dataset}-{i}-errorw-{args.model4}.png"

        # print(f_rgb)
        img_rgb = cv.imread(f_rgb)
        # print(f_gt)
        img_gt = cv.imread(f_gt)

        # print(f_pred)
        img_pred = cv.imread(f_pred)
        beta = (1.0 - alpha)

        dst = cv.addWeighted(img_rgb, alpha, img_pred, beta, 0.0)

        spacing = np.ones_like(img_rgb)[:,:10,:]*255

        stack = []
        errormaps = []


        if args.rgb:
            stack.append(img_rgb)
            if not args.nospacing: stack.append(spacing)
            if args.error:
                errormaps.append(255*np.ones_like(img_rgb))
                if not args.nospacing: errormaps.append(spacing)
        if args.depthraw:
            # print(f_d)
            img_d = cv.imread(f_draw)
            stack.append(img_d)
            if not args.nospacing: stack.append(spacing)
        if args.depth:
            # print(f_d)
            img_d = cv.imread(f_d)
            stack.append(img_d)
            if not args.nospacing: stack.append(spacing)
        if args.ir:
            # print(f_ir)
            img_ir = cv.imread(f_ir)
            stack.append(img_ir)
            if not args.nospacing: stack.append(spacing)
        if args.gt:
            stack.append(img_gt)
            if not args.nospacing and not args.nopred: stack.append(spacing)
            if args.error:
                errormaps.append(255*np.ones_like(img_rgb))
                if not args.nospacing: errormaps.append(spacing)
        if not args.nopred:
            stack.append(dst)
        if args.error:
            # print(f_error1)
            img_error1 = cv.imread(f_error1)
            errormaps.append(img_error1)

        if args.model2 and not args.nopred:
            if not args.nospacing: stack.append(spacing)
            img_pred2 = cv.imread(f_pred2)
            dst2 = cv.addWeighted(img_rgb, alpha, img_pred2, beta, 0.0)
            stack.append(dst2)
            if args.error:
                print(f_error2)
                if not args.nospacing: errormaps.append(spacing)
                img_error2 = cv.imread(f_error2)
                errormaps.append(img_error2)
        if args.model3 and not args.nopred:
            if not args.nospacing: stack.append(spacing)
            img_pred3 = cv.imread(f_pred3)
            dst3 = cv.addWeighted(img_rgb, alpha, img_pred3, beta, 0.0)
            stack.append(dst3)
            if args.error:
                print(f_error3)
                if not args.nospacing: errormaps.append(spacing)
                img_error3 = cv.imread(f_error3)
                errormaps.append(img_error3)
        if args.model4 and not args.nopred:
            if not args.nospacing: stack.append(spacing)
            img_pred4 = cv.imread(f_pred4)
            dst4 = cv.addWeighted(img_rgb, alpha, img_pred4, beta, 0.0)
            stack.append(dst4)

        out = np.hstack(stack)

        if args.error:
            errormaps_out = np.hstack(errormaps)
            out = np.vstack([out,errormaps_out])


        # cv.imshow('dst', out)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        i_str = str(i)
        i_str = "0"*(5-len(i_str)) + i_str




        cv.imwrite(f"results/{args.dataset}/{args.xp}/{save_folder}/{args.dataset}{i_str}-{args.xp}{args.prefix}-pred_overlay.png",out)
    except Exception as e:
    	print(f"stopped at i={i}",e)


# ffmpeg -r 5 -f image2 -s 1920x1080 -i overlay/freiburgthermal%05d-pred_overlay.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4
