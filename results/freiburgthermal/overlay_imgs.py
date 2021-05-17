import cv2 as cv
import numpy as np

alpha = 0.5

for i in range(2000):
	try:
		f_rgb = f"mishmash/rgb/freiburgthermal{i}-orig-rgb_affordances.png"
		f_gt = f"mishmash/freiburgthermal{i}-gt_affordances.png"
		f_ir = f"mishmash/freiburgthermal{i}-orig-ir_affordances.png"
		f_pred = f"mishmash/freiburgthermal{i}-cls-fusionfusion-custom16rll-multi-2021-05-03 11-30-freiburgthermal-c3-sord-1,2,3-a1-logl2-rgb,ir-epoch=39-val_loss=0.0038_affordances.png"
		img_rgb = cv.imread(f_rgb)
		img_gt = cv.imread(f_pred)
		img_ir = cv.imread(f_ir)
		img_pred = cv.imread(f_pred)
		beta = (1.0 - alpha)
		dst = cv.addWeighted(img_rgb, alpha, img_gt, beta, 0.0)
		out = np.hstack((img_rgb,img_ir,dst))
		# cv.imshow('dst', out)
		# cv.waitKey(0)
		# cv.destroyAllWindows()
		i_str = str(i)
		i_str = "0"*(5-len(i_str)) + i_str
		cv.imwrite(f"mishmash/overlay/freiburgthermal{i_str}-pred_overlay.png",out)
	except Exception as e:
		continue


# ffmpeg -r 5 -f image2 -s 1920x1080 -i overlay/freiburgthermal%05d-pred_overlay.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4

