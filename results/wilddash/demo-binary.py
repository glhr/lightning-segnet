import glob
from shutil import copyfile
import cv2
import numpy as np

folders = {
	"driv": "paper",
	"free": "rev-2class",
	"road": "rev-2class"
}
models = {
	"driv": "2021-08-15 11-58-combo-c30-kl-rgb-epoch=82-val_loss=0.1474",
	"free": "2021-11-19 17-42-combo-c30-kl-rgb-epoch=82-val_loss=0.0912",
	"road": "2021-11-19 14-08-combo-c30-kl-rgb-epoch=80-val_loss=0.0711"
}

for s in glob.glob("rev-2class/*rgb_affordances.png"):
	f = s.split("/")[-1].split("_")[0] + "_100000"
	
	try:
		rgb = cv2.imread(f"{folders['driv']}/{f}-orig-rgb_affordances.png")	
		inp1 = cv2.imread(f"{folders['road']}/{f}-overlay-pred-{models['road']}_affordances.png") 
		inp2 = cv2.imread(f"{folders['free']}/{f}-overlay-pred-{models['free']}_affordances.png") 
		inp3 = cv2.imread(f"{folders['driv']}/{f}-overlay-pred-{models['driv']}_affordances.png") 
		space = 255*np.ones_like(rgb)
		space = space[:,:3,:]
		res = cv2.hconcat([rgb,space,inp1,space,inp2,space,inp3])
		cv2.imwrite(f"binary_comp/{f}-pair-inp.png",res)
	except Exception as e:
		print(f,e)
	
