import glob
from shutil import copyfile
import cv2

for s in glob.glob("selected/*.png"):
	f = s.split("-")[0].split("/")[1]
	print(f)
	f = f.replace("wilddash","wilddash-")
	loc_pred = f"/media/gala/LaCie/gala/driveability-results/wilddash/paper/{f}-overlay-pred-2021-08-26 07-09-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=109-val_loss=0.0212_affordances.png"
	loc_in = f"/media/gala/LaCie/gala/driveability-results/wilddash/paper/{f}-orig-rgb_affordances.png"
	copyfile(loc_pred, f"selected_singlepred/{f}-overlay-pred-2021-08-26 07-09-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=109-val_loss=0.0212_affordances.png")
	copyfile(loc_in, f"selected_singlepred/{f}-orig-rgb_affordances.png")
	pred = cv2.imread(f"selected_singlepred/{f}-orig-rgb_affordances.png") 
	inp = cv2.imread(f"selected_singlepred/{f}-overlay-pred-2021-08-26 07-09-combo-c3-sord-1,2,3-a1-logl2-lw-rgb-epoch=109-val_loss=0.0212_affordances.png") 
	res = cv2.hconcat([inp,pred])
	cv2.imwrite(f"selected_singledemo/{f}-pair.png",res)
	
