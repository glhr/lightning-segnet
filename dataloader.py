import json

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision import transforms

import glob

ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MMDataLoader():
    def __init__(self, modalities, name):
        self.name = name
        self.idx_to_color, self.color_to_idx, self.class_to_idx, self.idx_to_idx = {}, {}, {}, {}
        self.modalities = modalities

        self.idx_to_color['objects'] = self.idx_to_color.get('objects', dict())
        self.class_to_idx['objects'] = self.class_to_idx.get('objects', dict())
        self.color_to_idx['objects'] = self.color_to_idx.get('objects', dict())

        self.filenames = []

        self.img_transforms = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self, pilRGB, pilDep, pilIR, imgGT, augment=False, color_GT=True):
        if color_GT:
            imgGT = np.array(imgGT)[:, :, ::-1]
        else:
            imgGT = np.array(imgGT)

        if pilRGB is not None: widthRGB, heightRGB = pilRGB.size
        if pilDep is not None: widthDep, heightDep = pilDep.size
        if pilIR is not None: widthIR, heightIR = pilIR.size

        if augment:
            if pilRGB is not None: pilRGB = self.data_augmentation(pilRGB, img_height=heightRGB, img_width=widthRGB)
            if pilDep is not None: pilDep = self.data_augmentation(pilDep, img_height=heightDep, img_width=widthDep)
            if pilIR is not None: pilIR = self.data_augmentation(pilIR, img_height=heightIR, img_width=widthIR)

        if pilRGB is not None: imgRGB = np.array(pilRGB)
        if pilDep is not None: imgDep = np.array(pilDep)
        if pilIR is not None: imgIR = np.array(pilIR)

        resize = (480,360)
        if pilRGB is not None: modRGB = cv2.resize(imgRGB, dsize=resize, interpolation=cv2.INTER_LINEAR) / 255
        if pilDep is not None: modDepth = cv2.resize(imgDep, dsize=resize, interpolation=cv2.INTER_NEAREST) / 255
        if pilIR is not None: modIR = cv2.resize(imgIR, dsize=resize, interpolation=cv2.INTER_LINEAR) / 255
        modGT = cv2.resize(imgGT, dsize=resize, interpolation=cv2.INTER_NEAREST)
        # print(modGT.shape)
        # print(modGT[0][0])

        if color_GT:
            modGT = cv2.cvtColor(modGT, cv2.COLOR_BGR2RGB)
            modGT = self.mask_to_class_rgb(modGT)
        # print(modGT.shape)

        if pilRGB is not None: modRGB = modRGB[: , :, 2]
        if pilDep is not None: modDepth = modDepth[: , :, 2]
        if pilIR is not None: modIR = modIR[: , :, 2]

        imgs = []
        img = {
            'rgb': modRGB if pilRGB is not None else None,
            'depth': modDepth if pilDep is not None else None,
            'ir': modIR if pilIR is not None else None
        }
        for mod in self.modalities:
            if img[mod] is not None:
                imgs.append(img[mod].copy())

        return torch.from_numpy(np.array(imgs)).float(), modGT

    def remap_classes(self, idx_to_color):

        undriveable = ['sky','vegetation','obstacle','person','car','pole','tree','building','guardrail','rider','motorcycle','bicycle',
        'bus','truck','trafficlight','trafficsign','wall','fence','train','trailer','caravan','polegroup','dynamic','licenseplate','static']
        void = ['void','egovehicle','outofroi','rectificationborder','unlabeled']
        driveable = ['road','path','ground','bridge','tunnel']
        between = ['grass','terrain','sidewalk','parking','railtrack']
        objclass_to_driveidx = dict()

        for i in undriveable:
            objclass_to_driveidx[i] = 1
        for i in driveable:
            objclass_to_driveidx[i] = 3
        for i in between:
            objclass_to_driveidx[i] = 2
        for i in void:
            objclass_to_driveidx[i] = 0
        print(objclass_to_driveidx)
        idx_to_color_new = {
            0: (0,0,0),
            1: (255,0,0),
            2: (255,255,0),
            3: (0,255,0)
        }
        color_to_idx_new = dict()
        conversion = dict()
        idx_to_idx = dict()
        for cls,new_idx in objclass_to_driveidx.items():
            try:
                old_idx = self.class_to_idx["objects"][cls]
                # print(old_idx)
                for v,k in idx_to_color.items():
                    # print(cls,k,v,old_idx,v==old_idx,new_idx)
                    if v==old_idx:
                        color_to_idx_new[k] = new_idx
                        conversion[old_idx] = idx_to_color_new[new_idx]
                        idx_to_idx[old_idx] = new_idx
            except KeyError:
                # print(cls, new_idx)
                pass

        print(conversion)
        return color_to_idx_new, idx_to_color_new, conversion, idx_to_idx

    def get_color(self, x, mode="objects"):
        try:
            return self.idx_to_color[mode][x]
        except KeyError:
            if mode=="objects": print(f"mapping {x} to black")
            return (0,0,255)

    def labels_to_color(self, labels, mode="objects"):
        bs = labels.shape
        data = np.zeros((bs[0], bs[1], 3), dtype=np.uint8)

        for idx in np.unique(labels):
            data[labels==idx] = self.get_color(idx, mode=mode)
        return data

    def labels_obj_to_aff(self, labels):
        # print(self.idx_to_idx["convert"])

        new_labels = torch.zeros_like(labels)

        for old_idx in torch.unique(labels):
            new_labels[labels==old_idx] = self.idx_to_idx["convert"][old_idx.item()]
        return new_labels

    def mask_to_class_rgb(self, mask, mode="objects"):
        # print('----mask->rgb----')
        mask = torch.from_numpy(np.array(mask))
        # mask = torch.squeeze(mask)  # remove 1

        # check the present values in the mask, 0 and 255 in my case
        # print('unique values rgb    ', torch.unique(mask))
        # -> unique values rgb     tensor([  0, 255], dtype=torch.uint8)

        class_mask = mask
        class_mask = class_mask.permute(2, 0, 1).contiguous()
        # print('unique values rgb    ', torch.unique(class_mask))
        h, w = class_mask.shape[1], class_mask.shape[2]
        # print(h,w)
        mask_out = torch.zeros(h, w, dtype=torch.long)

        for k in self.color_to_idx[mode]:
            # print(k)
            # print(torch.unique(class_mask), torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            idx = (class_mask == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            # print(idx)
            validx = (idx.sum(0) == 3)
            # print(validx[0])

            mask_out[validx] = torch.tensor(self.color_to_idx[mode][k], dtype=torch.long)

            #print(mask_out[validx])

        # check the present values after mapping, in my case 0, 1, 2, 3
        # print('unique values mapped ', torch.unique(mask_out))
        # -> unique values mapped  tensor([0, 1, 2, 3])

        return mask_out

    def result_to_image(self, result, iter, orig=None, gt=None, proba=None):
        """
        Converts the output of the network to an actual image
        :param result: The output of the network (with torch.argmax)
        :param iter: The name of the file to save it to
        :return:
        """

        # print(bs,np.max(b))
        if torch.is_tensor(result): result = result.detach().cpu().numpy()
        data = self.labels_to_color(result, mode="affordances")

        # print(colors)
        concat = [data]
        if proba is not None:
            if torch.is_tensor(proba): proba = proba.detach().cpu().numpy()
            # print(np.unique(proba))
            proba = (proba*255).astype(np.uint8)
            proba = np.stack((proba,)*3, axis=-1)
            concat.append(proba)

        if gt is not None:
            if torch.is_tensor(gt): gt = gt.detach().cpu().numpy()
            # concat.append(self.labels_to_color(gt, mode="objects"))
            concat.append(self.labels_to_color(gt, mode="affordances"))


        if orig is not None:
            if torch.is_tensor(orig): orig = orig.squeeze().detach().cpu().numpy()
            orig = (orig*255).astype(np.uint8)
            if orig.shape[-1] != 3:
                orig = np.stack((orig,)*3, axis=-1)
                # print(np.min(orig),np.max(orig))
                concat = [orig] + concat

        data = np.concatenate(concat, axis=1)

        img = Image.fromarray(data, 'RGB')
        img.save(f'results/segnet_{self.name}' + str(iter + 1) + '.png')

    def data_augmentation(self, mods, img_height=360, img_width=480):
        """
        Augments the data
        :param mods:
        :return:
        """
        rand_crop = np.random.uniform(low=0.8, high=0.9)
        rand_scale = np.random.uniform(low=0.5, high=2.0)
        rand_bright = np.random.uniform(low=0, high=0.4)
        rand_cont = np.random.uniform(low=0, high=0.5)
        transform = transforms.RandomApply([
            transforms.RandomApply([transforms.RandomRotation((-13, 13))], p=0.25),
            transforms.RandomApply([transforms.ColorJitter(brightness=rand_bright)], p=0.25),
            transforms.RandomApply([transforms.ColorJitter(contrast=rand_cont)], p=0.25),
            transforms.RandomApply([transforms.RandomCrop((int(img_height * rand_crop), int(img_width * rand_crop))),
                                    transforms.Resize((img_height,img_width))], p=0.25),
            transforms.RandomApply([transforms.Resize((int(img_height * rand_scale), int(img_width * rand_crop))),
                                    transforms.Resize((img_height,img_width))], p=0.25),
            transforms.RandomHorizontalFlip(p=0.25),
            transforms.RandomVerticalFlip(p=.25),
        ], p=.25)
        transformed_img = transform(mods)

        return transformed_img

    def __len__(self):
        # print(len(self.filenames))
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(self.sample(idx))
        if self.train:
            return self.sample(idx, augment=True)
        else:
            return self.sample(idx, augment=False)

class FreiburgDataLoader(MMDataLoader):

    def __init__(self, train=True, path = "../../datasets/freiburg-forest/freiburg_forest_multispectral_annotated/freiburg_forest_annotated/", modalities=["rgb"]):
        """
        Initializes the data loader
        :param path: the path to the data
        """
        super().__init__(modalities, name="freiburg")
        self.path = path

        classes = np.loadtxt(path + "classes.txt", dtype=str)
        print(classes)

        for x in classes:
            x = [int(i) if i.isdigit() else i for i in x]
            self.idx_to_color['objects'][x[4]] = tuple([x[1], x[2], x[3]])
            self.color_to_idx['objects'][tuple([x[1], x[2], x[3]])] = x[4]
            self.class_to_idx['objects'][x[0].lower()] = x[4]

        self.color_to_idx['affordances'], self.idx_to_color['affordances'], self.idx_to_color["convert"], self.idx_to_idx["convert"] = self.remap_classes(self.idx_to_color['objects'])

        if train:
            self.path = path + 'train/'
        else:
            self.path = path + 'test/'

        self.train = train

        for img in glob.glob(self.path + 'GT_color/*.png'):
            img = img.split("/")[-1].split("_")[0]
            # print(img)
            self.filenames.append(img)
        # print(self.filenames)

    def sample(self, sample_id, augment=False):
        a = self.filenames[sample_id]

        suffixes = {
            'depth': "_Clipped_redict_depth_gray.png",
            "rgb": "_Clipped.jpg",
            "gt": "_mask.png",
            "ir": ".tif"
        }

        try:
            # print(a)
            pilRGB = Image.open(self.path + "rgb/" + a + suffixes['rgb']).convert('RGB')
            pilDep = Image.open(self.path + "depth_gray/" + a + suffixes['depth']).convert('RGB')
            pilIR = Image.open(self.path + "nir_gray/" + a + suffixes['ir']).convert('RGB')

            # print(self.path + "GT_color/" + a + suffixes['gt'])
            try:
                # imgGT = cv2.imread(self.path + "GT_color/" + a + suffixes['gt'], cv2.IMREAD_UNCHANGED).astype(np.int8)
                imgGT = Image.open(self.path + "GT_color/" + a + suffixes['gt']).convert('RGB')
            except (AttributeError,IOError):
                suffixes['gt'] = "_Clipped.png"
                # imgGT = cv2.imread(self.path + "GT_color/" + a + suffixes['gt'], cv2.IMREAD_UNCHANGED).astype(np.int8)
                imgGT = Image.open(self.path + "GT_color/" + a + suffixes['gt']).convert('RGB')

            return self.prepare_data(pilRGB, pilDep, pilIR, imgGT, augment)
        except IOError as e:
            print("Error loading " + a, e)
        return False, False, False

class CityscapesDataLoader(MMDataLoader):

    def __init__(self, train=True, path = "../../datasets/cityscapes/", modalities=["rgb"]):
        """
        Initializes the data loader
        :param path: the path to the data
        """
        super().__init__(modalities, name="cityscapes")
        self.path = path

        classes = np.loadtxt(path + "classes.txt", dtype=str)
        print(classes)

        for x in classes:
            x = [int(i) if i.isdigit() or "-" in i else i for i in x]
            self.idx_to_color['objects'][x[4]] = tuple([x[1], x[2], x[3]])
            self.color_to_idx['objects'][tuple([x[1], x[2], x[3]])] = x[4]
            self.class_to_idx['objects'][x[0].lower()] = x[4]

        print("class to idx: ", self.class_to_idx['objects'])
        print("color to idx: ", self.color_to_idx['objects'].values())

        self.color_to_idx['affordances'], self.idx_to_color['affordances'], self.idx_to_color["convert"], self.idx_to_idx["convert"] = self.remap_classes(self.idx_to_color['objects'])

        if train:
            self.split_path = 'train/'
        else:
            self.split_path = 'val/'

        self.train = train
        self.city = "frankfurt"
        self.city_ids = {
            "frankfurt": "000294",
            "berlin": "000019"
        }

        for img in glob.glob(self.path + 'gtFine/' + self.split_path + f'{self.city}/*.png'):
            img = '_'.join(img.split("/")[-1].split("_")[1:3])
            # print(img)
            self.filenames.append(img)
        # print(self.filenames)

    def sample(self, sample_id, augment=False):
        a = self.filenames[sample_id]

        suffixes = {
            'depth': "_Clipped_redict_depth_gray.png",
            "rgb": "_Clipped.jpg",
            "gt": "_mask.png"
        }

        try:
            # print(a)
            pilRGB = Image.open(self.path + "leftImg8bit/" + self.split_path + f"{self.city}/{self.city}_{self.filenames[sample_id]}_leftImg8bit.png").convert('RGB')
            pilDep = Image.open(self.path + "disparity/" + self.split_path + f"{self.city}/{self.city}_{self.filenames[sample_id]}_disparity.png").convert('RGB')
            imgGT = Image.open(self.path + "gtFine/" + self.split_path + f"{self.city}/{self.city}_{self.filenames[sample_id]}_gtFine_labelIds.png").convert('L')
            # print(np.unique(imgGT))

            return self.prepare_data(pilRGB, pilDep, None, imgGT, augment, color_GT=False)
        except IOError as e:
            print("Error loading " + a, e)
        return False, FalseFalse, False

class KittiDataLoader(MMDataLoader):

    def __init__(self, train=True, path = "../../datasets/kitti/", modalities=["rgb"]):
        """
        Initializes the data loader
        :param path: the path to the data
        """
        super().__init__(modalities, name="kitti")
        self.path = path

        classes = np.loadtxt(path + "classes.txt", dtype=str)
        print(classes)

        for x in classes:
            x = [int(i) if i.isdigit() or "-" in i else i for i in x]
            self.idx_to_color['objects'][x[4]] = tuple([x[1], x[2], x[3]])
            self.color_to_idx['objects'][tuple([x[1], x[2], x[3]])] = x[4]
            self.class_to_idx['objects'][x[0].lower()] = x[4]

        print("class to idx: ", self.class_to_idx['objects'])
        print("color to idx: ", self.color_to_idx['objects'].values())

        self.color_to_idx['affordances'], self.idx_to_color['affordances'], self.idx_to_color["convert"] = self.remap_classes(self.idx_to_color['objects'])

        if train:
            self.split_path = 'training/'
        else:
            self.split_path = 'testing/'

        self.train = train

        for img in glob.glob(self.path + "data_semantics/" + self.split_path + "semantic/*.png"):
            img = img.split("/")[-1]
            # print(img)
            self.filenames.append(img)
        # print(self.filenames)

    def sample(self, sample_id, augment=False):
        a = self.filenames[sample_id]

        try:
            # print(a)
            pilRGB = Image.open(self.path + "data_scene_flow/" + self.split_path + "image_2/" + f"{self.filenames[sample_id]}").convert('RGB')
            pilDep = Image.open(self.path + "data_scene_flow/" + self.split_path + "disp_occ_0/" + f"{self.filenames[sample_id]}").convert('RGB')
            imgGT = Image.open(self.path + "data_semantics/" + self.split_path + "semantic/" + f"{self.filenames[sample_id]}").convert('L')
            # print(np.unique(imgGT))

            return self.prepare_data(pilRGB, pilDep, None, imgGT, augment, color_GT=False)
        except IOError as e:
            print("Error loading " + a, e)
        return False, False, False
