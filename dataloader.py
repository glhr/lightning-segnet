import json
import glob
import numpy as np

import cv2
from PIL import Image, ImageFile

import torch
from torchvision import transforms

import albumentations as A

ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MMDataLoader():
    def __init__(self, modalities, name, mode, augment):
        self.name = name
        self.idx_to_color, self.color_to_idx, self.class_to_idx, self.idx_to_idx = {}, {}, {}, {}
        self.modalities = modalities

        self.idx_to_color['objects'] = self.idx_to_color.get('objects', dict())
        self.class_to_idx['objects'] = self.class_to_idx.get('objects', dict())
        self.color_to_idx['objects'] = self.color_to_idx.get('objects', dict())

        self.filenames = []

        self.augment = augment
        self.img_transforms = transforms.Compose([transforms.ToTensor()])

        self.mode = mode
        self.has_affordance_labels = False

        self.cls_labels = ["void", "impossible","possible","preferable"]

    def prepare_data(self, pilRGB, pilDep, pilIR, imgGT, augment=False, color_GT=True):

        imgGT = np.array(imgGT)

        if pilRGB is not None: imgRGB = np.array(pilRGB)
        if pilDep is not None: imgDep = np.array(pilDep)
        if pilIR is not None: imgIR = np.array(pilIR)

        if augment:
            img_dict = {
                'image': imgRGB,
                # 'depth': imgDep,
                # 'ir': imgIR,
                'mask': imgGT
                }
            transformed_imgs = self.data_augmentation(img_dict)
            imgRGB, imgGT = transformed_imgs['image'], transformed_imgs['mask']

        if color_GT: imgGT = imgGT[:, :, ::-1]

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
        else:
            modGT = torch.tensor(modGT, dtype=torch.long)

        if pilRGB is not None: modRGB = modRGB[: , :, 2]
        if pilDep is not None: modDepth = modDepth[: , :, 2]
        if pilIR is not None: modIR = modIR[: , :, 2]

        if self.mode == "affordances" and not self.has_affordance_labels: modGT = self.labels_obj_to_aff(modGT)

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

        idx_mappings = {
            -1: set(),
            0: set(),
            1: set(),
            2: set()
        }

        for i in undriveable:
            objclass_to_driveidx[i] = 0
        for i in driveable:
            objclass_to_driveidx[i] = 2
        for i in between:
            objclass_to_driveidx[i] = 1
        for i in void:
            objclass_to_driveidx[i] = -1


        print(objclass_to_driveidx)
        idx_to_color_new = {
            -1: (0,0,0),
            0: (255,0,0),
            1: (255,255,0),
            2: (0,255,0)
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
                        if old_idx >= 0: idx_mappings[new_idx].add(old_idx)
            except KeyError:
                # print(cls, new_idx)
                pass

        idx_mappings = {k:list(v) for k,v in idx_mappings.items()}
        print(idx_mappings)
        print(conversion)
        print("idx_to_idx", idx_to_idx)
        return color_to_idx_new, idx_to_color_new, conversion, idx_to_idx, idx_mappings

    def get_color(self, x, mode="objects"):
        try:
            if mode in ["affordances","convert"]:
                return self.idx_to_color["affordances"][x]
            else:
                return self.idx_to_color[mode][x]
        except KeyError:
            if mode=="objects": print(f"mapping {x} to black")
            return (0,0,255)

    def labels_to_color(self, labels, mode="objects"):
        bs = labels.shape
        # print(bs)
        data = np.zeros((bs[0], bs[1], 3), dtype=np.uint8)

        for idx in np.unique(labels):
            data[labels==idx] = self.get_color(idx, mode=mode)
            # print(idx, "->", self.get_color(idx, mode=mode))
        return data

    def labels_obj_to_aff(self, labels, proba=False):
        if proba:
            # labels = labels.squeeze()
            # print(labels.shape)
            s = labels.shape
            new_proba = torch.zeros((labels.shape[0], 3, s[2], s[3]))
            # print(new_proba.shape)
            # print(new_proba[3])
            for idx in self.idx_mappings.keys():
                indices = [i for i in self.idx_mappings[idx] if i < labels.shape[1]]
                # print(indices)
                select = torch.index_select(labels,dim=1,index=torch.LongTensor(indices))
                # print(select.shape)
                s = torch.sum(select,dim=1,keepdim=True)
                # print(s.shape)
                new_proba[:,idx] = s.squeeze(1)
            # print(new_proba.shape)
            return new_proba
        else:
            new_labels = torch.zeros_like(labels)

            for old_idx in torch.unique(labels):
                new_labels[labels==old_idx] = self.idx_to_idx["convert"][old_idx.item()]
                # print(old_idx,"->",self.idx_to_idx["convert"][old_idx.item()])
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

    def result_to_image(self, iter, pred_cls=None, orig=None, gt=None, pred_proba=None, test=None, folder=None, filename_prefix=None):
        """
        Converts the output of the network to an actual image
        :param result: The output of the network (with torch.argmax)
        :param iter: The name of the file to save it to
        :return:
        """
        if filename_prefix is None:
            filename_prefix = self.name

        # print(bs,np.max(b))
        concat = []

        if pred_cls is not None:
            if torch.is_tensor(pred_cls): pred_cls = pred_cls.detach().cpu().numpy()
            data = self.labels_to_color(pred_cls, mode=self.mode)
            concat.append(data)

        if gt is not None:
            if torch.is_tensor(gt): gt = gt.detach().cpu().numpy()
            # concat.append(self.labels_to_color(gt, mode="objects"))
            concat.append(self.labels_to_color(gt, mode=self.mode))

        if pred_proba is not None:
            if torch.is_tensor(pred_proba): pred_proba = pred_proba.detach().cpu().numpy()
            # print(np.unique(proba))
            proba = pred_proba/2
            proba = (proba*255).astype(np.uint8)
            proba = np.stack((proba,)*3, axis=-1)
            concat.append(proba)

        # if pred_test is not None:
        #     if torch.is_tensor(test): test = test.detach().cpu().numpy()
        #     # print(np.unique(test))
        #     test = test/np.max(test)
        #     test = (test*255).astype(np.uint8)
        #     test = np.stack((test,)*3, axis=-1)
        #     concat.append(test)


        if orig is not None:
            if torch.is_tensor(orig): orig = orig.squeeze().detach().cpu().numpy()
            orig = (orig*255).astype(np.uint8)
            if orig.shape[-1] != 3:
                orig = np.stack((orig,)*3, axis=-1)
                # print(np.min(orig),np.max(orig))
                concat = [orig] + concat

        data = np.concatenate(concat, axis=1)

        img = Image.fromarray(data, 'RGB')
        folder = "" if folder is None else folder
        img.save(f'results/{folder}/{str(iter + 1)}-{filename_prefix}_{self.mode}.png')

    def data_augmentation(self, imgs, gt=None, img_height=360, img_width=480, p=0.5):
        rand_crop = np.random.uniform(low=0.8, high=0.9)
        transform = A.Compose([
            A.Rotate(limit=10, p=p),
            A.RandomCrop(width=int(img_width * rand_crop), height=int(img_height * rand_crop), p=p),
            A.RandomScale(scale_limit=0.2, p=p),
            A.HorizontalFlip(p=p),
            A.RandomBrightnessContrast(p=p)
            ]
            # additional_targets={'rgb': 'image', 'mask':'mask'}
        )

        transformed = transform(image=imgs['image'], mask=imgs['mask'])

        return transformed

    def sample(self, sample_id, augment=False):
        try:
            pilRGB, pilDep, pilIR, imgGT = self.get_image_pairs(sample_id)

            return self.prepare_data(pilRGB, pilDep, pilIR, imgGT, color_GT=self.color_GT, augment=augment)
        except IOError as e:
            print("Error loading " + self.filenames[sample_id], e)
        return False, False, False

    def __len__(self):
        # print(len(self.filenames))
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(self.sample(idx))
        if self.augment:
            return self.sample(idx, augment=True)
        else:
            return self.sample(idx, augment=False)

class FreiburgDataLoader(MMDataLoader):

    def __init__(self, set="train", path = "../../datasets/freiburg-forest/freiburg_forest_multispectral_annotated/freiburg_forest_annotated/", modalities=["rgb"], mode="affordances", augment=False):
        """
        Initializes the data loader
        :param path: the path to the data
        """
        super().__init__(modalities, name="freiburg", mode=mode, augment=augment)
        self.path = path

        classes = np.loadtxt(path + "classes.txt", dtype=str)
        print(classes)

        if self.mode == "objects":
            self.cls_labels = [0]*len(classes)

        for x in classes:
            x = [int(i) if i.isdigit() else i for i in x]
            self.idx_to_color['objects'][x[4]] = tuple([x[1], x[2], x[3]])
            self.color_to_idx['objects'][tuple([x[1], x[2], x[3]])] = x[4]
            self.class_to_idx['objects'][x[0].lower()] = x[4]
            if self.mode == "objects":
                self.cls_labels[x[4]] = x[0].lower()

        print(self.cls_labels)

        self.color_to_idx['affordances'], self.idx_to_color['affordances'], self.idx_to_color["convert"], self.idx_to_idx["convert"], self.idx_mappings = self.remap_classes(self.idx_to_color['objects'])

        if set == "train":
            self.path = path + 'train/'
        else:
            self.path = path + 'test/'

        self.augment = augment

        for img in glob.glob(self.path + 'GT_color/*.png'):
            img = img.split("/")[-1].split("_")[0]
            # print(img)
            self.filenames.append(img)
        # print(self.filenames)

        self.suffixes = {
            'depth': "_Clipped_redict_depth_gray.png",
            "rgb": "_Clipped.jpg",
            "gt": "_mask.png",
            "ir": ".tif"
        }
        self.color_GT = True

    def get_image_pairs(self, sample_id):
        pilRGB = Image.open(self.path + "rgb/" + self.filenames[sample_id] + self.suffixes['rgb']).convert('RGB')
        pilDep = Image.open(self.path + "depth_gray/" + self.filenames[sample_id] + self.suffixes['depth']).convert('RGB')
        pilIR = Image.open(self.path + "nir_gray/" + self.filenames[sample_id] + self.suffixes['ir']).convert('RGB')

        # print(self.path + "GT_color/" + a + suffixes['gt'])
        try:
            self.suffixes['gt'] = "_Clipped.png"
            # imgGT = cv2.imread(self.path + "GT_color/" + a + suffixes['gt'], cv2.IMREAD_UNCHANGED).astype(np.int8)
            imgGT = Image.open(self.path + "GT_color/" + self.filenames[sample_id] + self.suffixes['gt']).convert('RGB')
        except (AttributeError,IOError):
            self.suffixes['gt'] = "_mask.png"
            # imgGT = cv2.imread(self.path + "GT_color/" + a + suffixes['gt'], cv2.IMREAD_UNCHANGED).astype(np.int8)
            imgGT = Image.open(self.path + "GT_color/" + self.filenames[sample_id] + self.suffixes['gt']).convert('RGB')

        return pilRGB, pilDep, pilIR, imgGT

class CityscapesDataLoader(MMDataLoader):

    def __init__(self, set="train", path = "../../datasets/cityscapes/", modalities=["rgb"], mode="affordances", augment=False):
        """
        Initializes the data loader
        :param path: the path to the data
        """
        super().__init__(modalities, name="cityscapes", mode=mode, augment=augment)
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

        self.color_to_idx['affordances'], self.idx_to_color['affordances'], self.idx_to_color["convert"], self.idx_to_idx["convert"], self.idx_mappings = self.remap_classes(self.idx_to_color['objects'])

        if set == "train":
            self.split_path = 'train/'
        elif set == "val":
            self.split_path = 'val/'
        elif set == "test":
            self.split_path = 'test/'

        self.augment = augment
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

        self.color_GT = False

    def get_image_pairs(self, sample_id):

        pilRGB = Image.open(self.path + "leftImg8bit/" + self.split_path + f"{self.city}/{self.city}_{self.filenames[sample_id]}_leftImg8bit.png").convert('RGB')
        pilDep = Image.open(self.path + "disparity/" + self.split_path + f"{self.city}/{self.city}_{self.filenames[sample_id]}_disparity.png").convert('RGB')
        imgGT = Image.open(self.path + "gtFine/" + self.split_path + f"{self.city}/{self.city}_{self.filenames[sample_id]}_gtFine_labelIds.png").convert('L')
        return pilRGB, pilDep, None, imgGT



class KittiDataLoader(MMDataLoader):

    def __init__(self, set="train", path = "../../datasets/kitti/", modalities=["rgb"], mode="affordances", augment=False):
        """
        Initializes the data loader
        :param path: the path to the data
        """
        super().__init__(modalities, name="kitti", mode=mode, augment=augment)
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

        self.color_to_idx['affordances'], self.idx_to_color['affordances'], self.idx_to_color["convert"], self.idx_to_idx["convert"], self.idx_mappings = self.remap_classes(self.idx_to_color['objects'])

        if set == "train":
            self.split_path = 'training/'
        else:
            self.split_path = 'testing/'

        self.augment = augment

        for img in glob.glob(self.path + "data_semantics/" + self.split_path + "semantic/*.png"):
            img = img.split("/")[-1]
            # print(img)
            self.filenames.append(img)
        # print(self.filenames)
        self.color_GT = False

    def get_image_pairs(self, sample_id):
        pilRGB = Image.open(self.path + "data_scene_flow/" + self.split_path + "image_2/" + f"{self.filenames[sample_id]}").convert('RGB')
        pilDep = Image.open(self.path + "data_scene_flow/" + self.split_path + "disp_occ_0/" + f"{self.filenames[sample_id]}").convert('RGB')
        imgGT = Image.open(self.path + "data_semantics/" + self.split_path + "semantic/" + f"{self.filenames[sample_id]}").convert('L')
        return pilRGB, pilDep, None, imgGT


class OwnDataLoader(MMDataLoader):
    def __init__(self, set="train", path = "../../datasets/own/", modalities=["rgb"], mode="affordances", augment=False):
        super().__init__(modalities, name="own", mode=mode)
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

        self.color_to_idx['affordances'], self.idx_to_color['affordances'], self.idx_to_color["convert"], self.idx_to_idx["convert"], self.idx_mappings = self.remap_classes(self.idx_to_color['objects'])

        if set == "train":
            self.split_path = 'training/'
        else:
            self.split_path = 'testing/'

        self.augment = augment

        for img in glob.glob(self.path + self.split_path + "rgb/*.jpg"):
            img = img.split("/")[-1]
            # print(img)
            self.filenames.append(img)
        print(self.filenames)
        self.color_GT = False
        self.has_affordance_labels = True

    def get_image_pairs(self, sample_id):
        pilRGB = Image.open(self.path + self.split_path + "rgb/" + f"{self.filenames[sample_id]}").convert('RGB')
        width, height = pilRGB.size
        imgGT = Image.new('L', (width, height))
        return pilRGB, None, None, imgGT
