import json

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision import transforms

import glob

ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FreiburgDataLoader():

    def __init__(self, train=True, path = "../../datasets/freiburg-forest/freiburg_forest_multispectral_annotated/freiburg_forest_annotated/"):
        """
        Initializes the data loader
        :param path: the path to the data
        :param num_examples: The number of examples to use
        :param train_size: The size (in percentage) of the train set
        :param test_size: The size (in percentage) of the test set
        :param date: The signature/date of the model
        """
        self.path = path
        self.color_map = {}
        classes = np.loadtxt(path + "classes.txt", dtype=str)
        print(classes)
        for x in classes:
            x = [int(i) for i in x[1:]]
            self.color_map[x[3]] = [x[0], x[1], x[2]]
        print(self.color_map)

        self.mapping = {
            tuple(rgb):i for i,rgb in self.color_map.items()
        }
        print(self.mapping)

        if train:
            self.path = path + 'train/'
        else:
            self.path = path + 'test/'

        self.train = train
        self.filenames = []
        for img in glob.glob(self.path + 'GT_color/*.png'):
            img = img.split("/")[-1].split("_")[0]
            # print(img)
            self.filenames.append(img)
        # print(self.filenames)

        self.img_transforms = transforms.Compose([transforms.ToTensor()])

    def get_color(self, x):
        return self.color_map[x]

    def result_to_image(self, result, iter, orig=None):
        """
        Converts the output of the network to an actual image
        :param result: The output of the network (with torch.argmax)
        :param iter: The name of the file to save it to
        :return:
        """
        b = result.detach().cpu().numpy()
        # b = result.cpu().detach().numpy()
        bs = b.shape
        # print(bs,np.max(b))
        data = np.zeros((bs[0], bs[1], 3), dtype=np.uint8)
        colors = set()
        for y in range(bs[0]):
            for x in range(bs[1]):
                # if b[y, x]>0: print(b[y, x])
                data[y, x] = self.get_color(b[y, x])
                colors.add(b[y, x])

        # print(colors)
        if orig is not None:
            orig = orig.squeeze().detach().cpu().numpy()
            orig = (orig*255).astype(np.uint8)
            if orig.shape[-1] != 3:
                orig = np.stack((orig,)*3, axis=-1)
                # print(np.min(orig),np.max(orig))
            data = np.concatenate((orig,data), axis=1)

        img = Image.fromarray(data, 'RGB')
        img.save('results/segnet_' + str(iter + 1) + '.png')

    def mask_to_class_rgb(self, mask):
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

        for k in self.mapping:
            # print(torch.unique(class_mask), torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            idx = (class_mask == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            # print(idx)
            validx = (idx.sum(0) == 3)
            # print(validx[0])

            mask_out[validx] = torch.tensor(self.mapping[k], dtype=torch.long)

            #print(mask_out[validx])

        # check the present values after mapping, in my case 0, 1, 2, 3
        # print('unique values mapped ', torch.unique(mask_out))
        # -> unique values mapped  tensor([0, 1, 2, 3])

        return mask_out

    def sample(self, sample_id, modalities=["rgb"], augment=False):
        """
        Samples a single image
        :param sample_id: The ID of the image
        :return:
        """
        a = self.filenames[sample_id]

        suffixes = {
            'depth': "_Clipped_redict_depth_gray.png",
            "rgb": "_Clipped.jpg",
            "gt": "_mask.png"
        }

        try:
            # print(a)
            pilRGB = Image.open(self.path + "rgb/" + a + suffixes['rgb']).convert('RGB')
            pilDep = Image.open(self.path + "depth_gray/" + a + suffixes['depth']).convert('RGB')
            widthRGB, heightRGB = pilRGB.size
            widthDep, heightDep = pilDep.size

            if augment:
                pilRGB = self.data_augmentation(pilRGB, img_height=heightRGB, img_width=widthRGB)
                pilDep = self.data_augmentation(pilDep, img_height=heightDep, img_width=widthDep)

            imgRGB = np.array(pilRGB)
            imgDep = np.array(pilDep)

            # print(self.path + "GT_color/" + a + suffixes['gt'])
            try:
                # imgGT = cv2.imread(self.path + "GT_color/" + a + suffixes['gt'], cv2.IMREAD_UNCHANGED).astype(np.int8)
                imgGT = Image.open(self.path + "GT_color/" + a + suffixes['gt']).convert('RGB')
            except (AttributeError,IOError):
                suffixes['gt'] = "_Clipped.png"
                # imgGT = cv2.imread(self.path + "GT_color/" + a + suffixes['gt'], cv2.IMREAD_UNCHANGED).astype(np.int8)
                imgGT = Image.open(self.path + "GT_color/" + a + suffixes['gt']).convert('RGB')
            imgGT = np.array(imgGT)[:, :, ::-1]

            resize = (480,360)
            modRGB = cv2.resize(imgRGB, dsize=resize, interpolation=cv2.INTER_LINEAR) / 255
            modDepth = cv2.resize(imgDep, dsize=resize, interpolation=cv2.INTER_NEAREST) / 255
            modGT = cv2.resize(imgGT, dsize=resize, interpolation=cv2.INTER_NEAREST)
            # print(modGT.shape)
            # print(modGT[0][0])
            modGT = cv2.cvtColor(modGT, cv2.COLOR_BGR2RGB)
            modGT = self.mask_to_class_rgb(modGT)

            modRGB = modRGB[: , :, 2]
            modDepth = modDepth[: , :, 2]

            imgs = []
            img = {
                'rgb': modRGB,
                'depth': modDepth
            }
            for mod in modalities:
                imgs.append(img[mod].copy())

            return torch.from_numpy(np.array(imgs)).float(), modGT
        except IOError as e:
            print("Error loading " + a, e)
        return False, False, False

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
