import os.path
import torchvision.transforms.functional as tf
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile
from basicsr.utils import tensor2img
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import cv2
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import pdb
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def mask_bboxregion_coordinate(mask):
    w, h = np.shape(mask)[:2]
    valid_index = np.argwhere(mask == 255)  # [length,2]
    if np.shape(valid_index)[0] < 1:
        x_left = 0
        x_right = 0
        y_bottom = 0
        y_top = 0
    else:
        x_left = np.min(valid_index[:, 0])
        x_right = np.max(valid_index[:, 0])
        y_bottom = np.min(valid_index[:, 1])
        y_top = np.max(valid_index[:, 1])

    return x_left, x_right, y_bottom, y_top

def torch_mask_bboxregion_coordinate(mask):

    valid_index = torch.argwhere(mask == 1)  # [length,2]
    if valid_index.shape[0] < 1:
        x_left = 0
        x_right = 0
        y_bottom = 0
        y_top = 0
    else:
        x_left = torch.min(valid_index[:, 0])
        x_right = torch.max(valid_index[:, 0])
        y_bottom = torch.min(valid_index[:, 1])
        y_top = torch.max(valid_index[:, 1])

    return x_left, x_right, y_bottom, y_top


def findContours(im):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4
    Returns:
        contours, hierarchy
    """
    img = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)  # PIL转cv2
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if cv2.__version__.startswith('4'):
    #     contours, hierarchy = cv2.findContours(*args, **kwargs)
    # elif cv2.__version__.startswith('3'):
    #     _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    # else:
    #     raise AssertionError(
    #         'cv2 must be either version 3 or 4 to call this method')
    return contours, hierarchy



class COCOARTDataset(Dataset):
    """A template dataset class for you to implement custom datasets."""

    def __init__(self, load_size_H,load_size_W, is_for_train,train_root='',mask_root='',style_root='',test_root='',compare_num=0):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """

        # path
        self.train_root=train_root
        self.mask_root=mask_root
        self.style_root=style_root

        self.test_root=test_root

        # content and mask
        self.path_img = []
        self.path_mask = []
        # style
        self.path_style = []
        self.tag_style = []
        self.path_name=[]

        self.load_size_H = load_size_H
        self.load_size_W = load_size_W
        self.isTrain = is_for_train
        self.compare_num=compare_num

        self._load_images_paths()


    def _load_images_paths(self):
        if self.isTrain:
            path = self.train_root+'/'
            self.paths_all = list(Path(path).glob('*'))[4000:]

            print('loading training set')
            for path in self.paths_all:
                path_mask = str(path).replace(self.train_root, self.mask_root, 1).replace('jpg', 'png')
                if not os.path.exists(path_mask):
                    continue
                else:
                    img0 = Image.open(str(path))
                    w = img0.width
                    h = img0.height
                    if w >= 480 and h >= 480:
                        self.path_img.append(str(path))
                        self.path_mask.append(path_mask)


            style_csv=self.style_root+'/style_train.csv'
            for f in open(style_csv):

                temp_path = f.split(',')[0]
                style_tag = temp_path.split('/')[0]

                new_path = self.style_root + '/'+temp_path

                if not os.path.exists(new_path):
                    print('wikiart image not exist',new_path)
                    continue
                else:
                    img0 = Image.open(str(new_path))
                    w = img0.width
                    h = img0.height
                    if w >= 512 and h >= 512:
                        self.path_style.append(new_path)
                        self.tag_style.append(style_tag)

            print('foreground number', len(self.path_img))
            print('background number', len(self.path_style))


        else:
            print('loading testing set')
            content_path_origin =self.test_root+'/comp/'
            content_path_all = list(Path(content_path_origin).glob('*'))


            for content_path in content_path_all:
                mask_path = str(content_path).replace('comp', 'mask')
                #style_path = str(content_path).replace('comp', 'style')

                if os.path.exists(mask_path):
                    self.path_img.append(str(content_path))
                    self.path_mask.append(mask_path)
                    #self.path_style.append(style_path)

                    new_path_name=str(content_path).replace(content_path_origin,'').replace('.jpg','')
                    self.path_name.append(new_path_name)


            print('testing number', len(self.path_img))
      


    def select_mask(self, index):

        mask = Image.open(self.path_mask[index]).convert('L')
        mask_array = np.array(mask)

        mask_value = np.unique(np.sort(mask_array[mask_array > 0]))

        if self.isTrain:
            random_pixel = random.choice(mask_value)

        else:
            random_pixel = 255

        if random_pixel != 255:
            mask_array[mask_array == 255] = 0
        mask_array[mask_array == random_pixel] = 255

        mask_array[mask_array != 255] = 0

        return mask_array

    def select_mask_test(self, index):

        mask = Image.open(self.path_mask[index]).convert('L')
        mask_array = np.array(mask)

        return mask_array

    def __getitem__(self, index):

        c_index = index % len(self.path_img)

        if self.isTrain:
            path_name=''
            style = Image.open(self.path_style[index]).convert('RGB')
            style = style.resize([self.load_size_W, self.load_size_H],resample=Image.Resampling.LANCZOS)
            content = Image.open(self.path_img[c_index]).convert('RGB')
            select_mask = self.select_mask(c_index)
        else:
            path_name=self.path_name[index]
            content = Image.open(self.path_img[index]).convert('RGB')
            select_mask = self.select_mask_test(index)



        np_mask = np.uint8(select_mask)
        mask = Image.fromarray(np_mask, mode='L')

        content = content.resize([self.load_size_W, self.load_size_H],resample=Image.Resampling.LANCZOS)
        mask = mask.resize([self.load_size_W, self.load_size_H],resample=Image.Resampling.NEAREST)

        

        content = tf.to_tensor(content)
        mask = tf.to_tensor(mask)

        if self.isTrain:
            style = tf.to_tensor(style)
            comp = content * mask + style * (1 - mask)
            style = style * 2 - 1
        else:
            comp=content
            style=0

        comp = comp * 2 - 1

        style_comparison = []
        compare_num = self.compare_num
        if compare_num>0:
            total_length = len(self.path_style)

            count = 0
            while count < compare_num:
                random_index = random.randint(0, total_length - 1)
                if self.tag_style[random_index] != self.tag_style[index] :
                    style_comparison.append(Image.open(self.path_style[random_index]).convert('RGB'))
                    count += 1

            for i in range(compare_num):
                style_comparison[i] = style_comparison[i].resize([self.load_size_W, self.load_size_H],
                                                                 resample=Image.Resampling.LANCZOS)

                style_comparison[i] = tf.to_tensor(style_comparison[i]).unsqueeze(0)

            style_comparison = torch.cat(style_comparison, dim=0)

            style_comparison = style_comparison * 2 - 1

        return {
            'comp': comp, 'mask': mask, 'style': style,'text':"", 'style_comparison':style_comparison,'path_name':path_name
        }

    def __len__(self):
        if self.isTrain:
            return len(self.path_style)
        else:
            return len(self.path_img)
        


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True