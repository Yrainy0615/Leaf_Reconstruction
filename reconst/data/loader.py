import os
from os import listdir
from os.path import join
import json
from  torch.utils.data import DataLoader
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np
def is_image_file(filename):
    # search image files
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


class InstDataset(data.Dataset):

    def __init__(self, data_dir):

        super(InstDataset, self).__init__()
        # please name directories in your dataset as follows or change following each name
        # each directory has image files with the same name
        self.img_path = join(data_dir, 'img')
        self.mask_path = join(data_dir, 'mask')
        self.seg_path = join(data_dir, 'seg')
        self.files = [x for x in listdir(self.img_path) if is_image_file(x)]
        self.files.sort()

        # load json file (RoI position on the base image)
        self.img_json = json.load(open(join(data_dir, "img.json")))

        # transform numpy array to tensor
        transform_list_img = [transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ]
        transform_list_seg = [transforms.ToTensor()]
        self.transform_img = transforms.Compose(transform_list_img)
        self.transform_seg = transforms.Compose(transform_list_seg)

    def __getitem__(self, index):

        img = cv2.imread(join(self.img_path, self.files[index]))
        mask = cv2.imread(join(self.mask_path, self.files[index]), 0)
        seg = cv2.imread(join(self.seg_path, self.files[index]))

        # resize images to reduce calculating cost
        img = cv2.resize(img, (128, 128))
        mask = cv2.resize(mask, (128, 128))
        seg = cv2.resize(seg, (128, 128))
        img = self.transform_img(img)
        seg = self.transform_seg(seg)
        mask = torch.tensor(mask/255, dtype=torch.float)

        leaf_id = os.path.splitext(os.path.basename(self.files[index]))[0]
        u = self.img_json[leaf_id]["x"]
        v = self.img_json[leaf_id]["y"]

        return img, seg, mask, u, v

    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    data_path = '/home/yang/Desktop/konishiike/code/segment/test/crop_sam/001'
    train_data = DataLoader(dataset=InstDataset(data_path),
                                      num_workers=4,
                                      batch_size=1,
                                      shuffle=True)
    for iteration, data in enumerate(train_data, 1):
        print(data[0].shape)
    # batch = next(iter(train_data))
    # seg = batch[1].numpy().squeeze(0)
    # seg  = np.transpose(seg,(1,2,0))
    # mask = batch[2].numpy()
    # mask = np.transpose(mask,(1,2,0))
    # gray_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # plt.subplot(1,2,1)
    # plt.imshow(gray_mask)
    # plt.subplot(1,2,2)
    # plt.imshow(seg)
    # plt.show()