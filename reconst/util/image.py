import cv2
import math
import torch
import numpy as np
from kornia.geometry import rotate
#from pytorch3d.transforms import Rotate

def get_axis(mask):  # mask: (1, H, W)
    #mask = np.transpose(mask)

    mask = mask[0, :, :].clone().detach().to('cpu').numpy() * 255

    _, thresh = cv2.threshold(mask.astype('uint8'), 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1,2)
    cnt = contours[0]
    m = cv2.moments(cnt)
    axis = math.degrees(math.atan2(2 * m['m11'], (m['m20'] - m['m02'])) / 2)

    return axis


def rotation(input, angle):

    angle = torch.tensor(angle).to(input.device)
    transformed1 = rotate(input, angle)
    transformed2 = rotate(input, angle + 180)

    return transformed1, transformed2


def angle_fitting(img, mask, target_mask):
    # img: (1, C, H, W)
    # mask: (1, H, W)
    # target_mask: (1, H, W)
    # trans_img: (1, C, H, W)
    # trans_mask: (1, H, W)
    
    axis = get_axis(mask)
    # target_mask= target_mask.cpu().detach().numpy()
    # target_mask = np.transpose(target_mask,(1,2,0))
    # cv2.imshow('mask',target_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # target_mask = torch.from_numpy(np.transpose(target_mask, (2, 0, 1)).astype('float32') / 255.0).unsqueeze(0)
    # target_mask.squeeze(1)
    target_axis = get_axis(target_mask)
    angle = target_axis - axis
    trans_imgs = rotation(img, angle)
    trans_masks = rotation(mask, angle)

    return trans_imgs, trans_masks, angle
