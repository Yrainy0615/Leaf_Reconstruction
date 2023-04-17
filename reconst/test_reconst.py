import os
import argparse

import torch

from network.tester import Tester


def get_args():

    parser = argparse.ArgumentParser(description='test leaf reconstruction from a RoI')
    parser.add_argument('--gpu', type=str, default='6', help='ID of GPU (single GPU)')

    parser.add_argument('--dataset', type=str, required=False, default='/home/yyang/projects/konishiike/code/segment/test/crop_sam/002',help='dataset path')
    parser.add_argument('--obj', type=str, required=False,default='/home/yyang/projects/konishiike/data/synthetic/leaf.obj', help='path to the initial object (same object for training)')
    parser.add_argument('--base_x', type=int, default=1024,help='width of bush image')
    parser.add_argument('--base_y', type=int, default=768,help='height of bush image')
    parser.add_argument('--size', type=int,default=128,required=False)
    parser.add_argument('--threads', default=4, type=int, help='# threads for loading data')

    parser.add_argument('--save_dir', type=str, default='./test/', help='save result in this directory')
    parser.add_argument('--checkpoint', type=str, required=False,default='/home/yyang/projects/konishiike/code/reconst/checkpoint/epoch47.pth', help='trained model path ')

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda:0')

    os.makedirs(os.path.join(args.save_dir, 'obj'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'img'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'mask'), exist_ok=True)

    tester = Tester(args, device)
    tester.test()
