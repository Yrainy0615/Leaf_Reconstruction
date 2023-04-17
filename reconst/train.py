import os
import argparse
import torch
from network.trainer import Trainer


def get_args():

    parser = argparse.ArgumentParser(description='train leaf reconstruction from a RoI')
    parser.add_argument('--gpu', type=str, default='0', help='ID of GPU (single GPU)')

    parser.add_argument('--dataset', type=str, required=False, default='/home/yang/Desktop/konishiike/code/segment/test/crop_sam/001',help='dataset path (img/mask/seg)')
    parser.add_argument('--obj', type=str, required=False,default='/home/yang/Desktop/konishiike/data/synthetic/leaf.obj', help='initial object path')
    parser.add_argument('--base_x', type=int, default=1024, help='width of bush image')
    parser.add_argument('--base_y', type=int, default=768,help='height of bush image')
    parser.add_argument('--size', type=int, default=256,required=False, help='size of cropped RoI')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--threads', default=4, type=int)

    parser.add_argument('--save_freq', type=int, default=50, help='save mean shape & model')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/', help='save in this directory')
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.makedirs(args.checkpoint, exist_ok=True)
    device = torch.device('cuda:0')
    trainer = Trainer(args, device)
    trainer.train()
