import os
import argparse
import glob

import cv2
import numpy as np

from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer


def get_args():

    parser = argparse.ArgumentParser(description='test Mask R-CNN (detectron2)')
    parser.add_argument('--gpu', type=str, default='0', help='ID of GPU (single GPU)')
    parser.add_argument('--dataset', type=str, required=True, help='dataset path')
    parser.add_argument('--checkpoint', type=str, required=True, help='trained model')
    parser.add_argument('--save_dir', type=str, default='./test/', help='save result in this directory')

    args = parser.parse_args()

    return args


def set_config(checkpoint, dataset):

    # register_coco_instances("Leaves", {}, json, dataset)
    # meta_data = MetadataCatalog.get("Leaves")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = checkpoint
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.DATASETS.TEST = ("Leaves", )
    cfg.TEST.DETECTIONS_PER_IMAGE = 500

    return cfg


def segmentation(predictor, data_list, mask_dir):

    for d in data_list:
        im_name = os.path.splitext(os.path.basename(d))[0]
        im = cv2.imread(d)
        outputs = predictor(im)

        # if you want to visualize result on one image,
        # use the following code or concat each masks (outputs of this code)
        # v = Visualizer(im[:, :, ::-1],
        #                metadata=meta_data,
        #                instance_mode=ColorMode.IMAGE_BW
        #                )
        #
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imwrite(os.path.join(result_dir, "{}.png".format(im_name)), out.get_image()[:, :, ::-1])

        mask = outputs["instances"].pred_masks.detach().cpu().numpy() * 255
        os.makedirs(os.path.join(mask_dir, im_name), exist_ok=True)
        for i in range(mask.shape[0]):
            cv2.imwrite(os.path.join(mask_dir, im_name, "{:03}.png".format(i+1)), mask[i].astype(np.uint8))


if __name__ == "__main__":

    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    data_list = glob.glob(os.path.join(args.dataset, "*.png"))
    data_list.sort()
    # result_dir = os.path.join(args.save_dir, "pred")
    # os.makedirs(result_dir, exist_ok=True)
    mask_dir = os.path.join(args.save_dir, "inst")
    os.makedirs(mask_dir, exist_ok=True)

    cfg = set_config(args.checkpoint, args.dataset)
    predictor = DefaultPredictor(cfg)
    segmentation(predictor, data_list, mask_dir)
