import os
import argparse

from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg


def get_args():

    parser = argparse.ArgumentParser(description='train Mask R-CNN (detectron2)')
    parser.add_argument('--gpu', type=str, default='0', help='ID of GPU (single GPU)')
    parser.add_argument('--dataset', type=str, required=True, help='dataset path')
    parser.add_argument('--json', type=str, required=True, help='coco format json file for dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='save model in this directory')
    args = parser.parse_args()

    return args


def set_config(checkpoint, dataset, json):

    register_coco_instances("Augmentedleaves", {}, json, dataset)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("Augmentedleaves_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0002
    cfg.SOLVER.MAX_ITER = 50000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.OUTPUT_DIR = checkpoint
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


if __name__ == "__main__":

    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg = set_config(args.checkpoint, args.dataset, args.json)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
