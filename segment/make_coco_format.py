import os
import argparse
from PIL import Image  # (pip install Pillow)
import numpy as np  # (pip install numpy)
from skimage import measure  # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon  # (pip install Shapely)
import json
import collections as cl
import datetime
import glob
from tqdm import tqdm


def create_sub_masks(mask_image):

    width, height = mask_image.size
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x, y))
            # if pixel != 0:
            if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                    # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width + 2, height + 2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x + 1, y + 1), 1)

    return sub_masks


def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        try:
            polygons.append(poly)
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            if len(segmentation) == 0:
                continue
            segmentations.append(segmentation)

        except AttributeError:
            segs = []
            for pol in poly:
                seg = np.array(pol.exterior.coords).ravel().tolist()
                if len(seg) == 0:
                    continue
                segs.append(seg)

            x, y, max_x, max_y = poly.bounds
            width = max_x - x
            height = max_y - y
            bbox = (x, y, width, height)
            area = poly.area

            annotation = {
                    'segmentation': segs,
                    'iscrowd': is_crowd,
                    'image_id': image_id,
                    'category_id': category_id,
                    'id': annotation_id,
                    'bbox': bbox,
                    'area': area
                    }
            
            return annotation
    
    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    if multi_poly.is_empty:
        return None
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation


def info():
    tmp = cl.OrderedDict()
    tmp["description"] = "Leaf Segmentation Dataset"
    tmp["url"] = "none"
    tmp["version"] = "0.0"
    tmp["year"] = 2020
    tmp["contributor"] = "contributor"
    tmp["data_created"] = "2020/10/01"
    return tmp


def licenses():
    tmp = cl.OrderedDict()
    tmp["id"] = 1
    tmp["url"] = "none"
    tmp["name"] = "administrater"
    return tmp


def images(rgb_images):
    tmps = []
    for i, image_name in enumerate(tqdm(rgb_images)):
        image = Image.open(image_name)
        w, h = image.size
        tmp = cl.OrderedDict()
        tmp["license"] = 1
        tmp["id"] = i + 1
        tmp["file_name"] = image_name
        tmp["width"] = w
        tmp["height"] = h
        tmp["date_captured"] = str(datetime.datetime.now())
        tmp["coco_url"] = ""
        tmp["flickr_url"] = ""
        tmps.append(tmp)
    return tmps


def annotation(label_images):
    annotations = []
    annotation_id = 1
    for i, image_name in enumerate(tqdm(label_images)):
        plant_image = Image.open(image_name)
        # bottle_book_mask_image = Image.open('/path/to/images/bottle_book_mask.png')
        width, height = plant_image.size
        p_l = []
        for x in range(width):
            for y in range(height):
                # pixel = plant_image.getpixel((x, y))[:3]
                # if pixel != (0,0,0):
                pixel = plant_image.getpixel((x, y))
                if pixel != (0, 0, 0):
                    p_l.append(pixel)

        mask_images = [plant_image]  # , bottle_book_mask_image]

        # Define which colors match which categories in the images
        # houseplant_id, book_id, bottle_id, lamp_id = [1, 2, 3, 4]
        leaf_id = 1
        dic = {}
        for col_id in set(p_l):
            dic[str(col_id)] = leaf_id
        category_ids = {i + 1: dic}

        is_crowd = 0

        # These ids will be automatically increased as we go
        image_id = i + 1

        # Create the annotations
        for mask_image in mask_images:
            sub_masks = create_sub_masks(mask_image)
            for color, sub_mask in sub_masks.items():
                sub_mask = np.array(sub_mask)
                category_id = category_ids[image_id][color]
                annotation = create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd)
                if annotation is None:
                    continue
                annotations.append(annotation)
                annotation_id += 1

    return annotations


def categories():
    tmps = []
    tmp = cl.OrderedDict()
    tmp["id"] = 1
    tmp["supercategory"] = "leaf"
    tmp["name"] = "leaf"
    tmps.append(tmp)
    return tmps


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='make coco format from custom dataset')
    parser.add_argument('--dataset', type=str, required=True, help='dataset path (img/annotation)')
    parser.add_argument('--save_dir', type=str, required=True, help='save in this directory')
    args = parser.parse_args()

    rgb_files = glob.glob(os.path.join(args.dataset, 'img', '*.png'))
    rgb_files.sort()
    label_files = glob.glob(os.path.join(args.dataset, 'annotation', '*.png'))
    label_files.sort()

    plant_coco = {}
    plant_coco['info'] = info()
    plant_coco['licenses'] = [licenses()]
    plant_coco['images'] = images(rgb_files)
    plant_coco['annotations'] = annotation(label_files)
    plant_coco['categories'] = categories()

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "coco_format.json"), 'w') as f:
        json.dump(plant_coco, f, indent=2)
