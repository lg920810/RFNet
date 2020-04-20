import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from pycocotools import mask
from itertools import groupby
import json
import os
import cv2


def resize_binary_mask(array, new_size):
    image = cv2.resize(array, new_size)
    return image.astype(np.uint8)


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def binary_mask_to_polygon(binary_mask, tolerance=0):
    polygons = []

    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    return polygons


def create_annotation_info(annotation_id, image_id, category_id, is_crowd, binary_mask,
                           image_size=None, tolerance=0, bounding_box=None):
    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask, image_size)
    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask))

    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None
    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)
    if is_crowd:
        segmentation = binary_mask_to_rle(binary_mask)
    else:
        segmentation = binary_mask_to_polygon(binary_mask, tolerance)
        if not segmentation:
            return None

    annotation_info = {
        'id': annotation_id,
        'image_id': image_id,
        'category_id': category_id,
        'iscrowd': is_crowd,
        'area': area.tolist(),
        'bbox': bounding_box.tolist(),
        'segmentation': segmentation,
        'width': binary_mask.shape[1],
        'height': binary_mask.shape[0]
    }

    return annotation_info


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='格式转换工具——语义分割标注转实例分割标注')
    parser.add_argument('--root_dir', type=str, default=r'E:\legal\FL-challenge\FL2018-Valid',
                        help='root directory')
    parser.add_argument('--mask_filename', type=str, default='mask',
                        help='filename of mask')
    parser.add_argument('--classes', type=str, default='classes.txt',
                        help='need a txt file, for example classes.txt')
    parser.add_argument('--type', type=str, default='valid',
                        help='choose train or valid annotations, defaule is train')

    args = parser.parse_args()
    print('****************************************************')
    print('root directory: ', args.root_dir)
    print('annotations type:', args.type)

    mask_dir = os.path.join(args.root_dir, args.mask_filename)
    annotation_dir = os.path.join(args.root_dir, 'annotations')
    if not os.path.exists(annotation_dir):
        os.mkdir(annotation_dir)

    classes_file = os.path.join(args.root_dir, args.classes)

    INFO = {
        'description': '',
        'url': '',
        'version': '',
        'contributor': 'Ge Li',
        'year': 2019
    }

    coco_output = {
        'info': INFO,
        'categories': [],
        'images': [],
        'annotations': []
    }
    class_dict = {128: 1, 191: 2, 255: 3}

    with open(classes_file) as f:
        classes = f.read().strip().split()
    for index, cls in enumerate(classes):
        coco_output['categories'].append({'id': index + 1, 'name': cls, 'supercategory': 'mark'})

    annotation_id = 1  # 实例个数，从1开始计数，一直递增
    image_id = 0
    for id_index, image_name in enumerate(os.listdir(mask_dir)):
        image = cv2.imread(os.path.join(mask_dir, image_name), cv2.IMREAD_GRAYSCALE)
        height, width = image.shape[0], image.shape[1]

        if np.sum(image) != 0:
            image_id = image_id + 1
            coco_output['images'].append({'file_name': image_name,
                                          'id': image_id,
                                          'width': width,
                                          'height': height})

            color_list = np.unique(image)  # 从小到大排序

            masks = np.zeros((height, width, len(color_list)), dtype=np.uint8)
            for i, color in enumerate(color_list):
                if color == 0:
                    continue
                category_id = class_dict[color]
                masks[image == color, i] = 255

                _, contours, hierarch = cv2.findContours(masks[:, :, i].astype(np.uint8),
                                                         cv2.RETR_LIST,
                                                         cv2.CHAIN_APPROX_NONE)

                for j in range(len(contours)):
                    binary_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.drawContours(binary_mask,
                                     contours=contours,
                                     contourIdx=j,
                                     color=(1, 1, 1),
                                     thickness=-1)

                    annotation_info = create_annotation_info(annotation_id=annotation_id,
                                                             image_id=image_id,
                                                             category_id=category_id,
                                                             is_crowd=0,
                                                             binary_mask=binary_mask)

                    coco_output['annotations'].append(annotation_info)
                    annotation_id = annotation_id + 1

    with open('{}/instances_{}.json'.format(annotation_dir, args.type), 'w') as output_file:
        json.dump(coco_output, output_file, indent=4)
