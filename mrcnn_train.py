import matplotlib.pyplot as plt

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import cv2
import numpy as np
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn.mobilenet import *
from mrcnn import visualize


############################################################
#  Configurations
############################################################

class OpticConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "Optic"
    GPU_COUNT = 1
    LEARNING_RATE = 0.0001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background 一个框 分割分为背景+视盘+视杯
    MASK_NUM_CLASSES = 1 + 2

    # Number of training and validation steps per epoch
    TRAIN_IMAGE_NUM = 1029
    STEPS_PER_EPOCH = TRAIN_IMAGE_NUM // IMAGES_PER_GPU + 1
    VAL_IMAGE_IDS = 380
    VALIDATION_STEPS = max(1, VAL_IMAGE_IDS // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0.7

    # Backbone network architecture
    # Supported values are: resnet50, resnet101, mobilenetv1, mobilenetv2, mobilenetv3
    BACKBONE = mobilenetv1
    COMPUTE_BACKBONE_SHAPE = True

    # Mask branch architecture
    # Supported values are: pyramid(origin version), adaptive, fusion.
    MASK_BRANCH = "pyramid"

    # Segmentation loss
    # Supported values are: binary_crossentropy(origin version),
    # binary_crossentropy + dice_loss.
    MASK_BRANCH_LOSS = "dice_loss"

    # Bounding box loss
    # Supported values are: l1_loss (origin version), GIoU_loss
    BBOX_LOSS = 'l1_loss'
    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    # if the object is small, make the scale size smaller
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 500
    POST_NMS_ROIS_INFERENCE = 1000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # image mean (RGB)
    MEAN_PIXEL = np.array([0., 0., 0.])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
    # MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    TRAIN_BN = True
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    # TRAIN_ROIS_PER_IMAGE = 100
    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 28

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [56, 56]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 10

    # Max number of final detections per image
    # DETECTION_MAX_INSTANCES = 3

    # LOSS_WEIGHTS = {
    #     "rpn_class_loss": 1.,
    #     "rpn_bbox_loss": 1.,
    #     "mrcnn_class_loss": 1.,
    #     "mrcnn_bbox_loss": 1.,
    #     "mrcnn_mask_loss": 1.
    # }
    LOSS_WEIGHTS = {
        "rpn_class_loss": 0.,
        "rpn_bbox_loss": 0.,
        "mrcnn_class_loss": 0.,
        "mrcnn_bbox_loss": 0.,
        "mrcnn_mask_loss": 1.
    }


class OpticInferenceConfig(OpticConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.5
    DETECTION_NMS_THRESHOLD = 0.5
    DETECTION_MIN_CONFIDENCE = 0.7


############################################################
#  Dataset
############################################################

class OpticDataset(utils.Dataset):

    def load_data(self, txt_path):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        self.add_class("Optic", 1, "Disc")

        with open(txt_path) as f:
            all_lines = f.readlines()
        for image_id, image_path in enumerate(all_lines):
            if image_path.endswith('\n'):
                image_path = image_path[:-1]
            self.add_image('Optic', image_id=image_id, path=image_path)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = cv2.imread(self.image_info[image_id]['path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        info_temp = info['path'].replace('image', 'mask')
        info = info_temp.replace('jpg', 'bmp')
        # Read mask files from .png image

        mask = cv2.imread(info, cv2.IMREAD_GRAYSCALE)
        h, w = mask.shape
        masks = np.zeros((h, w, config.MASK_NUM_CLASSES), dtype=np.uint8)
        #
        masks[mask == 255, 0] = 1  # bg
        masks[mask == 0, 1] = 1  # cup
        masks[mask == 128, 2] = 1  # disc
        # plt.subplot(121)
        # plt.imshow(masks[:, :, 0], cmap='gray')
        # plt.subplot(122)
        # plt.imshow(masks[:, :, 1], cmap='gray')
        #
        # plt.show()
        # masks = np.stack(masks, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        classID = np.array([1], dtype=np.int32)
        return masks, classID

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "Optic":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, train_path, valid_path):
    """Train the model."""
    # Training dataset.
    dataset_train = OpticDataset()
    dataset_train.load_data(train_path)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = OpticDataset()
    dataset_val.load_data(valid_path)
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0)),
        iaa.Sequential([
            iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
            iaa.WithChannels(0, iaa.Add((50, 100))),
            iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
        ])
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    # print("Train network heads")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=20,
    #             augmentation=augmentation,
    #             layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=120,
                augmentation=augmentation,
                layers='mask')


############################################################
#  Detection
############################################################

def detect(model, txt_path):
    # Read dataset
    dataset = OpticDataset()
    dataset.load_data(txt_path)
    dataset.prepare()
    # Load over images
    for image_id in dataset.image_ids:
        # Load image and run detection
        image_path = dataset.image_info[image_id]['path']
        image_name = image_path.split('/')[-1]
        image = dataset.load_image(image_id)
        r = model.detect([image], verbose=0)[0]

        scores = r['scores']
        masks = r['masks']
        rois = r['rois']

        if len(scores) == 0:
            print(image_name)
            final_mask = 255 * np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.imwrite(os.path.join('result', image_name.split('.')[0] + '.bmp'), final_mask)
            continue
        final_bbox = rois[np.argmax(scores)]
        y1, x1, y2, x2 = final_bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)

        final_mask = masks[:, :, np.argmax(scores)]
        final_mask[final_mask == 0] = 255
        final_mask[final_mask == 2] = 128
        final_mask[final_mask == 1] = 0

        plt.subplot(121)
        plt.imshow(image)
        plt.subplot(122)
        plt.imshow(final_mask, cmap='gray')
        plt.show()
        # print(image_name)
        # cv2.imwrite(os.path.join('result', image_name.split('.')[0] + '.bmp'),
        #             cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


############################################################
#  Command Line
############################################################
def dice_coef_numpy(seg, gt, k=1):
    # dice = 2.0 * (np.sum(seg & gt)) / (np.sum(seg) + np.sum(gt))
    dice = 2.0 * np.sum(seg[gt == k] == k) / (np.sum(seg[seg == k] == k) + np.sum(gt[gt == k] == k))
    # print('Dice similarity score is {}'.format(dice))
    return dice


def cal_dice():
    gt_path = r'E:\legal\REFUGE-challenge\REFUGE-Valid\mask'
    pred_path = 'result'

    dice_cup = []
    dice_disc = []
    for name in os.listdir(gt_path):
        gt = cv2.imread(os.path.join(gt_path, name), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(os.path.join(pred_path, name), cv2.IMREAD_GRAYSCALE)

        dice_cup.append(dice_coef_numpy(pred, gt, k=0))
        dice_disc.append(dice_coef_numpy(pred, gt, k=128))

    print('dice cup', np.sum(dice_cup) / len(dice_cup))
    print('dice disc', np.sum(dice_disc) / len(dice_disc))


if __name__ == '__main__':
    #   train or inference
    mode = 'train'

    weights_form = 'logs/optic20200409T2018/mask_rcnn_optic_0101.h5'
    log_dir = 'logs'
    if mode == "train":
        config = OpticConfig()
    else:
        config = OpticInferenceConfig()
    config.display()

    # Create model
    if mode == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=log_dir)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=log_dir)

    model.load_weights(weights_form, by_name=True)
    print('load weights sucessfully')

    dataset_path = r'E:\REFUGE-challenge'
    train_path = 'txtfile/train.txt'
    valid_path = 'txtfile/valid.txt'
    # Train or evaluate
    if mode == "train":
        train(model, train_path, valid_path)
    elif mode == "inference":
        detect(model, valid_path)
        # cal_dice()
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(mode))
