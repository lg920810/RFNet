import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def dice_coef_numpy(seg, gt, k=1):
    # dice = 2.0 * (np.sum(seg & gt)) / (np.sum(seg) + np.sum(gt))
    dice = 2.0 * np.sum(seg[gt == k] == k) / (np.sum(seg[seg == k] == k) + np.sum(gt[gt == k] == k))
    return dice


def convex_hull(image):

    cup = np.zeros_like(image, dtype=np.uint8)
    disc = np.zeros_like(image, dtype=np.uint8)

    cup[image == 0] = 255
    disc[image == 128] = 255

    cup_copy, contours, hiearachy= cv2.findContours(cup, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(contours[0], returnPoints=True)
    cv2.fillConvexPoly(cup_copy, hull, (255, 255, 255))

    disc_copy, contours, hiearachy = cv2.findContours(disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(contours[0], returnPoints=True)
    cv2.fillConvexPoly(disc_copy, hull, (255, 255, 255))

    mask = 255 * np.ones_like(image, dtype=np.uint8)
    mask[disc_copy == 255] =  128
    mask[cup_copy == 255] = 0

    # plt.subplot(121)
    # plt.imshow(image, cmap='gray')
    # plt.subplot(122)
    # plt.imshow(mask, cmap='gray')
    # plt.show()
    return mask

#dice cup 0.8242202233806537
#dice disc 0.8157182844990123

#dice cup 0.8444208123075633
#dice disc 0.8471048128372444

#dice cup 0.8554208123075633
#dice disc 0.872048128372444

if __name__ == '__main__':
    gt_path = r'E:\legal\REFUGE-challenge\REFUGE-Valid\mask'
    # pred_path = r'E:\legal\REFUGE-challenge\MNet_result\REFUGE_result'
    pred_path = 'result'

    dice_cup = []
    dice_disc = []
    for name in os.listdir(gt_path):
        gt = cv2.imread(os.path.join(gt_path, name), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(os.path.join(pred_path, name), cv2.IMREAD_GRAYSCALE)
        pred = convex_hull(pred)
        dice_cup.append(dice_coef_numpy(pred, gt, k=0))
        dice_disc.append(dice_coef_numpy(pred, gt, k=128))

    print('dice cup', np.sum(dice_cup) / len(dice_cup))
    print('dice disc',np.sum(dice_disc) / len(dice_disc))