import cv2
import os
import shutil
from copy import deepcopy


def clahe(img, clipLimit=3, gridsize=(64, 64), mode='lab'):
    if mode == 'lab':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit, gridsize)
        img_planes = cv2.split(img)
        img_planes[0] = clahe.apply(img_planes[0])
        img = cv2.merge(img_planes)
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    elif mode == 'hsv':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit, gridsize)
        img_planes = cv2.split(img)
        img_planes[2] = clahe.apply(img_planes[2])
        img = cv2.merge(img_planes)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    return img


def equalizehist(img, mode='lab'):
    if mode == 'lab':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    elif mode == 'hsv':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    return img


if __name__ == '__main__':
    img_path = input('test image file\'s path')
    ori_img = cv2.imread(img_path)

    if not os.path.exists('./test/clahe'):
        os.makedirs('test/clahe')
    else:
        shutil.rmtree('test/clahe')
        os.makedirs('test/clahe')

    img = deepcopy(ori_img)
    equa_img = equalizehist(img, 'hsv')
    cv2.imwrite('test/clahe/test_equalize_hsv.jpg', equa_img)
    img = deepcopy(ori_img)
    equa_img = equalizehist(img, 'lab')
    cv2.imwrite('test/clahe/test_equalize_lab.jpg', equa_img)

    for cliplimit in range(1, 10):
        img = deepcopy(ori_img)
        clahe_img = clahe(img, cliplimit, (64, 64), 'lab')
        cv2.imwrite(f'./test/clahe/test_clahe_lab_{cliplimit}.jpg', clahe_img)
        img = deepcopy(ori_img)
        clahe_img = clahe(img, cliplimit, (64, 64), 'hsv')
        cv2.imwrite(f'./test/clahe/test_clahe_hsv_{cliplimit}.jpg', clahe_img)

    cv2.imwrite(f'./test/clahe/test_clahe_ori.jpg', ori_img)