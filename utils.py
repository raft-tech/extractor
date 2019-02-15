import copy
import json

import cv2
import imutils
import numpy as np


def cut_polygon_deskew(image, org_pts, is_gray=False):
    pts = np.array(org_pts)

    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    W = rect[1][0]
    H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1, x2, y1, y2 = min(Xs), max(Xs), min(Ys), max(Ys)

    rotated = False
    angle = rect[2]
    if angle < -60:
        angle += 90
        rotated = True

    center = (int((x1+x2)/2), int((y1+y2)/2))
    size = (int(1*(x2-x1)), int(1*(y2-y1)))

    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

    cropped = cv2.getRectSubPix(image, size, center)
    cropped = cv2.warpAffine(cropped, M, size)

    croppedW = W if not rotated else H
    croppedH = H if not rotated else W

    crop_rotated = cv2.getRectSubPix(
        cropped, (int(croppedW), int(croppedH)), (size[0]/2, size[1]/2))
    if is_gray:
        crop_rotated = cv2.cvtColor(crop_rotated, cv2.COLOR_RGB2GRAY)
    return crop_rotated


def save_json(data, path):
    with open(path, 'w') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)


def imread_buffer(buffer_):
    image = np.frombuffer(buffer_, dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def pprint(raw_dict):
    return json.dumps(copy.deepcopy(raw_dict), indent=4, ensure_ascii=False)