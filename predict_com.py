from paddleocr import PaddleOCR, draw_ocr
import math
import cv2
import sys
import os
import numpy as np

corepath = os.path.abspath("../ultralytics")
sys.path.append(corepath)
from ultralytics import YOLO

# print(sys.path)


def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def angle_between_points(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    if x1 == x2:
        return 90
    elif y1 == y2:
        return 0
    else:
        angle = math.degrees(math.atan((y2 - y1) / (x2 - x1)))
        return angle


def Nrotate(angle, valuex, valuey, pointx, pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    nRotatex = (valuex - pointx) * math.cos(angle) - (valuey - pointy) * math.sin(angle) + pointx
    nRotatey = (valuex - pointx) * math.sin(angle) + (valuey - pointy) * math.cos(angle) + pointy
    return nRotatex, nRotatey


def rotation(img, degree):
    # degree左转
    height, width = img.shape[:2]
    heightNew = int(
        width * math.fabs(math.sin(math.radians(degree))) + height * math.fabs(math.cos(math.radians(degree))))
    widthNew = int(
        height * math.fabs(math.sin(math.radians(degree))) + width * math.fabs(math.cos(math.radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    return imgRotation


def recognize_text(img_path):
    ocr = PaddleOCR(det_model_dir='./output/det_db_inference1/', rec_model_dir='./inference/en_shuibiao-OCRv4_rec/',
                    rec_char_dict_path='./ppocr/utils/shuibiao_dict.txt', cls_model_dir='./output/cls_inference/',
                    use_angle_cls=True)
    result = ocr.ocr(img_path, cls=True)
    return result

def recognize_ocr(img):
    ocr = PaddleOCR(det_model_dir='./ch_PP-OCRv3_det_infer', rec_model_dir='./ch_PP-OCRv4_rec_infer')
    result = ocr.ocr(img)
    return result



if __name__ == '__main__':
    path = 'E:\\test\\ocrtest.png'
    path = cv2.imread(path)
    res = recognize_ocr(path)
    print(res)
