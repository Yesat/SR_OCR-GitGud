import cv2
import pytesseract as ptes
import numpy as np
from matplotlib import pyplot as plt
import re

ptes.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def thresholding(image):
    return cv2.threshold(image, 140, 255, cv2.THRESH_BINARY)[1]


def shearing(img):
    shear = -2
    rows, cols, ch = img.shape

    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

    pt1 = 5+shear
    pt2 = 20+shear

    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

    shear_M = cv2.getAffineTransform(pts1, pts2)

    img = cv2.warpAffine(img, shear_M, (cols, rows),
                         borderMode=cv2.BORDER_CONSTANT, borderValue=[255, 255, 255])
    return img

def resizeImage(image, scaleTouple):
    width = int(image.shape[1] * scaleTouple[0])
    height = int(image.shape[0] * scaleTouple[1])
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_LINEAR )
    return resized

def pre_proc(img):
    SIZE_PEAK = (40, 25)
    origin_options =  [(897,460),(897, 515), (897, 570), (897, 625)]
    origin_peak = {}

    origin_peak['t'],origin_peak['d'], origin_peak['s'] = origin_options[1:4] if oq_check(img) else origin_options[0:3]

    peak_img = {}
    results = {}
    for x in origin_peak:
        ori = origin_peak[x]
        cut = img[ori[1]:ori[1]+SIZE_PEAK[1], ori[0]:ori[0]+SIZE_PEAK[0]]
        rez = resizeImage(cut, (2.1, 2.1)) #Tesseract works better by rescaling the image. 
        bord = cv2.copyMakeBorder(
            rez, 5, 1, 1, 1, cv2.BORDER_CONSTANT, value=[255, 255, 255]) # For safety before the shearing adding a small border prevent the text to be cut off
        transf = shearing(bord) # Blizzard slanted font is not playing well with Tesseract, so a simple shearing help.
        thresh = thresholding(transf) # Due to the resolution, thresholding gives better results
        peak_img[x]=transf #for easy check
        results[x]=thresh
        cv2.imshow('transf',transf)
        cv2.waitKey(3000)
    return [results,peak_img]

def sr_ocr(imgs):
    results={}
    for r in imgs:
        img = imgs[r]
        config = (
            "-l eng --oem 1 --psm {} -c tessedit_char_whitelist=0123456789".format(11))
        res = ptes.image_to_string(img, config=config)
        # print(res)
        res_clean = ''.join(e for e in res if e.isdigit())
        if res_clean == '':
            res_clean = 'NoSR'
        results[r]=res_clean
        print(res_clean)
    return results   

def oq_check(img): # As the Open Queue box doesn't appear if you don't play it, simple OCR check if the text is present in the highest SR Box
    origin = (460,540)
    size = (45,110)
    
    cut = img[origin[0]: origin[0]+size[0], origin[1]: origin[1]+size[1]]
    
    res = ptes.image_to_string(cut)
    # print(res)
    if re.match('OPEN QUEUE',res):
        print('foo')
        return True
    else:
        print('bar')
        return False
    
def main():
    #TODO: Automate the pass on images
    img = cv2.imread('Overwatch.png')
    img2 = cv2.imread('Overwatch_oq.jpeg')
    
    preproc, peak_img = pre_proc(img)
    
    # print(preproc.keys())
    results1 = sr_ocr(preproc)
    preproc, peak_img = pre_proc(img2)

    # print(preproc.keys())
    
    results2=sr_ocr(preproc)


if __name__ == '__main__':
    	main()
