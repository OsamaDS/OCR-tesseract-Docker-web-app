import numpy as np
import sys, os
import cv2
import pytesseract
import re
import imutils
from imutils.contours import sort_contours


class OCR:
    def License_read_img(self, img):
        vert = []
        text = []
        #pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
        #tessdata_dir_config = r'--tessdata-dir "/usr/local/share/tessdata/configs"'
        img_copy = img.copy()
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        img_copy = cv2.GaussianBlur(img_copy, (5, 5), 0)
        img_canny = cv2.Canny(img_copy, 50, 200, apertureSize=3)
        contours, hierarchy = cv2.findContours(img_canny.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            if w > 50 and h > 50 and h < 300 and w < 400:
                vert.append([x, y, w, h])

        for v in vert:
            x, y, w, h = v
            cropped_img = img_copy[y:y + h, x:x + w]
            ret, thresh = cv2.threshold(cropped_img, 127, 255, cv2.THRESH_OTSU)
            #print(ret)
            text.append(pytesseract.image_to_string(thresh, lang='eng'))

        for i, t in enumerate(text):
            res = t.replace('\n', '\n').replace('@', '0')
            res = res.lstrip()
            # print(res)
            text[i] = res
        text = [x.rstrip() for i, x in enumerate(text) if x not in text[:i] if x]

        img_copy = img.copy()
        col_slice = int(np.round(img_copy.shape[1] * .35))
        img_copy = img_copy[:col_slice]
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        img_copy = cv2.GaussianBlur(img_copy, (5, 5), 0)
        name_and_address = pytesseract.image_to_string(img_copy, lang='eng')
        name_and_address = name_and_address.split('\n')
        name_and_address = [x.strip() for x in name_and_address]
        name_and_address = [x for x in name_and_address if x]
        #name_and_address
        #print(text)

        return name_and_address, text
    
    def ID_read_img(self, img):
        #pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
        y_limit = []
        text = []
        previous_ind = 0
        img_copy = img.copy()
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        img_copy = cv2.GaussianBlur(img_copy, (5, 5), 0)
        img_canny = cv2.Canny(img_copy, 50, 200, apertureSize=3)
        contours, hierarchy = cv2.findContours(img_canny.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            if w > 500:
                y_limit.append(y)

        y_limit = sorted(y_limit)
        y_limit = [x for i, x in enumerate(y_limit) if x not in y_limit[:i]]
        cropped_img = img.copy()

        for i, y in enumerate(y_limit):
            if previous_ind == 0:
                temp = cropped_img[:y + 8, :]
                temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
                temp = cv2.GaussianBlur(temp, (5, 5), 0)
                res = pytesseract.image_to_string(temp, lang='eng')
                text.append(res)
            else:
                temp = cropped_img[previous_ind:y + 8, :]
                temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
                temp = cv2.GaussianBlur(temp, (5, 5), 0)
                res = pytesseract.image_to_string(temp, lang='eng')
                text.append(res)
            previous_ind = y

        return text

    def passport_read_img(self, img):
        # load the input image, convert it to grayscale, and grab its
        # dimensions
        image = img.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (H, W) = gray.shape
        # initialize a rectangular and square structuring kernel
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        # smooth the image using a 3x3 Gaussian blur and then apply a
        # blackhat morpholigical operator to find dark regions on a light
        # background
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
        grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad = np.absolute(grad)
        (minVal, maxVal) = (np.min(grad), np.max(grad))
        grad = (grad - minVal) / (maxVal - minVal)
        grad = (grad * 255).astype("uint8")
        grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
        thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
        thresh = cv2.erode(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sort_contours(cnts, method="bottom-to-top")[0]
        mrzBox = None
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            percentWidth = w / float(W)
            percentHeight = h / float(H)
            if percentWidth > 0.8 and percentHeight > 0.04:
                mrzBox = (x, y, w, h)
                break
        if mrzBox is None:
            print("[INFO] MRZ could not be found")
        (x, y, w, h) = mrzBox
        mrz = image[y:y + h, x:x + w]
        #pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
        res = pytesseract.image_to_string(mrz, lang='eng')

        return res


    

