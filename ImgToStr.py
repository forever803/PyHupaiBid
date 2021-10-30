from pytesser.pytesser import *
from PIL import Image, ImageEnhance
import pytesseract
import cv2, imutils
import pyautogui as auto
import numpy as np


def run1():
    image = Image.open("example.jpg")

    print(image_to_string(image))
    # enhancer = ImageEnhance.Contrast(image)
    # image_enhancer = enhancer.enhance(4)

    # print(image_to_string(image_enhancer))

def run2():
    pytesseract.pytesseract.tesseract_cmd = r'./pytesser/tesseract.exe'
    image = Image.open("example.jpg")
    r = pytesseract.image_to_string(image)
    print(r)

def run3():
    # 读取输入图片
    image = cv2.imread("example.jpg")
    # 将输入图片裁剪到固定大小
    image = imutils.resize(image, height=200)
    # 将输入转换为灰度图片
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('edge.png', gray)
    # rawImage = Image.open("edge.png")
    rawImage = Image.fromarray(gray)
    ret = image_to_string(rawImage)
    print("xxx{}bbb".format(ret))
    px = int("".join(list(filter(str.isdigit, ret.split('\n')[0]))))
    print("xxx{}bbb".format(px))

def run4():
    # region = (起始x, 起始y, 长, 宽)
    price_img = auto.screenshot(region=(1440, 200, 100, 20))
    gray = cv2.cvtColor(np.array(price_img), cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('edge.png', imutils.resize(gray, height=200))
    # cv2.imwrite('edge.png', gray)
    rawImage = Image.fromarray(gray)
    ret = image_to_string(rawImage)
    print("xxx{}bbb".format(ret))
    px = int("".join(list(filter(str.isdigit, ret.split('\n')[0]))))
    print("px:￥{}.".format(px))

if __name__=='__main__':
    run4()
    pass