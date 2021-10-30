from re import T
from pytesser.pytesser import *
from PIL import Image, ImageEnhance
import pytesseract
import cv2, imutils
import pyautogui as auto
import pandas as pd
import numpy as np
import threading
import datetime as dt
import time as tm


class Config(object):
    operation_delay = 0.2
    input_px_seconds = 30                       # 多少秒开始输入价格
    # px_region = (1213, 487, 100, 20)            # 价格所在区域 (右边价格)
    px_region = (600, 712, 100, 20)             # 价格所在区域   (左边价格)
    tm_region = (567, 690, 100, 20)             # 时间所在区域
    target_px_input_pos = (1250, 670)           # 目标价格输入区域
    bid_button_pos = (1400, 670)                # 出价按钮区域
    submit_button_pos = (1140, 825)             # 提价按钮区域


class HPPage(object):
    def __init__(self):
        self.running = True
        self.cur_time = None
        self.cur_px = None
        self.submit_seconds = None
        self.target_price = None
        self.input_px_succ = False
        self.submit_succ = False
        self.refresh_display_tm_dt = None
        self.prices = {}
        self.data = pd.read_csv("hist.csv", index_col="seconds")
        self.data['today'] = np.nan
        # start update px thread
        self.update_px_t = threading.Thread(target=self.update_px, args=())
        self.update_px_t.start()
        # start update tm thread
        self.update_tm_t = threading.Thread(target=self.update_tm, args=())
        self.update_tm_t.start()

    def stop(self):
        self.running = False
        data_df = pd.DataFrame.from_dict(self.prices, orient='index')
        data_df.to_csv("data.csv")

    def update_px(self):
        while self.running:
            try:
                # 截屏
                # region = (起始x, 起始y, 长, 宽)
                price_img = auto.screenshot(region=Config.px_region)
                # 换为灰度图片
                gray = cv2.cvtColor(np.array(price_img), cv2.COLOR_BGR2GRAY)
                # 转换位image obj
                rawImage = Image.fromarray(gray)
                ret = image_to_string(rawImage, temp_scratch_name="temp_px")
                self.cur_px = int("".join(list(filter(str.isdigit, ret.split('\n')[0]))))
                if self.cur_time is not None and self.cur_time.tm_min == 29:
                    self.prices[self.cur_time.tm_sec] = self.cur_px
                    self.data['today'][self.cur_time.tm_sec] = self.cur_px
            except Exception as e:
                print(e)
            finally:
                tm.sleep(0.01)    

    def update_tm(self):
        while self.running:
            try:
                # 截屏
                # region = (起始x, 起始y, 长, 宽)
                price_img = auto.screenshot(region=Config.tm_region)
                # 换为灰度图片
                gray = cv2.cvtColor(np.array(price_img), cv2.COLOR_BGR2GRAY)
                # 转换位image obj
                rawImage = Image.fromarray(gray)
                ret = image_to_string(rawImage, temp_scratch_name="temp_tm")
                tm_str = "".join(list(filter(lambda x: str.isdigit(x) or x == ":", ret.split('\n')[0])))
                display_time = tm.strptime(tm_str, "%H:%M:%S")
                if self.cur_time != display_time:
                    self.cur_time = tm.strptime(tm_str, "%H:%M:%S")
                    self.refresh_display_tm_dt = dt.datetime.now()
                    if self.cur_time.tm_min == 30:
                        self.stop()
                # 计算提交时间和target_price
                self.calc_submit_info()
                # 尝试出价
                self.try_input_px()
                # 尝试提交
                self.try_submit()
            except Exception as e:
                print(e)
            finally:
                tm.sleep(0.01)
        
    def try_input_px(self):        
        cur_seconds = self.get_cur_seconds()
        if cur_seconds >= Config.input_px_seconds and not self.input_px_succ:
            # Input Target Price
            auto.click(Config.target_px_input_pos)
            auto.hotkey('ctrl', 'a')
            auto.press("backspace", interval=0.01)
            auto.typewrite(message=str(self.target_price), interval=0.01)
            tm.sleep(Config.operation_delay)

            # 出价
            auto.click(Config.bid_button_pos)
            self.input_px_succ = True

    def try_submit(self):
        cur_seconds = self.get_cur_seconds()
        if self.submit_seconds is None \
            or cur_seconds < self.submit_seconds \
            or not self.input_px_succ \
            or self.submit_succ: 
            return
        # 提交
        auto.click(Config.submit_button_pos)
        self.submit_succ = True

    def get_cur_seconds(self):
        cur_seconds = -1
        if self.cur_time is not None and self.cur_time.tm_min == 29:
            delta = dt.datetime.now() - self.refresh_display_tm_dt
            cur_seconds = self.cur_time.tm_sec + delta.seconds + delta.microseconds / 1000000.0
        return cur_seconds

    def calc_submit_info(self):
        df = self.data - self.data.iloc[0]
        max_corr = None
        col_name = None
        for col in df.columns:
            if col == "today":
                continue
            corr = df["today"].corr(df[col])
            if corr != np.nan:
                if max_corr is None or max_corr < corr:
                    max_corr = corr
                    col_name = col
        print(col_name)
        # TODO
        self.submit_seconds = 55
        # TODO
        if self.cur_px is not None:
            self.target_price = self.cur_px + 300

    def print(self):
        tm_str = None if self.cur_time is None else tm.strftime("%H:%M:%S", self.cur_time)
        print("当前时间：{0} 当前价格: {1} 预估目标价:{2}.".format(tm_str, self.cur_px, self.target_price))


if __name__=='__main__':
    page = HPPage()
    # for i in range(140):
    while page.running:
        page.print()
        tm.sleep(0.5)