from re import T
from pytesser.pytesser import *
from PIL import Image, ImageEnhance
import cv2
import pyautogui as auto
import pandas as pd
import numpy as np
import threading
import datetime as dt
import time as tm


class Config(object):
    operation_delay = 0.2
    input_px_seconds = 48                       # 多少秒开始输入价格
    # px_region = (1213, 487, 100, 20)            # 价格所在区域 (右边价格)
    # # px_region = (600, 712, 100, 20)             # 价格所在区域   (左边价格)
    # tm_region = (567, 690, 100, 20)             # 时间所在区域
    # target_px_input_pos = (1250, 670)           # 目标价格输入区域
    # bid_button_pos = (1400, 670)                # 出价按钮区域
    # submit_button_pos = (1140, 825)             # 提价按钮区域    

    # px_region = (600, 644, 80, 20)             # 价格所在区域   (左边价格)
    # tm_region = (564, 617, 80, 20)             # 时间所在区域
    # target_px_input_pos = (1226, 597)           # 目标价格输入区域
    # bid_button_pos = (1410, 600)                # 出价按钮区域
    # submit_button_pos = (1142, 756)             # 提价按钮区域
    px_region = (600, 644, 80, 20)             # 价格所在区域   (左边价格)
    tm_region = (567, 621, 80, 20)             # 时间所在区域
    target_px_input_pos = (1226, 600)           # 目标价格输入区域
    bid_button_pos = (1410, 600)                # 出价按钮区域
    submit_button_pos = (1230, 756)             # 提价按钮区域

    today_participants = 200779                 # 今天参拍人数
    today_licenses = 11391                      # 今天 放牌总量

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
        self.data = pd.read_csv(r"data/hist_tick.csv", index_col="seconds")
        self.data['today'] = np.nan
        self.info_df = pd.read_csv(r"data/hist_info.csv", index_col="info")
        # start update px thread
        self.update_px_t = threading.Thread(target=self.update_px, args=())
        self.update_px_t.start()
        # start update tm thread
        self.update_tm_t = threading.Thread(target=self.update_tm, args=())
        self.update_tm_t.start()

    def stop(self):
        self.running = False
        # data_df = pd.DataFrame.from_dict(self.prices, orient='index')
        # data_df.to_csv("data.csv")

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
                cur_px = int("".join(list(filter(str.isdigit, ret.split('\n')[0]))))
                if self.cur_time is not None and self.cur_time.time().minute == 29:
                    if self.cur_px is None or cur_px >= self.cur_px:
                        self.cur_px = cur_px
                    if len(self.prices) == 0:
                        self.data['today'] = self.cur_px
                    second = self.cur_time.time().second
                    if second not in self.prices:
                        self.prices[second] = self.cur_px
                        self.data['today'][self.data.index >= second] = self.cur_px
                        # 计算提交时间和target_price
                        self.calc_submit_info(second)
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
                # tm_str = "".join(list(filter(lambda x: str.isdigit(x) or x == ":", ret.split('\n')[0])))
                # display_time = tm.strptime(tm_str, "%H:%M:%S")
                display_time = self.parse_display_time(ret)
                self.refresh_cur_time(display_time)
                # 尝试出价
                self.try_input_px()
                # 尝试提交
                self.try_submit()
                # stop
                if self.cur_time.time().minute == 30:
                    self.stop()
            except Exception as e:
                print(e)
            finally:
                tm.sleep(0.01)
    
    def parse_display_time(self, ret):
        l = []
        for x in ret.split('\n')[0]:
            if x == "I":
                l.append("1")
            elif str.isdigit(x):
                l.append(x)
        try:
            # display_time = tm.strptime("".join(l), "%H%M%S")
            display_time = dt.datetime.strptime("".join(l), "%H%M%S")
            return display_time
        except Exception as e:
            print(e)
            return None

    def refresh_cur_time(self, display_time):
        now = dt.datetime.now()
        if self.cur_time is None:            
            self.cur_time = display_time
            self.refresh_display_tm_dt = now
        else:
            delta = dt.datetime.now() - self.refresh_display_tm_dt
            if display_time is None \
                or display_time <= self.cur_time \
                or (display_time - self.cur_time) > delta + dt.timedelta(seconds=1):
                # 1. display_time 没解析出来
                # 2. display_time 解析错误 小
                # 3. display_time 解析错误 大
                self.cur_time += delta
                self.refresh_display_tm_dt = now
            else:
                self.cur_time = display_time
                self.refresh_display_tm_dt = now

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
        # if self.cur_time is not None and self.cur_time.tm_min == 29:
        if self.cur_time is not None and self.cur_time.time().minute == 29:
            delta = dt.datetime.now() - self.refresh_display_tm_dt
            cur_seconds = self.cur_time.time().second + delta.seconds + delta.microseconds / 1000000.0
        return cur_seconds

    def calc_submit_info(self, cur_seconds):
        if cur_seconds < 40:
            return
        df = self.data[(self.data.index <= cur_seconds) & (self.data.index >= 40)].copy()
        df = df - df.iloc[0]
        # print(df)
        max_corr = None
        col_name = None
        corr_dict = {}
        for col in df.columns:
            if col == "today":
                continue
            corr = df["today"].corr(df[col])
            corr_dict[col] = corr
            if corr != np.nan:
                if max_corr is None or max_corr < corr:
                    max_corr = corr
                    col_name = col
        # print(corr_dict)
        print(col_name)
        # TODO
        # self.submit_seconds = 55
        self.submit_seconds = self.info_df[col_name]["submit_seconds"]
        # TODO
        if self.cur_px is not None:
            # self.target_price = self.cur_px + 300
            px_delta = self.data[col_name].iloc[-1] - self.data[col_name].iloc[cur_seconds]
            self.target_price = self.cur_px + px_delta + 200

    def print(self):
        tm_str = None if self.cur_time is None else self.cur_time.time().strftime("%H:%M:%S")
        print("当前时间：{0} 当前价格: {1} 预估目标价:{2}.".format(tm_str, self.cur_px, self.target_price))


if __name__=='__main__':
    page = HPPage()
    # for i in range(140):
    while page.running:
        page.print()
        tm.sleep(0.3)