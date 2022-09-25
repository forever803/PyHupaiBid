import imp
from re import T
from pytesser.pytesser import *
from PIL import Image, ImageEnhance
import cv2
from sys import argv
import pyautogui as auto
import pandas as pd
import numpy as np
import threading
import datetime as dt
import time as tm
import keyboard

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class Config(object):
    fake = False
    test_col_name = "202203"                    # 需要测试的月份
    operation_delay = 0.2
    start_auto_calc_seconds = 30                 # 只能计算最终价格和提交时间的起始时间
    input_px_seconds = 48                       # 多少秒开始输入价格
    # px_region = (1213, 487, 100, 20)            # 价格所在区域 (右边价格)
    # # px_region = (600, 712, 100, 20)             # 价格所在区域   (左边价格)
    # tm_region = (567, 690, 100, 20)             # 时间所在区域
    # target_px_input_pos = (1250, 670)           # 目标价格输入区域
    # bid_button_pos = (1400, 670)                # 出价按钮区域
    # submit_button_pos = (1140, 825)             # 提价按钮区域    

    px_region = (590, 680, 80, 25)             # 价格所在区域   (左边价格)
    tm_region = (557, 659, 80, 25)             # 时间所在区域
    target_px_input_pos = (1226, 630)           # 目标价格输入区域
    verification_code_input_pos = (1277, 680)   # 验证码输入区域
    bid_button_pos = (1410, 630)                # 出价按钮区域
    submit_button_pos = (1100, 800)             # 提交按钮区域

    confirm_button_pos = (1229, 776)             # 提交后成功确认按钮区域

    # 家里 已经调试成功
    # px_region = (600, 644, 80, 20)             # 价格所在区域   (左边价格)
    # tm_region = (567, 621, 80, 20)             # 时间所在区域
    # target_px_input_pos = (1226, 600)           # 目标价格输入区域
    # bid_button_pos = (1410, 600)                # 出价按钮区域
    # submit_button_pos = (1230, 756)             # 提价按钮区域

    today_participants = 200779                 # 今天参拍人数
    today_licenses = 11391                      # 今天 放牌总量

class SubmitInfo(object):
    class Status(object):
        New = 0
        Inputting = 1
        Ready = 2
        Finished = 3

    def __init__(self, input_seconds, submit_seconds, px_diff):
        self.input_seconds = input_seconds
        self.submit_seconds = submit_seconds
        self.px_diff = px_diff
        self._status = SubmitInfo.Status.New
        pass

    @property
    def status(self):
        return self._status
    
    @status.setter
    def status(self, status):
        self._status = status
    
    def print(self):
        print("input_seconds:{} submit_seconds: {} px_diff:{}".format(self.input_seconds, self.submit_seconds, self.px_diff))

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
        self._today_rate = None
        self.prices = {}
        self.data = pd.read_csv(r"data/hist_tick.csv", index_col="seconds")
        self.data['today'] = np.nan
        self.info_df = pd.read_csv(r"data/hist_info.csv", index_col="info")
        self.info_df.loc['rate'] = self.info_df.loc['licenses_count'] / self.info_df.loc['participants_count']
        # start update px thread
        self.update_px_t = threading.Thread(target=self.update_px, args=())
        self.update_px_t.start()
        # start update tm thread
        self.update_tm_t = threading.Thread(target=self.update_tm, args=())
        self.update_tm_t.start()
        # start submit thread        
        self.submit_t = threading.Thread(target=self.run, args=())
        self.submit_t.start()
        self.submit_infos = [
            SubmitInfo(15, 0, 300),
            SubmitInfo(49, 53.9, 600),
            SubmitInfo(57, 0, 300),
        ]
        keyboard.hook(self.on_keyboard)

    @property
    def today_rate(self):
        if self._today_rate is None:
            if Config.fake:
                self._today_rate = self.info_df.loc['licenses_count'][Config.test_col_name] / self.info_df.loc['participants_count'][Config.test_col_name]
            else:
                self._today_rate = Config.today_licenses / Config.today_participants
        return self._today_rate

    def stop(self):
        self.running = False
        if Config.fake:
            print("月份：{0} 准确的提交时间:{1}.".format(Config.test_col_name, self.info_df[Config.test_col_name]["submit_seconds"]))
        # data_df = pd.DataFrame.from_dict(self.prices, orient='index')
        # data_df.to_csv("data.csv")

    def get_px_from_screenshot(self):
        if not Config.fake:
            # 截屏
            # region = (起始x, 起始y, 长, 宽)
            price_img = auto.screenshot(region=Config.px_region)
            # 换为灰度图片
            gray = cv2.cvtColor(np.array(price_img), cv2.COLOR_BGR2GRAY)
            # 转换位image obj
            rawImage = Image.fromarray(gray)
            ret = image_to_string(rawImage, temp_scratch_name="temp_px")
            cur_px = int("".join(list(filter(str.isdigit, ret.split('\n')[0]))))
        else:
            if self.cur_time is not None:
                cur_px = self.data[Config.test_col_name][self.cur_time.time().second]
            else:
                cur_px = None
        return cur_px
    
    def get_time_from_screenshot(self):
        if not Config.fake:            
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
        else:
            if self.cur_time is not None:
                display_time = self.cur_time + dt.timedelta(seconds=0.05)
            else:
                display_time = dt.datetime.strptime("11:29:{0}".format(Config.start_auto_calc_seconds - 1), "%H:%M:%S")
        return display_time

    def update_px(self):
        while self.running:
            try:
                cur_px = self.get_px_from_screenshot()
                if self.cur_time is not None and self.cur_time.time().minute == 29:
                    if self.cur_px is None or cur_px >= self.cur_px:
                        self.cur_px = cur_px
                    continue
                    # 下面是之前code, 来获取并保存历史行情
                    if len(self.prices) == 0:
                        self.data['today'] = self.cur_px
                    second = self.cur_time.time().second
                    if second not in self.prices:
                        self.prices[second] = self.cur_px
                        self.data['today'][self.data.index >= second] = self.cur_px
                        # 计算提交时间和target_price
                        # self.calc_submit_info(second)
                        self.calc_submit_info_old(second)
            except Exception as e:
                print(e)
            finally:
                tm.sleep(0.01)    

    def update_tm(self):
        while self.running:
            try:
                display_time = self.get_time_from_screenshot()
                self.refresh_cur_time(display_time)
                continue
                # 下面是之前code, 提交数据
                if not Config.fake:
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
    
    def run(self):        
        while self.running:
            try:
                cur_seconds = self.get_cur_seconds()
                for info in self.submit_infos:
                    if info.status == SubmitInfo.Status.New:
                        if info.input_seconds <= cur_seconds:
                            info.print()
                            # Input Target Price
                            auto.click(Config.target_px_input_pos)
                            auto.hotkey('ctrl', 'a')
                            auto.press("backspace", interval=0.01)
                            auto.typewrite(message=str(self.cur_px + info.px_diff), interval=0.01)
                            tm.sleep(Config.operation_delay)
                            # 出价
                            auto.click(Config.bid_button_pos)
                            info.status = SubmitInfo.Status.Inputting
                            tm.sleep(Config.operation_delay)
                            auto.click(Config.verification_code_input_pos)
                        break
                    elif info.status == SubmitInfo.Status.Inputting:
                        # Inputting --> Ready 是通过enter键盘输入告知
                        break
                    elif info.status == SubmitInfo.Status.Ready:
                        if info.submit_seconds <= cur_seconds:
                            # 提交
                            auto.click(Config.submit_button_pos)
                            info.status = SubmitInfo.Status.Finished
                        break
                    else:
                        # submit_info 中前一个Finished 才执行下一个，其他情况一直处理这个info的状态
                        continue
                # stop
                if self.cur_time.time().minute == 30:
                    self.stop()
            except Exception as e:
                print(e)
            finally:
                tm.sleep(0.01)

    def on_keyboard(self, key):
        enter_key = keyboard.KeyboardEvent('down', 28, 'enter')
        if key.event_type == 'down' and key.name == enter_key.name:
            print("你按下了enter键！")
            for info in self.submit_infos:
                if info.status == SubmitInfo.Status.Inputting:
                    info.status = SubmitInfo.Status.Ready
        
        esc_key = keyboard.KeyboardEvent('down', 27, 'esc')
        if key.event_type == 'down' and key.name == esc_key.name:
            auto.click(Config.confirm_button_pos)        

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
        # self.info_df['rate'] = self.info_df['licenses_count'] / self.info_df['participants_count']
        if cur_seconds < Config.start_auto_calc_seconds:
            return
        df = self.data[(self.data.index <= cur_seconds) & (self.data.index >= Config.start_auto_calc_seconds)].copy()
        df = df - df.iloc[0]
        df = df[df.columns.difference(['today'])]
        # target_price 训练
        self.train_target_price(cur_seconds)
        # submit_seconds 训练
        self.train_submit_seconds(cur_seconds)

    def train_target_price(self, cur_seconds):        
        # target_price 训练
        px_feature_df = pd.DataFrame(index=self.info_df.columns)
        px_feature_df["rate"] = self.info_df.loc['rate']
        data = self.data[self.info_df.columns]
        px_feature_df["px_delta"] = (data.iloc[cur_seconds] - data.iloc[Config.start_auto_calc_seconds]).values
        px_feature_df["target_px_diff"] = (data.iloc[-1] - data.iloc[cur_seconds]).values
        if Config.fake:
            # px_feature_df = px_feature_df.loc[px_feature_df.index.difference([str(Config.test_col_name)])]
            px_feature_df = px_feature_df.loc[px_feature_df.index < Config.test_col_name]
        # 划分特征值和目标值
        px_feature = px_feature_df[['rate', 'px_delta']].values
        target = np.array(px_feature_df['target_px_diff'])
        # 训练
        lrTool = LinearRegression()
        lrTool.fit(px_feature, target)
        # 预测结果
        cur_delta = self.data["today"][cur_seconds] - self.data["today"][Config.start_auto_calc_seconds]
        feature_test = [[self.today_rate, cur_delta]]
        target_px_diff = lrTool.predict(feature_test)[0]
        self.target_price = self.cur_px + (round(target_px_diff / 100)) * 100 + 100

    def train_submit_seconds(self, cur_seconds):
        # submit_seconds 训练
        sec_feature_df = pd.DataFrame(index=self.info_df.columns)
        sec_feature_df["rate"] = self.info_df.loc['rate']
        data = self.data[self.info_df.columns]
        sec_feature_df["px_delta"] = (data.iloc[cur_seconds] - data.iloc[Config.start_auto_calc_seconds]).values
        sec_feature_df["target_seconds"] = self.info_df.loc['submit_seconds']
        if Config.fake:
            sec_feature_df = sec_feature_df.loc[sec_feature_df.index.difference([str(Config.test_col_name)])]
            # sec_feature_df = sec_feature_df.loc[sec_feature_df.index < Config.test_col_name]
            pass
        # 划分特征值和目标值
        sec_feature = sec_feature_df[['rate', 'px_delta']].values
        target = np.array(sec_feature_df['target_seconds'])
        # 训练
        lrTool = LinearRegression()
        lrTool.fit(sec_feature, target)
        # 预测结果
        cur_delta = self.data["today"][cur_seconds] - self.data["today"][Config.start_auto_calc_seconds]
        feature_test = [[self.today_rate, cur_delta]]
        self.submit_seconds = lrTool.predict(feature_test)[0]

    def calc_submit_info_old(self, cur_seconds):
        if cur_seconds < Config.start_auto_calc_seconds:
            return
        df = self.data[(self.data.index <= cur_seconds) & (self.data.index >= Config.start_auto_calc_seconds)].copy()
        df = df - df.iloc[0]
        # print(df)
        k = 3
        
        corr_infos = []
        for col in df.columns:
            if str(col) == "today" or col >= Config.test_col_name:
                continue
            corr = df["today"].corr(df[col])
            size = len(corr_infos)
            if size == 0:
                corr_infos.insert(0, (corr, col))
            else:
                for i in range(size):
                    if i >= k:
                        break
                    else:
                        if corr_infos[i][0] < corr:
                            corr_infos.insert(i, (corr, col))
                            break
                        if i == size - 1:
                            corr_infos.insert(size, (corr, col))
                            break
        # print(corr_infos[:k])
        col_name = corr_infos[0][1]
        # TODO
        # self.submit_seconds = 55
        self.submit_seconds = self.info_df[col_name]["submit_seconds"]
        if self.submit_seconds == 0:
            self.submit_seconds = 55
        self.submit_seconds -= 0.1
        # TODO
        if self.cur_px is not None:
            # self.target_price = self.cur_px + 300
            ai_submit_seconds = 0
            px_delta = 0
            for i in range(k):
                col_name = corr_infos[i][1]
                px_delta += self.data[col_name].iloc[-1] - self.data[col_name].iloc[cur_seconds]

                submit_seconds = self.info_df[col_name]["submit_seconds"]                
                if submit_seconds == 0:
                    submit_seconds = 55
                ai_submit_seconds += submit_seconds
            px_delta = int(int(px_delta/ (k * 100)) * 100)
            self.target_price = self.cur_px + px_delta + 100
            self.submit_seconds = ai_submit_seconds / k

    def print(self):
        tm_str = None if self.cur_time is None else self.cur_time.time().strftime("%H:%M:%S")
        print("当前时间：{0} 当前价格: {1} 预估目标价:{2} 预估提交时间:{3}.".format(tm_str, self.cur_px, self.target_price, self.submit_seconds))


if __name__=='__main__':
    if len(argv) == 2:
        Config.fake = True
        Config.test_col_name = argv[1]
    page = HPPage()
    # for i in range(140):
    while page.running:
        page.print()
        tm.sleep(1)
