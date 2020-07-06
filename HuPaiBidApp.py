from HuPaiBidGui import HuPaiBidGui
import tkinter as tk
import time
import threading
import keyboard
import pytesseract
import pyautogui as auto
from Config import *
try:
    from PIL import Image
except ImportError:
    import Image


class HuPaiBidApp(HuPaiBidGui):
    bid_button_pos = (0, 0)
    cancel_button_pos = (0, 0)
    custom_price_input_pos = (0, 0)
    price_300_button_pos = (0, 0)
    rise_prise_button_pos = (0, 0)
    submit_button_pos = (0, 0)
    verification_code_input_pos = (0, 0)
    confirm_after_bid_button_pos = (0, 0)

    def __init__(self):
        HuPaiBidGui.__init__(self)

        self.debug = False
        self.event = threading.Event()
        #_thread.start_new_thread( thread_ocr, ("price", ))
        self.update_time_t = threading.Thread(
            target=self.update_time, args=())
        self.update_time_t.setDaemon(True)
        self.update_time_t.start()
        self.update_price_display_t = threading.Thread(
            target=self.update_price_display, args=())
        self.update_price_display_t.setDaemon(True)
        self.update_price_display_t.start()
        self.local_time = time.time()
        self.rtc_alarm_events = []
        self.mainloop()

    def update_time(self):
        while True:
            try:
                if self.debug:
                    self.current_time = time.time() - self.debug_start_time + DBG_TIMESTAMP + \
                        float(self.time_drift_entry.get())
                else:
                    self.current_time = time.time() + float(self.time_drift_entry.get())
                current_time_str = time.strftime(
                    "%H:%M:%S", time.localtime(self.current_time))
                self.current_time_text.set(current_time_str)

                if not self.first_bid_is_processed and current_time_str == (f"11:29:{self.first_bid_time_entry_text.get()}"):
                    self.first_bid_is_processed = True
                    self.first_bid_t = threading.Thread(target=self.first_bid_handle, args=())
                    self.first_bid_t.setDaemon(True)
                    self.first_bid_t.start()
                
                if not self.second_bid_is_processed and current_time_str == (f"11:29:{self.second_bid_time_entry_text.get()}"):
                    self.second_bid_is_processed = True
                    self.second_bid_t = threading.Thread(target=self.second_bid_handle, args=())
                    self.second_bid_t.setDaemon(True)
                    self.second_bid_t.start()

                time.sleep(0.05)
            except:
                pass
        print("exit")

    def first_bid_handle(self):
        print(time.strftime(
                    "%H:%M:%S", time.localtime(self.current_time)))

    def second_bid_handle(self):
        print(time.strftime(
                    "%H:%M:%S", time.localtime(self.current_time)))

    def update_price_display(self):
        self.event.wait()

        RMB_AREA = auto.locateOnScreen(
            './img/rmb.png', region=LEFT_AREA, confidence=0.9, grayscale=True)
        PRICE_AREA = (RMB_AREA.left + RMB_AREA.width,
                      RMB_AREA.top, 140, RMB_AREA.height)

        pytesseract.pytesseract.tesseract_cmd = r'./tesseract/tesseract.exe'

        while True:
            try:
                price_img = auto.screenshot(region=PRICE_AREA)

                new_img = Image.new(mode='RGB', size=(
                    140*2, 40*2), color=(255, 255, 255))
                new_img.paste(price_img, (int(140/2), int(40/2)))
                new_img.save("tmp.png")
                r = pytesseract.image_to_string(new_img)
                print(int("".join(list(filter(str.isdigit, r)))))
                self.current_price_text.set(
                    int("".join(list(filter(str.isdigit, r)))))
            except:
                print("OCR Wrong!")
                self.current_price_text.set("00000")

    def debug_start(self):
        self.debug = True
        self.debug_start_time = time.time()
        self.first_bid_is_processed = False
        self.second_bid_is_processed = False

        print(self.time_drift_entry.get())
        print(self.first_bid_time_entry.get())
        print(self.first_bid_price_entry.get())
        print(self.second_bid_time_entry.get())
        print(self.second_bid_price_entry.get())
        self.clear_log_display()

    def release_start(self):
        self.debug = False

    def get_pos(self, img):
        area = auto.locateOnScreen(img, region=(
            900, 300, 500, 400), confidence=0.9)
        try:
            center = auto.center(area)
        except:
            center = 0

        return center

    def screnn_coordinate_calibration(self):
        print("Get rise_prise_button_pos")
        rise_prise_button_pos = self.get_pos(RISE_PRISE_BUTTON_PNG)
        print(rise_prise_button_pos)
        auto.click(rise_prise_button_pos)
        print(rise_prise_button_pos)
        time.sleep(CAL_DELAY)

        print("Get custom_price_input_pos")
        custom_price_input_pos = self.get_pos(CUSTOM_PRICE_INPUT_PNG)
        if custom_price_input_pos:
            auto.click(custom_price_input_pos)
            auto.typewrite(message='600', interval=0.2)
            auto.moveTo(custom_price_input_pos)
        else:
            custom_price_input_pos = (
                rise_prise_button_pos[0] - 100, rise_prise_button_pos[1])
            auto.moveTo(custom_price_input_pos)
        print(custom_price_input_pos)
        time.sleep(CAL_DELAY)

        print("Get price_300_button_pos")
        price_300_button_pos = self.get_pos(PRICE_300_BUTTON_PNG)
        auto.moveTo(price_300_button_pos)
        print(price_300_button_pos)
        time.sleep(CAL_DELAY)

        print("Get bid_button_pos")
        bid_button_pos = self.get_pos(BID_BUTTON_PNG)
        auto.moveTo(bid_button_pos)
        print(bid_button_pos)
        time.sleep(CAL_DELAY)

        self.event.set()

    def update_log_display(self, log):
        self.log_display_text.insert(tk.INSERT, log + '\n')
        self.log_display_text.see(tk.END)

    def clear_log_display(self):
        self.log_display_text.delete(0.0, tk.END)

    def about(self):
        tk.messagebox.showinfo('关于', 'Python版拍牌小助手\nwmpluto@gmail.com')


a = HuPaiBidApp()
