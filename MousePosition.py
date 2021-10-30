import time
from pynput.mouse import Button, Controller


if __name__ == "__main__":
    mouse_er = Controller()
    while True:
        print('当前鼠标坐标为 {0}'.format(mouse_er.position))
        time.sleep(2)
    pass