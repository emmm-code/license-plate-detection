from PyQt5.QtGui import QPixmap
from window import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QInputDialog
from PyQt5 import QtGui, QtWidgets
import sys
import cv2
import time
import os
import numpy as np
# from test import YOLO_License_plate_Detector
from demo import YOLO_License_plate_Detector

def label_show(label, image):
    qt_img_buf = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    qt_img = QtGui.QImage(qt_img_buf.data, qt_img_buf.shape[1], qt_img_buf.shape[0], QtGui.QImage.Format_RGB32)
    image = QPixmap.fromImage(qt_img).scaled(label.width(), label.height())
    label.setPixmap(image)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.image_file = False

        self.input_dir = False
        self.save_dir = False

        self.setupUi(self)
        self.initUI()
        self.mosaic_level = 0.6  # 模糊程度
        self.detector = YOLO_License_plate_Detector()
        self.videoFPS = 24
        self.image = False
        self.video = False
        self.video_name = False
        self.camera = False
        self.camera_selected = 0
        self.loop = False
        self.detect = False
        self.cap = None

    def print_log(self, log_words):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.textLog.append(current_time + ':' + log_words)  # 在指定的区域显示提示信息
        cursor = self.textLog.textCursor()
        self.textLog.moveCursor(cursor.End)  # 光标移到最后，这样就会自动显示出来
        QtWidgets.QApplication.processEvents()

    def initUI(self):
        self.setWindowTitle('基于YOLOv3的汽车车牌打码')
        self.actionLoadImage.triggered.connect(self.load_image)

        self.actionLoadImageDir.triggered.connect(self.load_image_dir)
        self.actionSaveImageDir.triggered.connect(self.save_image_dir)

        self.actionStart.triggered.connect(self.start_detect)

        self.actionBatchProcess.triggered.connect(self.batch_process)

        # self.actionLoadVideo.triggered.connect(self.load_video)
        # self.actionVideoFPS.triggered.connect(self.change_video_fps)
        # self.actionOpenCamera.triggered.connect(self.open_camera)
        # self.actionCloseCamera.triggered.connect(self.close_camera)
        # self.actionSelectedCamera.triggered.connect(self.select_camera)
        # self.actionLoop.triggered.connect(self.loop_video)
        self.actionMosaicLevel.triggered.connect(self.change_mosaic_level)

    def load_image(self):
        image_name, image_type = QFileDialog.getOpenFileName(self, '打开图片', r'./', '图片 (*.png *.jpg *.jpeg)')
        if image_name == '':
            self.print_log(log_words=f'取消加载图片')
            return
        cv2_image = cv2.imread(image_name)
        label_show(label=self.labelImageOrVideo, image=cv2_image)
        self.image = True
        self.print_log(log_words=f'加载图片:{image_name}')
        self.image_file = cv2_image

    def load_image_dir(self):
        input_path = QFileDialog.getExistingDirectory(self, '选择图片导入目录', os.getcwd())
        if input_path == '':
            self.print_log(log_words=f'取消加载图片路径')
            return
        # self.image_dir = True
        self.print_log(log_words=f'加载图片路径:{input_path}')
        self.input_dir = input_path

    def save_image_dir(self):
        save_path = QFileDialog.getExistingDirectory(self, '选择图片保存目录', os.getcwd())
        if save_path == '':
            self.print_log(log_words=f'取消选择图片保存路径')
            return
        self.print_log(log_words=f'加载图片保存路径:{save_path}')
        self.save_dir = save_path

    def batch_process(self):
        self.detect = True
        image_input_path = self.input_dir
        image_output_path = self.save_dir

        if image_input_path != '' and image_output_path != '':
            self.print_log(log_words=f'开始车牌打码')
            images_list = os.listdir(image_input_path)
            for idx, img_name in enumerate(images_list):
                image_path = os.path.join(image_input_path, img_name)
                frame, success = self.detector.rectangle(image_path, self.mosaic_level)
                if success:
                    self.print_log(log_words=f'成功定位车辆车牌')
                    cv2.imwrite(os.path.join(image_output_path, img_name), frame.astype(np.uint8))  # cv.imwrite只能保存BGR图像

                else:
                    self.print_log(log_words=f'无法定位车辆车牌')
            self.print_log(log_words=f'结束车牌打码')
        else:
            self.print_log(log_words=f'请选择车牌导入路径和保存路径')

    def start_detect(self):
        self.detect = True
        if self.image:
            rectangle, success = self.detector.rectangle(self.image_file, self.mosaic_level)
            if success:
                label_show(label=self.labelImageOrVideo, image=rectangle)
            else:
                self.print_log(log_words=f'无法定位车辆车牌')



    def change_mosaic_level(self):
        number, ok = QInputDialog.getDouble(self, "设置模糊程度", "[0.1-1] (默认0.6)")
        if number < 0.1 or number > 1:
            self.print_log(log_words=f'非法数值，设置失败')
            return
        self.mosaic_level = number
        self.print_log(log_words=f'设置模糊程度: {str(number)}')




if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())