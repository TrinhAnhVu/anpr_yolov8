import numpy as np
import psutil
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QFileInfo, QTimer, QTime, QThread
from PyQt5.uic import loadUi
from ultralytics import YOLO
import cv2
import easyocr
import re

model = YOLO('best.pt')


class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        loadUi('ex_ui_1.ui', self)
        self.setWindowTitle("Nhận diện biển số xe")

        self.btnOpen.clicked.connect(self.open_image)
        self.btnDetect.clicked.connect(self.detect_objects)
        self.btnReset.clicked.connect(self.reset_image)
        self.btnOpenVideo.clicked.connect(self.open_video)

        self.rbCamera.clicked.connect(self.connect_camera)

        self.video_cap = None
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video_frame)

        self.video_file_name = ""
        self.templateSizeSlider.valueChanged.connect(self.detect_objects)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)  # Cập nhật thời gian mỗi giây

        self.update_time()

    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        if file_name:

            file_info = QFileInfo(file_name)
            file_name = file_info.fileName()
            pixmap = QPixmap(file_name)
            height_image = pixmap.height()
            width_image = pixmap.width()
            self.lblHroot.setText(f"{height_image} pixel")
            self.lblWroot.setText(f"{width_image} pixel")
            self.lblImage.setPixmap(pixmap.scaled(self.lblImage.size(), Qt.AspectRatioMode.KeepAspectRatio))
            self.lblPath.setText(file_name)

    def open_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Video', '', 'Video Files (*.mp4 *.avi *.mov)')
        if file_name:

            file_info = QFileInfo(file_name)
            file_name = file_info.fileName()
            self.lblVideo.setText(file_name)
            self.video_file_name = file_name
            self.video_cap = cv2.VideoCapture(file_name)
            self.video_timer.start(1)

    def play_video(self, video_path):
        if self.rbCamera.isChecked():
            cap = cv2.VideoCapture(0)  # Mở kết nối với camera
        else:
            cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)

        interval = int(1000 / fps)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))

            q_image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.lblVideo.setPixmap(pixmap)
            self.lblVideo.setScaledContents(True)

            QApplication.processEvents()

            QThread.msleep(interval)

        cap.release()

    def update_video_frame(self):

        ret, frame = self.video_cap.read()
        if ret:
            results = model(frame)
            result = results[0]
            if len(result.boxes) > 0:
                box = result.boxes[0]
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]
                x_min, y_min, x_max, y_max = cords
                cropped_image = frame[y_min:y_max, x_min:x_max]

                gray = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
                if cropped_image.shape[0] < 80 or cropped_image.shape[1] < 180:
                    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                if cropped_image.shape[0] < 80:
                    kernel = np.ones((4, 3), np.uint8)
                else:
                    kernel = np.ones((3, 3), np.uint8)
                thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                reader = easyocr.Reader(['en'])
                text = ''

                if len(contours) == 2:
                    # License plate with 2 lines
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        char_img = thresh[y:y + h, x:x + w]
                        result = reader.readtext(char_img)
                        if len(result) > 0:
                            word = result[0][1]
                            word = re.sub(r'[^a-zA-Z0-9]', '', word)
                            if len(word) > 0:
                                if word.isdigit():
                                    text += word
                                else:
                                    text += word.upper()
                                text += ' '
                else:
                    # License plate with 1 line
                    result = reader.readtext(thresh)
                    for res in result:
                        word = res[1]
                        word = re.sub(r'[^a-zA-Z0-9]', '', word)
                        if len(word) > 0:
                            if word.isdigit():
                                text += word
                            else:
                                text += word.upper()
                            text += ' '

                # Draw bounding box and text on the frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, text.strip(), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))
            q_image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.lblVideo.setPixmap(pixmap)
            self.lblVideo.setScaledContents(True)

    def detect_objects_video(self):
        if self.video_file_name:
            cap = cv2.VideoCapture(self.video_file_name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            interval = int(1000 / fps)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                result = results[0]
                if len(result.boxes) > 0:
                    box = result.boxes[0]
                    cords = box.xyxy[0].tolist()
                    cords = [round(x) for x in cords]
                    x_min, y_min, x_max, y_max = cords

                    cropped_image = frame[y_min:y_max, x_min:x_max]

                    cropped_image = cv2.fastNlMeansDenoisingColored(cropped_image, None, 10, 10, 7, 21)
                    gray = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
                    if cropped_image.shape[0] < 80 or cropped_image.shape[1] < 180:
                        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                    if cropped_image.shape[0] < 80:
                        kernel = np.ones((4, 3), np.uint8)
                    else:
                        kernel = np.ones((3, 3), np.uint8)
                    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    reader = easyocr.Reader(['en'])
                    text = ''

                    if len(contours) == 2:
                        # License plate with 2 lines
                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            char_img = thresh[y:y + h, x:x + w]
                            result = reader.readtext(char_img)
                            if len(result) > 0:
                                word = result[0][1]
                                word = re.sub(r'[^a-zA-Z0-9]', '', word)
                                if len(word) > 0:
                                    if word.isdigit():
                                        text += word
                                    else:
                                        text += word.upper()
                                    text += ' '
                    else:
                        # License plate with 1 line
                        result = reader.readtext(thresh)
                        for res in result:
                            word = res[1]
                            word = re.sub(r'[^a-zA-Z0-9]', '', word)
                            if len(word) > 0:
                                if word.isdigit():
                                    text += word
                                else:
                                    text += word.upper()
                                text += ' '

                    # Draw bounding box and text on the frame
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, text.strip(), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))
                q_image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                self.lblVideo.setPixmap(pixmap)
                self.lblVideo.setScaledContents(True)
                QApplication.processEvents()
                QThread.msleep(interval)
            cap.release()

    def connect_camera(self):
        if self.rbCamera.isChecked():

            self.timer = QTimer()
            self.timer.timeout.connect(self.display_video_camera)
            self.timer.start(30)

    def display_video_camera(self):

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

        if ret:

            results = model(frame)
            result = results[0]
            if len(result.boxes) > 0:
                box = result.boxes[0]
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]
                x_min, y_min, x_max, y_max = cords

                cropped_image = frame[y_min:y_max, x_min:x_max]


                cropped_image = cv2.fastNlMeansDenoisingColored(cropped_image, None, 10, 10, 7, 21)
                gray = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
                if cropped_image.shape[0] < 80 or cropped_image.shape[1] < 180:
                    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                if cropped_image.shape[0] < 80:
                    kernel = np.ones((4, 3), np.uint8)
                else:
                    kernel = np.ones((3, 3), np.uint8)
                thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                reader = easyocr.Reader(['en'])
                text = ''

                if len(contours) == 2:
                    # License plate with 2 lines
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        char_img = thresh[y:y + h, x:x + w]
                        result = reader.readtext(char_img)
                        if len(result) > 0:
                            word = result[0][1]
                            word = re.sub(r'[^a-zA-Z0-9]', '', word)
                            if len(word) > 0:
                                if word.isdigit():
                                    text += word
                                else:
                                    text += word.upper()
                                text += ' '
                else:
                    # License plate with 1 line
                    result = reader.readtext(thresh)
                    for res in result:
                        word = res[1]
                        word = re.sub(r'[^a-zA-Z0-9]', '', word)
                        if len(word) > 0:
                            if word.isdigit():
                                text += word
                            else:
                                text += word.upper()
                            text += ' '

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, text.strip(), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            self.lblVideo.setPixmap(QPixmap.fromImage(q_image))
            self.lblVideo.setScaledContents(True)
            self.lblVideo.setAlignment(Qt.AlignCenter)

        cap.release()

    def detect_objects(self):
        file_name = self.lblPath.text()
        if file_name:

            img = cv2.imread(file_name)
            results = model(img)
            result = results[0]

            if len(result.boxes) > 0:
                box = result.boxes[0]
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]
                x_min, y_min, x_max, y_max = cords
                cropped_image = img[y_min:y_max, x_min:x_max]
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                height, width, channel = cropped_image.shape
                self.lblHcrop.setText(f"{height} pixel")
                self.lblWcrop.setText(f"{width} pixel")
                bytes_per_line = 3 * width
                q_image = QImage(cropped_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                self.lblCrop.setPixmap(pixmap.scaled(self.lblCrop.size(), Qt.AspectRatioMode.KeepAspectRatio))

                template_window_size = self.templateSizeSlider.value()
                self.labelSize.setText(str(template_window_size))
                cropped_image = cv2.fastNlMeansDenoisingColored(cropped_image, None, 10, 10, template_window_size, 21)
                gray = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
                if cropped_image.shape[0] < 80 or cropped_image.shape[1] < 180:
                    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                if cropped_image.shape[0] < 80:
                    kernel = np.ones((4, 3), np.uint8)
                else:
                    kernel = np.ones((3, 3), np.uint8)
                thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                reader = easyocr.Reader(['en'])
                text = ''

                if len(contours) == 2:

                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        char_img = thresh[y:y + h, x:x + w]
                        result = reader.readtext(char_img)
                        if len(result) > 0:
                            word = result[0]
                            word = re.sub(r'[^a-zA-Z0-9]', '', word)
                            if len(word) > 0:
                                if word.isdigit():
                                    text += word
                                else:
                                    text += word.upper()
                                text += ' '
                    self.lblLicensePlate.setText(text.strip())
                else:
                    
                    result = reader.readtext(thresh)
                    for res in result:
                        word = res[1]
                        word = re.sub(r'[^a-zA-Z0-9]', '', word)
                        if len(word) > 0:
                            if word.isdigit():
                                text += word
                            else:
                                text += word.upper()
                            text += ' '
                    self.lblLicensePlate.setText(text.strip())
            else:
                self.lblCrop.clear()
                self.lblLicensePlate.clear()

    def update_time(self):
        current_time = QTime.currentTime()
        time_string = current_time.toString("hh:mm:ss")
        self.lcdTime.display(time_string)
        cpu_percent = psutil.cpu_percent(interval=1)
        ram_percent = psutil.virtual_memory().percent
        self.lblCPU.setText(f"{cpu_percent}%")
        self.lblRAM.setText(f"{ram_percent}%")

    def reset_image(self):
        self.lblImage.setPixmap(QPixmap())
        self.lblCrop.setPixmap(QPixmap())
        self.lblPath.clear()
        self.lblLicensePlate.clear()
        self.lblHroot.clear()
        self.lblWroot.clear()
        self.lblHcrop.clear()
        self.lblWcrop.clear()


if __name__ == '__main__':
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec_()
