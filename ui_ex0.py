import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('best.pt')

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        loadUi('ui_crop.ui', self)

    def crop_image(self):
        img_path = 't4.jpg'
        results = self.model(img_path, show=True)
        cv2.waitKey(0)

        result = results[0]
        box = result.boxes[0]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]

        # Load the original image
        image = cv2.imread(img_path)

        # Extract the coordinates
        x_min, y_min, x_max, y_max = cords

        # Crop the image using the coordinates
        cropped_image = image[y_min:y_max, x_min:x_max]

        # Convert the cropped image to QPixmap
        height, width, channel = cropped_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(cropped_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Display the cropped image on lblCrop
        self.lblCrop.setPixmap(pixmap)


if __name__ == '__main__':
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec_()
