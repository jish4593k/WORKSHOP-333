import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PIL import ImageGrab, Image
from tensorflow import keras
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, 800, 600)
        self.setWindowTitle(' ')
        self.begin = QtCore.QPoint()
        self.end = QtCore.QPoint()
        self.setWindowOpacity(0.3)
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        print('Capture the screen...')
        self.show()

    def paintEvent(self, event):
        qp = QtGui.QPainter(self)
        qp.setPen(QtGui.QPen(QtGui.QColor('black'), 3))
        qp.setBrush(QtGui.QColor(128, 128, 255, 128))
        qp.drawRect(QtCore.QRect(self.begin, self.end))

    def mousePressEvent(self, event):
        self.begin = event.pos()
        self.end = self.begin
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.close()

        x1, y1 = min(self.begin.x(), self.end.x()), min(self.begin.y(), self.end.y())
        x2, y2 = max(self.begin.x(), self.end.x()), max(self.begin.y(), self.end.y())

        img = ImageGrab.grab(bbox=(x1, y1, x2, y2))
        img.save('capture.png')
        img_np = np.array(img)

    
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        edges_img = cv2.Canny(blurred_img, 50, 150)

        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        torch_img = transform(edges_img)
        torch_img = torch_img.unsqueeze(0)  # Add batch dimension

        
        resnet_model = models.resnet18(pretrained=True)
        resnet_model.eval()

     
        with torch.no_grad():
            output = resnet_model(torch_img)

        self.display_results(output)

    def display_results(self, output):
        result_str = f"ResNet Output: {output.argmax()}"
        msg_box = QtWidgets.QMessageBox()
        msg_box.setWindowTitle("Image Processing Results")
        msg_box.setText(result_str)
        msg_box.exec_()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyWidget()
    window.show()
    app.aboutToQuit.connect(app.deleteLater)
    sys.exit(app.exec_())
