import numpy as np
import torch
from torchvision import transforms
import cv2
from PyQt6.QtCore import pyqtSignal, pyqtSlot, QThread
from PyQt6.QtWidgets import QWidget, QApplication, QLabel, QSlider, QGridLayout
from PyQt6.QtGui import QImage, QPixmap

from constructGUI import construct

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])

model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
model.eval().to(device)

class VideoThread(QThread):

    change_pixmap_signal = pyqtSignal(QImage)
    playing = True

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    crop = 0
    leftside = 280
    bokashi = 0

    def run(self):
        
        while self.playing:
            ret, frame = self.cap.read()
            
            frame = frame[:, self.leftside:self.leftside+720, :]
            
            if not self.crop == 0:
                frame = cv2.resize(frame[self.crop:-self.crop, self.crop:-self.crop, :], dsize = (720, 720))
            
            if not self.bokashi == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = transform_fn(frame_rgb)
                input_batch = input_tensor.unsqueeze(0)
            
                with torch.no_grad():
                    output = model(input_batch.to(device))['out'][0]
                
                predict = output.argmax(0).to('cpu').numpy()

                mask_1 = np.where(predict == 15, 1, 0)[...,np.newaxis]
                mask_2 = np.where(predict == 15, 0, 1)[...,np.newaxis]
                if self.bokashi ==0:
                    blurred_img = cv2.blur(frame, (1, 1))
                else:
                    blurred_img = cv2.blur(frame, (self.bokashi, self.bokashi))
                frame = (frame * mask_1 + blurred_img * mask_2).astype('uint8')

            h, w, ch = frame.shape
            bytesPerLine = ch * w
            image = QImage(frame.copy(), w, h, bytesPerLine, QImage.Format.Format_BGR888)
            
            self.change_pixmap_signal.emit(image)

        self.cap.release()

    def stop(self):
        self.playing = False
        self.wait()
    
    def set_zoom_scale(self, x):    
        self.crop = x
    
    def set_move_scale(self, x):    
        self.leftside = x

    def set_bokashi_scale(self, x):    
        self.bokashi = x

class Window(QWidget):

    def __init__(self):
        super().__init__()    
        self.initUI()
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def initUI(self):
        self.setWindowTitle("OpenCV-Python sample")
        
        self.img_label1 = construct(QLabel(), 'settings.yml', 'img_label')

        self.zoom_label = construct(QLabel('ズーム'), 'settings.yml', 'text_label')
        self.move_label = construct(QLabel('首振り'), 'settings.yml', 'text_label')
        self.bokashi_label = construct(QLabel('背景ぼかし'), 'settings.yml', 'text_label')

        self.zoom_slider = construct(QSlider(), 'settings.yml', 'zoom_slider')
        self.zoom_slider.valueChanged.connect(self.zoom_slider_change)

        self.move_slider = construct(QSlider(), 'settings.yml', 'move_slider')
        self.move_slider.valueChanged.connect(self.move_slider_change)

        self.bokashi_slider = construct(QSlider(), 'settings.yml', 'bokashi_slider')
        self.bokashi_slider.valueChanged.connect(self.bokashi_slider_change)

        self.grid = QGridLayout()
        
        self.grid.addWidget(self.img_label1, 0, 0, 1, 5)
        self.grid.addWidget(self.zoom_label, 1, 0)
        self.grid.addWidget(self.zoom_slider, 1, 1, 1, 4)
        self.grid.addWidget(self.move_label, 2, 0)
        self.grid.addWidget(self.move_slider, 2, 1, 1, 4)
        self.grid.addWidget(self.bokashi_label, 3, 0)
        self.grid.addWidget(self.bokashi_slider, 3, 1, 1, 4)

        self.setLayout(self.grid)

    def closeEvent(self, e):
       self.thread.stop()
       e.accept()
    
    def zoom_slider_change(self):
        self.thread.set_zoom_scale(self.zoom_slider.value() * 30)
    
    def move_slider_change(self):
        self.thread.set_move_scale(self.move_slider.value() * 70)
    
    def bokashi_slider_change(self):
        self.thread.set_bokashi_scale(self.bokashi_slider.value() * 5)

    @pyqtSlot(QImage)
    def update_image(self, image):
        self.img_label1.setPixmap(QPixmap.fromImage(image))

if __name__ == "__main__":
    app = QApplication([])
    ex =Window()
    ex.show()
    app.exec()