import cv2
from PySide6.QtCore import Qt, Signal, Slot, QThread, QSize
from PySide6.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QHBoxLayout, QWidget
from PySide6.QtGui import QImage, QPixmap

class VideoThread(QThread):

    change_pixmap_signal = Signal(QImage)
    playing = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.playing:
            ret, frame = cap.read()
            if ret:
                h, w, ch = frame.shape
                bytesPerLine = ch * w
                image = QImage(frame, w, h, bytesPerLine, QImage.Format.Format_BGR888)
                self.change_pixmap_signal.emit(image)
        cap.release()

    def stop(self):
        self.playing = False
        self.wait()

class SubWindow(QMainWindow):
    video_size = QSize(640, 480)
    def __init__(self):
        super().__init__()
        self.setFixedSize(QSize(self.video_size))
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.label1 = QLabel()
        self.setCentralWidget(self.label1)

class Window(QMainWindow):

    def __init__(self):
        super().__init__()    
        self.secondWindow = SubWindow()
        self.secondWindow.show()

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        self.initUI()

    def initUI(self):
        
        self.button1 = QPushButton('push')
        
        self.button2 = QPushButton('change_label')
        
        layout = QHBoxLayout()
        layout.addWidget(self.button1)
        layout.addWidget(self.button2)

        mainWidget = QWidget()
        mainWidget.setLayout(layout)
        
        self.setCentralWidget(mainWidget)

    def closeEvent(self, e):
       self.thread.stop()
       self.secondWindow.close()
       e.accept()
    
    @Slot(QImage)
    def update_image(self, image):
        self.secondWindow.label1.setPixmap(QPixmap.fromImage(image))
     
if __name__ == "__main__":
    app = QApplication([])
    ex =Window()
    ex.show()
    app.exec()