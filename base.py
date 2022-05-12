from PySide6.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QHBoxLayout, QWidget

class SecondWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.label1 = QLabel()
        self.setCentralWidget(self.label1)

class Window(QMainWindow):

    def __init__(self):
        super().__init__()    
        self.initUI()

    def initUI(self):
        self.secondWindow = SecondWindow()

        self.button1 = QPushButton('push')
        self.button1.clicked.connect(self.clickedButton1)
        
        self.button2 = QPushButton('change_label')
        self.button2.clicked.connect(self.clickedButton2)

        layout = QHBoxLayout()
        layout.addWidget(self.button1)
        layout.addWidget(self.button2)

        mainWidget = QWidget()
        mainWidget.setLayout(layout)
        
        self.setCentralWidget(mainWidget)

    def clickedButton1(self):
        self.secondWindow.show()
        self.secondWindow.label1.setText('aaa')
    
    def clickedButton2(self):
        self.secondWindow.hide()
        
if __name__ == "__main__":
    app = QApplication([])
    ex =Window()
    ex.show()
    app.exec()