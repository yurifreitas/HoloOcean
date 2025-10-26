from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtWebEngineWidgets import QWebEngineView
import sys

app = QApplication(sys.argv)
window = QMainWindow()
view = QWebEngineView()

view.load("/home/yuri/Documents/code2/magneto/holo_ocean_tensorial.html")  # ou qualquer URL
window.setCentralWidget(view)
window.showMaximized()
app.exec()
