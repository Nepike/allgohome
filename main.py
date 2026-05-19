from PyQt6.QtWidgets import QApplication
import sys

from GUI import MainWindow

from config import *



if __name__ == "__main__":


    app = QApplication(sys.argv)
    window = MainWindow(GRID_W, GRID_H, CELL_SIZE, DT)
    window.show()
    sys.exit(app.exec())