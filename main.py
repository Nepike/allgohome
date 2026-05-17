import random
import sys
import math
from dataclasses import dataclass

from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QColor, QPainter, QPen, QAction, QPolygonF, QPixmap, QCursor

from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QListWidget, QMainWindow, QPushButton, QVBoxLayout, \
    QWidget, QFrame, QTextEdit, QGroupBox, QFormLayout, QLineEdit, QMessageBox

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# CONFIG
CELL_SIZE = 25
GRID_W = 30
GRID_H = 30

MAP_W = GRID_W * CELL_SIZE
MAP_H = GRID_H * CELL_SIZE

@dataclass
class Robot:
    x: float
    y: float
    theta: float
    home_x: int
    home_y: int
    color: tuple


class MapWidget(QWidget):

    def __init__(self, logger):
        super().__init__()

        self.setMinimumSize(MAP_W, MAP_H)

        self.mode = ""

        self.obstacles = set()
        self.robots = []
        self.logs = logger


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.draw_background(painter)
        self.draw_grid(painter)
        self.draw_axes(painter)
        self.draw_obstacles(painter)
        self.draw_homes(painter)
        self.draw_robots(painter)

    def draw_background(self, painter):
        painter.fillRect(self.rect(), QColor(245, 245, 245))

    def draw_grid(self, painter):
        pen = QPen(QColor(210, 210, 210))
        pen.setWidth(1)

        painter.setPen(pen)

        # vertical
        for x in range(GRID_W + 1):
            px = x * CELL_SIZE
            painter.drawLine(px, 0, px, MAP_H)

        # horizontal
        for y in range(GRID_H + 1):
            py = y * CELL_SIZE
            painter.drawLine(0, py, MAP_W, py)

    def draw_axes(self, painter):
        pen = QPen(QColor(100, 100, 100))
        pen.setWidth(2)
        painter.setPen(pen)

        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)

        for x in range(GRID_W):
            px = x * CELL_SIZE
            painter.drawText(px + 4, 18,str(x))

        for y in range(GRID_H):
            py = y * CELL_SIZE
            painter.drawText(5, py + 15,str(y))

    def draw_obstacles(self, painter):
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(40, 40, 40, 120))

        for ox, oy in self.obstacles:
            rect = QRectF(
                ox * CELL_SIZE,
                oy * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )
            painter.drawRect(rect)

    def draw_homes(self, painter):
        for robot in self.robots:
            color = QColor(*robot.color)
            painter.setBrush(color)
            painter.setPen(QPen(Qt.GlobalColor.black, 2))

            rect = QRectF(
                robot.home_x * CELL_SIZE,
                robot.home_y * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )

            painter.drawRect(rect)

    def draw_robots(self, painter):
        for robot in self.robots:
            px = robot.x * CELL_SIZE
            py = robot.y * CELL_SIZE

            cx = px + CELL_SIZE / 2
            cy = py + CELL_SIZE / 2

            color = QColor(*robot.color)

            painter.setBrush(color)
            painter.setPen(QPen(Qt.GlobalColor.black, 2))

            painter.drawEllipse(
                QPointF(cx, cy),
                CELL_SIZE * 0.3,
                CELL_SIZE * 0.3,
            )

            arrow_len = CELL_SIZE * 0.5

            ex = cx + math.cos(robot.theta) * arrow_len
            ey = cy + math.sin(robot.theta) * arrow_len

            painter.setPen(QPen(Qt.GlobalColor.black, 3))
            painter.drawLine(int(cx), int(cy), int(ex), int(ey))

            # arrow head
            angle = robot.theta

            left = QPointF(
                ex - math.cos(angle - 0.5) * 5,
                ey - math.sin(angle - 0.5) * 5,
            )

            right = QPointF(
                ex - math.cos(angle + 0.5) * 5,
                ey - math.sin(angle + 0.5) * 5,
            )

            triangle = QPolygonF([QPointF(ex, ey), left, right,])

            painter.setBrush(Qt.GlobalColor.black)
            painter.drawPolygon(triangle)

    def mousePressEvent(self, event):
        if self.mode == "":
            return

        gx = int(event.position().x() // CELL_SIZE)
        gy = int(event.position().y() // CELL_SIZE)

        if not (0 <= gx < GRID_W and 0 <= gy < GRID_H):
            self.mode = ""
            return

        if self.mode == "draw":
            self.obstacles.add((gx, gy))
            self.update()

        elif self.mode == "erase":
            self.obstacles.discard((gx, gy))
            self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ALLGOHOME Simulator")
        self.resize(1400, 850)

        self.logs = QTextEdit()

        self.map_widget = MapWidget(logger=self.logs)

        self.build_ui()

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        main_layout = QHBoxLayout()
        root.setLayout(main_layout)

        # ==============================================
        # LEFT PANEL
        # ==============================================
        left_panel = QFrame()
        left_panel.setFixedWidth(self.width() // 8)
        left_panel.setFrameShape(QFrame.Shape.StyledPanel)

        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)

        left_title = QLabel("TOOLS")
        left_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        left_layout.addWidget(left_title)

        btn_draw_wall = QPushButton("Draw Walls")
        btn_erase_wall = QPushButton("Erase Walls")
        btn_map_clear = QPushButton("Clear Map")

        robot_group = QGroupBox("New Robot")
        robot_form = QFormLayout()

        new_robot_x = QLineEdit()
        new_robot_y = QLineEdit()
        new_robot_theta = QLineEdit()
        new_robot_home_x = QLineEdit()
        new_robot_home_y = QLineEdit()

        robot_form.addRow("X:", new_robot_x)
        robot_form.addRow("Y:", new_robot_y)
        robot_form.addRow("Theta:", new_robot_theta)
        robot_form.addRow("Home X:", new_robot_home_x)
        robot_form.addRow("Home Y:", new_robot_home_y)

        btn_add_robot = QPushButton("Add Robot")
        row_btn_add_robot = QHBoxLayout()
        row_btn_add_robot.addWidget(btn_add_robot)
        robot_form.addRow(row_btn_add_robot)

        robot_group.setLayout(robot_form)


        left_layout.addWidget(btn_draw_wall)
        left_layout.addWidget(btn_erase_wall)
        left_layout.addWidget(robot_group)
        left_layout.addWidget(btn_map_clear)

        left_layout.addStretch()

        main_layout.addWidget(left_panel)

        # ==============================================
        # CENTER
        # ==============================================
        central_panel = QFrame()
        central_panel.setFrameShape(QFrame.Shape.StyledPanel)
        central_layout = QVBoxLayout()
        central_panel.setLayout(central_layout)

        center_title = QLabel("MAP")
        center_title.setStyleSheet("font-size: 18px; font-weight: bold;")

        central_layout.addWidget(center_title, alignment=Qt.AlignmentFlag.AlignHCenter)

        central_layout.addWidget(self.map_widget, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.logs.setReadOnly(True)
        self.logs.append("LOGS:\n")
        self.logs.setMinimumHeight(150)
        self.logs.setMaximumHeight(300)
        central_layout.addWidget(self.logs)

        main_layout.addWidget(central_panel)

        # ==============================================
        # RIGHT PANEL
        # ==============================================
        right_panel = QFrame()
        right_panel.setFixedWidth(self.width() // 2)
        right_panel.setFrameShape(QFrame.Shape.StyledPanel)

        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        right_title = QLabel("SIMULATION")
        right_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        right_layout.addWidget(right_title, alignment=Qt.AlignmentFlag.AlignHCenter)

        formula_box = QTextEdit()
        formula_box.setReadOnly(True)
        formula_box.setMaximumHeight(220)

        formula_box = QLabel()
        formula_box.setText("""
        <div style="
            padding:12px;
            border-radius:10px;
            font-family: Rubik;
            font-size: 18px;
        ">

        <p><b>State:</b> x = [x, y, θ]</p>
        <p><b>Control:</b> u = [v, ω]</p>

        <pre style="margin-top:6px;">
        xₖ₊₁ = xₖ + v·cos(θₖ)·Δt
        yₖ₊₁ = yₖ + v·sin(θₖ)·Δt
        θₖ₊₁ = θₖ + ω·Δt
        </pre>

        </div>
        """)

        formula_box.setWordWrap(True)
        right_layout.addWidget(formula_box)


        btn_simulation_step = QPushButton("Next Step")
        right_layout.addWidget(btn_simulation_step)

        # ТУТ БУДЕТ MATPLOTLIB ГРАФИК

        right_layout.addStretch()

        main_layout.addWidget(right_panel)

        # ==============================================
        # SIGNALS
        # ==============================================

        def draw_wall_callback():
            if self.map_widget.mode == "draw":
                self.unsetCursor()
                self.map_widget.mode = ""
                return

            self.map_widget.mode = "draw"
            pixmap = QPixmap("cursors/pen-solid.png")
            pixmap = pixmap.scaled(16, 16, Qt.AspectRatioMode.KeepAspectRatio)
            cursor = QCursor(pixmap, 0, 16)
            self.setCursor(cursor)

        btn_draw_wall.clicked.connect(draw_wall_callback)

        def erase_wall_callback():
            if self.map_widget.mode == "erase":
                self.unsetCursor()
                self.map_widget.mode = ""
                return

            self.map_widget.mode = "erase"
            pixmap = QPixmap("cursors/eraser-solid.png")
            pixmap = pixmap.scaled(16, 16, Qt.AspectRatioMode.KeepAspectRatio)
            cursor = QCursor(pixmap, 0, 8)
            self.setCursor(cursor)

        btn_erase_wall.clicked.connect(erase_wall_callback)

        def add_robot_callback():
            try:
                new_robot = Robot(
                    int(new_robot_x.text()),
                    int(new_robot_y.text()),
                    float(new_robot_theta.text()),
                    int(new_robot_home_x.text()),
                    int(new_robot_home_y.text()),
                    (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                )
                self.map_widget.robots.append(new_robot)
                self.map_widget.update()
                self.logs.append(f">>> ROBOT ADDED: {new_robot}")
            except Exception as e:
                QMessageBox.warning(self, "Invalid input", f"{e}")
                return

        btn_add_robot.clicked.connect(add_robot_callback)

        def clear_callback():
            self.map_widget.mode = ""
            self.unsetCursor()
            self.map_widget.obstacles.clear()
            self.map_widget.robots.clear()
            self.map_widget.update()

        btn_map_clear.clicked.connect(clear_callback)



if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())