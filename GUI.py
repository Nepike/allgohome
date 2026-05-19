import random

import numpy as np
import math

from PyQt6.QtCore import Qt, QRectF, QPointF, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF, QPixmap, QCursor, QBrush
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QListWidget, QMainWindow,  QPushButton, QVBoxLayout, \
    QWidget, QFrame, QTextEdit, QGroupBox, QFormLayout, QLineEdit, QMessageBox, QScrollArea

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from filters import normalize_angle
from simulation import Simulation

from config import *


class MapWidget(QWidget):
    def __init__(self, grid_w, grid_h, cell_size, logger):
        super().__init__()

        self.grid_w = grid_w
        self.grid_h = grid_h
        self.cell_size = cell_size
        self.map_w = grid_w * cell_size
        self.map_h = grid_h * cell_size
        self.logs = logger

        self.setMinimumSize(self.map_w, self.map_h)
        self.current_action = ""

        self.obstacles = set()
        self.inflated = set()
        self.robots: list[Robot] = []
        self.active_robot_index = 0


    def inflate_obstacles(self, radius=1):
        inflated = set()
        for ox, oy in self.obstacles:
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx, ny = ox + dx, oy + dy
                    if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                        inflated.add((nx, ny))
        self.inflated = inflated

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.draw_background(painter)
        self.draw_grid(painter)
        self.draw_axes(painter)
        self.draw_obstacles(painter)
        self.draw_homes(painter)
        self.draw_paths(painter)
        self.draw_robots(painter)

    def draw_background(self, painter):
        painter.fillRect(self.rect(), QColor(245, 245, 245))

    def draw_grid(self, painter):
        pen = QPen(QColor(210, 210, 210))
        pen.setWidth(1)
        painter.setPen(pen)

        for x in range(self.grid_w + 1):
            px = x * self.cell_size
            painter.drawLine(px, 0, px, self.map_h)
        for y in range(self.grid_h + 1):
            py = y * self.cell_size
            painter.drawLine(0, py, self.map_w, py)

    def draw_axes(self, painter):
        pen = QPen(QColor(100, 100, 100))
        pen.setWidth(1)
        painter.setPen(pen)
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        for x in range(self.grid_w):
            painter.drawText(x * self.cell_size + 3, 12, str(x))
        for y in range(self.grid_h):
            painter.drawText(2, y * self.cell_size + 20, str(y))

    def draw_obstacles(self, painter):
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(40, 40, 40, 120))
        for ox, oy in self.obstacles:
            painter.drawRect(QRectF(ox * self.cell_size, oy * self.cell_size, self.cell_size, self.cell_size))

        painter.setBrush(QColor(180, 80, 80, 90))
        for ox, oy in self.inflated:
            if (ox, oy) in self.obstacles:
                continue
            painter.drawRect(QRectF(ox * self.cell_size, oy * self.cell_size, self.cell_size, self.cell_size))

    def draw_homes(self, painter):
        for robot in self.robots:
            color = QColor(*robot.color)
            painter.setBrush(color)
            painter.setPen(QPen(Qt.GlobalColor.black, 2))
            painter.drawRect(QRectF(robot.home_x * self.cell_size, robot.home_y * self.cell_size, self.cell_size, self.cell_size))

    def draw_paths(self, painter):
        """
        draws A* path and real trajectory
        """
        for robot in self.robots:
            color = QColor(*robot.color)
            pen = QPen(color)
            pen.setWidth(3)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            pts = []
            for cell in robot.astar_path:
                x, y = cell
                # + 0.5 - center of the cell
                pts.append(QPointF((x+0.5) * self.cell_size, (y+0.5) * self.cell_size))

            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])

            pts = []
            pen = QPen(QColor(0, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for point in robot.true_path:
                x, y, _ = point
                pts.append(QPointF(x, y))

            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])

    def draw_robots(self, painter):
        for robot in self.robots:
            color = QColor(*robot.color)
            painter.setBrush(color)
            painter.setPen(QPen(Qt.GlobalColor.black, 2))
            painter.drawEllipse(QPointF(robot.x, robot.y), self.cell_size * 0.30, self.cell_size * 0.30)

            arrow_len = self.cell_size * 0.55
            ex = robot.x + math.cos(robot.theta) * arrow_len
            ey = robot.y + math.sin(robot.theta) * arrow_len
            painter.setPen(QPen(Qt.GlobalColor.black, 3))
            painter.drawLine(int(robot.x), int(robot.y), int(ex), int(ey))


            left = QPointF(ex - math.cos(robot.theta - 0.5) * 6, ey - math.sin(robot.theta - 0.5) * 6)
            right = QPointF(ex - math.cos(robot.theta + 0.5) * 6, ey - math.sin(robot.theta + 0.5) * 6)
            triangle = QPolygonF([QPointF(ex, ey), left, right])
            painter.setBrush(Qt.GlobalColor.black)
            painter.drawPolygon(triangle)

    def mousePressEvent(self, event):
        if self.current_action == "":
            return

        gx = int(event.position().x() // self.cell_size)
        gy = int(event.position().y() // self.cell_size)

        if not (0 <= gx < self.grid_w and 0 <= gy < self.grid_h):
            self.current_action = ""
            return

        if self.current_action == "draw":
            self.obstacles.add((gx, gy))
            self.inflate_obstacles()
            self.update()
        elif self.current_action == "erase":
            self.obstacles.discard((gx, gy))
            self.inflate_obstacles()
            self.update()

    @property
    def blocked(self):
        return self.obstacles | self.inflated


class FiltersPlot(FigureCanvas):
    def __init__(self, grid_w, grid_h, cell_size, dt, parent=None):
        self.fig = Figure(figsize=(8, 9), constrained_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)

        self.grid_w = grid_w
        self.grid_h = grid_h
        self.cell_size = cell_size
        self.dt = dt

        gs = GridSpec(3,1,figure=self.fig, height_ratios=[3, 1, 1])
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax2 = self.fig.add_subplot(gs[1])
        self.ax3 = self.fig.add_subplot(gs[2])

        self.filter_mode = "EKF"

        self.toolbar = NavigationToolbar(self, parent)
        self.toolbar.update()

    def get_toolbar(self):
        return self.toolbar

    def plot_robot(self, robot: Robot):

        def pixels_to_cells(path):
            if not path:
                return np.empty((0, 2))
            arr = np.asarray(path, dtype=float)
            if arr.size == 0:
                return arr
            res = arr.copy()
            if res.ndim == 2 and res.shape[1] >= 2:
                res[:, :2] /= self.cell_size
            return res

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        astar_path = np.array([(x + 0.5, y + 0.5) for x, y in robot.astar_path])
        true_path = pixels_to_cells(robot.true_path)
        measurements = pixels_to_cells(robot.measurements)

        if self.filter_mode == "EKF":
            filter_path = pixels_to_cells(robot.ekf_path)
        elif self.filter_mode == "UKF":
            filter_path = pixels_to_cells(robot.ukf_path)
        else:
            filter_path = pixels_to_cells(robot.pf_path)

        # 1) Trajectory plot
        self.ax1.set_title("Trajectories")
        if len(astar_path):
            self.ax1.plot(astar_path[:, 0], astar_path[:, 1], linestyle='--', linewidth=2, label='A* path')
        if len(true_path):
            self.ax1.plot(true_path[:, 0], true_path[:, 1], linewidth=2, label='True trajectory')
        if len(measurements):
            self.ax1.scatter(measurements[:, 0], measurements[:, 1], s=12, alpha=0.45, label='Camera measurements')
        if len(filter_path):
            self.ax1.plot(filter_path[:, 0], filter_path[:, 1], label=f'{self.filter_mode} estimate')

        self.ax1.set_xlim(0, self.grid_w)
        self.ax1.set_ylim(self.grid_h, 0)
        self.ax1.set_aspect('equal')
        self.ax1.grid(True, alpha=0.25)
        self.ax1.legend(fontsize=8, loc='best')

        n = max(len(true_path), len(filter_path))
        t = np.arange(n) * self.dt

        def pos_err(path):
            m = min(len(path), len(true_path))
            d = np.sqrt((true_path[:m, 0] - path[:m, 0]) ** 2 + (true_path[:m, 1] - path[:m, 1]) ** 2)
            return d

        def theta_err(path):
            m = min(len(path), len(true_path))
            d = np.array([abs(normalize_angle(true_path[i, 2] - path[i, 2])) for i in range(m)])
            return d

        # 2) Position error
        ekf_path = pixels_to_cells(robot.ekf_path)
        ukf_path = pixels_to_cells(robot.ukf_path)
        pf_path = pixels_to_cells(robot.pf_path)

        self.ax2.set_title("Position error")
        for hist, label in [(ekf_path, 'EKF'), (ukf_path, 'UKF'), (pf_path, 'PF')]:
            err = pos_err(hist)
            if err is not None:
                self.ax2.plot(t[:len(err)], err, label=label)
        self.ax2.set_ylabel('cells')
        self.ax2.grid(True, alpha=0.25)
        self.ax2.legend(fontsize=8, loc='best')

        # 3) Heading error
        self.ax3.set_title("Heading error")
        for hist, label in [(ekf_path, 'EKF'), (ukf_path, 'UKF'), (pf_path, 'PF')]:
            err = theta_err(hist)
            if err is not None:
                self.ax3.plot(t[:len(err)], err, label=label)
        self.ax3.set_ylabel('rad')
        self.ax3.set_xlabel('time, s')
        self.ax3.grid(True, alpha=0.25)
        self.ax3.legend(fontsize=8, loc='best')

        self.draw_idle()



class MainWindow(QMainWindow):
    def __init__(self, grid_w, grid_h, cell_size, dt):
        super().__init__()
        self.setWindowTitle("ALLGOHOME Simulator")
        self.resize(1600, 900)

        self.grid_w = grid_w
        self.grid_h = grid_h
        self.cell_size = cell_size
        self.dt = dt

        self.logs = QTextEdit()
        self.map_widget = MapWidget(grid_w, grid_h, cell_size, logger=self.logs)
        self.filters_plot = FiltersPlot(grid_w, grid_h, cell_size, dt)
        self.simulation = None

        self.timer = QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.tick)

        self.build_ui()

    def tick(self):
        if not self.simulation or self.simulation.done:
            return

        self.simulation.step()
        self.map_widget.update()
        self.filters_plot.plot_robot(self.simulation.robot)

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

        robot_form.addRow("X (cell):", new_robot_x)
        robot_form.addRow("Y (cell):", new_robot_y)
        robot_form.addRow("Theta (rad):", new_robot_theta)
        robot_form.addRow("Home X:", new_robot_home_x)
        robot_form.addRow("Home Y:", new_robot_home_y)

        btn_add_robot = QPushButton("Add Robot")
        robot_form.addRow(btn_add_robot)
        robot_group.setLayout(robot_form)

        left_layout.addWidget(btn_draw_wall)
        left_layout.addWidget(btn_erase_wall)
        left_layout.addWidget(robot_group)
        left_layout.addWidget(btn_map_clear)
        left_layout.addStretch()

        main_layout.addWidget(left_panel)

        # ==============================================
        # CENTER PANEL
        # ==============================================
        center_panel = QFrame()
        center_panel.setFrameShape(QFrame.Shape.StyledPanel)
        center_layout = QVBoxLayout()
        center_panel.setLayout(center_layout)

        center_title = QLabel("MAP")
        center_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        center_layout.addWidget(center_title, alignment=Qt.AlignmentFlag.AlignHCenter)
        center_layout.addWidget(self.map_widget, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.logs.setReadOnly(True)
        self.logs.append("LOGS:")
        self.logs.setMinimumHeight(160)
        self.logs.setMaximumHeight(260)
        center_layout.addWidget(self.logs)

        main_layout.addWidget(center_panel)

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

        # formula_box = QLabel()
        # formula_box.setText(
        #     """
        #     <div style="padding:12px; border-radius:10px; font-family: Rubik; font-size: 16px;">
        #     <p><b>State:</b> x = [x, y, θ]</p>
        #     <p><b>Control:</b> u = [v, ω]</p>
        #     <pre style="margin-top:6px; white-space:pre-wrap;">
        #     xₖ₊₁ = xₖ + v·cos(θₖ)·Δt
        #     yₖ₊₁ = yₖ + v·sin(θₖ)·Δt
        #     θₖ₊₁ = θₖ + ω·Δt
        #     </pre>
        #     </div>
        #     """
        # )
        # formula_box.setWordWrap(True)
        # formula_box.setMaximumHeight(170)
        # right_layout.addWidget(formula_box)

        row_simulation_buttons = QHBoxLayout()
        btn_start_simulation = QPushButton("Start Simulation")
        btn_pause_simulation = QPushButton("Pause Simulation")
        row_simulation_buttons.addWidget(btn_start_simulation)
        row_simulation_buttons.addWidget(btn_pause_simulation)
        right_layout.addLayout(row_simulation_buttons)

        row_filter_buttons = QHBoxLayout()
        btn_set_filter_ekf = QPushButton("EKF Filter")
        btn_set_filter_ukf = QPushButton("UKF Filter")
        btn_set_filter_pf = QPushButton("PF Filter")

        row_filter_buttons.addWidget(btn_set_filter_ekf)
        row_filter_buttons.addWidget(btn_set_filter_ukf)
        row_filter_buttons.addWidget(btn_set_filter_pf)

        right_layout.addLayout(row_filter_buttons)

        self.filters_plot.setMinimumHeight(620)
        right_layout.addWidget(self.filters_plot)
        right_layout.addWidget(self.filters_plot)
        right_layout.addWidget(self.filters_plot.get_toolbar())
        right_layout.addStretch()

        main_layout.addWidget(right_panel)

        # ==============================================
        # SIGNALS
        # ==============================================
        def draw_wall_callback():
            if self.map_widget.current_action == "draw":
                self.unsetCursor()
                self.map_widget.current_action = ""
                return
            self.map_widget.current_action = "draw"

            pixmap = QPixmap("cursors/pen-solid.png")
            pixmap = pixmap.scaled(16, 16, Qt.AspectRatioMode.KeepAspectRatio)
            cursor = QCursor(pixmap, 0, 16)
            self.setCursor(cursor)

        btn_draw_wall.clicked.connect(draw_wall_callback)

        def erase_wall_callback():
            if self.map_widget.current_action == "erase":
                self.unsetCursor()
                self.map_widget.current_action = ""
                return
            self.map_widget.current_action = "erase"

            pixmap = QPixmap("cursors/eraser-solid.png")
            pixmap = pixmap.scaled(16, 16, Qt.AspectRatioMode.KeepAspectRatio)
            cursor = QCursor(pixmap, 0, 8)
            self.setCursor(cursor)

        btn_erase_wall.clicked.connect(erase_wall_callback)

        def add_robot_callback():
            try:
                new_robot = Robot(
                    (int(new_robot_x.text()) + 0.5 ) * self.cell_size,
                    (int(new_robot_y.text()) + 0.5 ) * self.cell_size,
                    float(new_robot_theta.text()),
                    int(new_robot_home_x.text()),
                    int(new_robot_home_y.text()),
                    (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                )


                if not (0 <= int(new_robot_x.text()) < self.grid_w and
                        0 <= int(new_robot_y.text()) < self.grid_h and
                        0 <= new_robot.home_x < self.grid_w and
                        0 <= new_robot.home_y < self.grid_h):
                    raise ValueError("Coordinates must be inside the map")

                self.map_widget.robots.append(new_robot)
                self.map_widget.update()
                self.logs.append(f">>> ROBOT ADDED: start=({new_robot_x.text()},{new_robot_y.text()}), home=({new_robot.home_x},{new_robot.home_y})")
            except Exception as e:
                QMessageBox.warning(self, "Invalid input", str(e))

        btn_add_robot.clicked.connect(add_robot_callback)


        def start_simulation_callback():
            if self.simulation and self.simulation.done:
                self.map_widget.active_robot_index += 1
                self.simulation = None
                pause_simulation_callback()
            if not self.simulation or self.simulation.done:
                if self.map_widget.active_robot_index < len(self.map_widget.robots):
                    robot = self.map_widget.robots[self.map_widget.active_robot_index]
                    self.simulation = Simulation(
                        robot, self.grid_w, self.grid_h, self.cell_size, self.map_widget.blocked,
                        self.dt, ANGULAR_SPEED_K, MEAS_POS_STD, MEAS_THETA_STD, MEAS_DROP_PROB,
                        KALMAN_Q, KALMAN_R, PF_POS_STD, PF_THETA_STD
                    )
                    robot.true_path.clear()
                    robot.astar_path.clear()
                    robot.measurements.clear()
                    robot.ekf_path.clear()
                    robot.ukf_path.clear()
                    robot.pf_path.clear()
                else:
                    self.logs.append(">>> NO MORE ROBOTS")
                    return

            if not self.timer.isActive():
                self.timer.start()
                self.is_running = True
                self.logs.append(">>> SIMULATION STARTED")

        btn_start_simulation.clicked.connect(start_simulation_callback)

        def pause_simulation_callback():
            if self.timer.isActive():
                self.timer.stop()
                self.is_running = False
                self.logs.append(">>> SIMULATION PAUSED")
            else:
                start_simulation_callback()

        btn_pause_simulation.clicked.connect(pause_simulation_callback)

        def clear_callback():
            pause_simulation_callback()
            self.simulation = None
            self.map_widget.current_action = ""
            self.unsetCursor()
            self.map_widget.obstacles.clear()
            self.map_widget.inflated.clear()
            self.map_widget.robots.clear()
            self.map_widget.active_robot_index = 0
            self.map_widget.update()
            self.filters_plot.ax1.clear()
            self.filters_plot.ax2.clear()
            self.filters_plot.ax3.clear()
            self.filters_plot.draw_idle()
            self.logs.append(">>> MAP CLEARED")

        btn_map_clear.clicked.connect(clear_callback)

        def set_filter_ekf_callback():
            self.filters_plot.filter_mode = "EKF"
            if self.simulation:
                self.filters_plot.plot_robot(self.simulation.robot)
                self.simulation.main_filter = "EKF"
        btn_set_filter_ekf.clicked.connect(set_filter_ekf_callback)

        def set_filter_ukf_callback():
            self.filters_plot.filter_mode = "UKF"
            if self.simulation:
                self.filters_plot.plot_robot(self.simulation.robot)
                self.simulation.main_filter = "UKF"
        btn_set_filter_ukf.clicked.connect(set_filter_ukf_callback)

        def set_filter_pf_callback():
            self.filters_plot.filter_mode = "PF"
            if self.simulation:
                self.filters_plot.plot_robot(self.simulation.robot)
                self.simulation.main_filter = "PF"
        btn_set_filter_pf.clicked.connect(set_filter_pf_callback)
