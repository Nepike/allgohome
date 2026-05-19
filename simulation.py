import math
import random
import numpy as np

from filters import normalize_angle, EKF, UKF, ParticleFilter

from config import *




class Simulation:
    def __init__(self, robot: Robot, grid_w, grid_h, cell_size, blocked,
                 dt, angular_speed_k,
                 measurements_pos_std, measurements_theta_std, measurements_drop_prob,
                 Kalman_Q, Kalman_R, PF_pos_std, PF_theta_std):


        self.grid_w = grid_w
        self.grid_h = grid_h
        self.cell_size = cell_size
        self.blocked = blocked

        self.dt = dt
        self.angular_speed_k = angular_speed_k

        self.measurements_pos_std = measurements_pos_std * cell_size
        self.measurements_theta_std = measurements_theta_std
        self.measurements_drop_prob = measurements_drop_prob


        self.robot = robot
        self.waypoint_index = 0
        self.main_filter = "UKF"
        self.done = False

        self.ekf = EKF(np.array([self.robot.x, self.robot.y, self.robot.theta]), np.eye(3) * 0.2, Kalman_Q, Kalman_R)
        self.ukf = UKF(np.array([self.robot.x, self.robot.y, self.robot.theta]), np.eye(3) * 0.2, Kalman_Q, Kalman_R)
        self.pf = ParticleFilter(np.array([self.robot.x, self.robot.y, self.robot.theta]),
                                 100, PF_pos_std, PF_theta_std, measurements_pos_std, measurements_theta_std)

    def a_star(self, start, goal):
        if start in self.blocked or goal in self.blocked:
            return None

        def h(c):
            # Manhattan distance
            return abs(c[0] - goal[0]) + abs(c[1] - goal[1])

        open_set = {start}
        came_from = {}
        g = {start: 0.0}
        f = {start: h(start)} # f = g + h

        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while open_set:
            current = min(open_set, key=lambda c: f.get(c, float('inf')))
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            open_set.remove(current)

            for dx, dy in neighbors:
                nb = (current[0] + dx, current[1] + dy)
                if not (0 <= nb[0] < self.grid_w and 0 <= nb[1] < self.grid_h):
                    continue
                if nb in self.blocked:
                    continue

                tentative_g = g[current] + 1.0
                if tentative_g < g.get(nb, float('inf')):
                    came_from[nb] = current
                    g[nb] = tentative_g
                    f[nb] = tentative_g + h(nb)
                    open_set.add(nb)
        return None


    def camera_measurement(self):
        if random.random() < self.measurements_drop_prob:
            return None
        x, y, th = self.robot.x, self.robot.y, self.robot.theta
        mx = x + random.gauss(0, self.measurements_pos_std)
        my = y + random.gauss(0, self.measurements_pos_std)
        mth = normalize_angle(th + random.gauss(0, self.measurements_theta_std))
        return np.array([mx, my, mth], dtype=float)

    def controller(self, x_est, y_est, theta_est): # -> v, omega
        if not self.robot.astar_path or self.done:
            return 0.0, 0.0

        while self.waypoint_index < len(self.robot.astar_path):
            cell_x, cell_y = self.robot.astar_path[self.waypoint_index]
            target_x = (cell_x + 0.5) * self.cell_size
            target_y = (cell_y + 0.5) * self.cell_size
            distance = math.hypot(target_x - x_est, target_y - y_est)
            if distance < WAYPOINT_THRESHOLD * self.cell_size and self.waypoint_index < len(self.robot.astar_path) - 1:
                self.waypoint_index += 1
                continue
            break

        if self.waypoint_index >= len(self.robot.astar_path):
            return 0.0, 0.0

        cell_x, cell_y = self.robot.astar_path[self.waypoint_index]
        target_x = (cell_x + 0.5) * self.cell_size
        target_y = (cell_y + 0.5) * self.cell_size
        dx = target_x - x_est
        dy = target_y - y_est

        desired_theta = math.atan2(dy, dx)
        angle_error = normalize_angle(desired_theta - theta_est)

        omega = self.angular_speed_k * angle_error
        omega = np.clip(omega, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)

        if abs(angle_error) < ANGLE_THRESHOLD:
            v = MAX_LINEAR_SPEED * self.cell_size
        else:
            v = MAX_LINEAR_SPEED * self.cell_size * max(0.0, math.cos(angle_error))
            if abs(angle_error) > 1.0:
                v = 0.0
        return v, omega

    def step(self):
        if self.done:
            print("done") # TODO handler
            return

        if not self.robot.astar_path:
            x_cell = int(self.robot.x // self.cell_size)
            y_cell = int(self.robot.y // self.cell_size)
            self.robot.astar_path = self.a_star((x_cell, y_cell), (self.robot.home_x, self.robot.home_y))
            self.robot.true_path.append((self.robot.x, self.robot.y, self.robot.theta))
            if not self.robot.astar_path:
                raise ValueError # TODO нормальное логирование и исключения

        if self.main_filter == "EKF":
            x_est, y_est, th_est = self.ekf.x # x - это вектор (x,y,th) # TODO переименовать в X
        elif self.main_filter == "UKF":
            x_est, y_est, th_est = self.ukf.x
        else:
            x_est, y_est, th_est = self.pf.x

        v, omega = self.controller(x_est, y_est, th_est)

        # true motion
        x, y, th = self.robot.x, self.robot.y, self.robot.theta
        x = x + v * math.cos(th) * self.dt
        y = y + v * math.sin(th) * self.dt
        th = normalize_angle(th + omega * self.dt)
        self.robot.x, self.robot.y, self.robot.theta = x, y, th
        self.robot.true_path.append((self.robot.x, self.robot.y, self.robot.theta))

        # measurements
        meas = self.camera_measurement()
        if meas is not None:
            self.robot.measurements.append(meas.copy())

        # filters
        u = (v, omega)
        self.ekf.predict(u, self.dt)
        self.ukf.predict(u, self.dt)
        self.pf.predict(u, self.dt)
        self.ekf.update(meas)
        self.ukf.update(meas)
        self.pf.update(meas)

        self.robot.ekf_path.append(self.ekf.x.copy())
        self.robot.ukf_path.append(self.ukf.x.copy())
        self.robot.pf_path.append(self.pf.x.copy())

        cell_x, cell_y = self.robot.astar_path[-1]
        goal_x = (cell_x + 0.5) * self.cell_size
        goal_y = (cell_y + 0.5) * self.cell_size

        if math.hypot(goal_x - self.robot.x, goal_y - self.robot.y) < WAYPOINT_THRESHOLD * 0.5 * self.cell_size:
            self.done = True
            return







