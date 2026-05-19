import random

import numpy as np
import math

def normalize_angle(angle):
    """
    angle -> [-pi, pi]
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

class EKF:
    def __init__(self, x0, P0, Q, R):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        # self.P = np.eye(3) * 0.2 if P0 is None else np.array(P0, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)

    def predict(self, u, dt):
        v, omega = u
        x, y, th = self.x
        th_new = normalize_angle(th + omega * dt)
        x_new = x + v * math.cos(th) * dt
        y_new = y + v * math.sin(th) * dt
        self.x = np.array([x_new, y_new, th_new], dtype=float)

        F = np.array([
            [1.0, 0.0, 0],
            [0.0, 1.0,  0],
            [0.0, 0.0, 1.0],
        ], dtype=float)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        if z is None:
            return
        z = np.array(z, dtype=float)
        H = np.eye(3)
        y = z - self.x
        y[2] = normalize_angle(y[2])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = normalize_angle(self.x[2])
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P


class UKF:
    def __init__(self, x0, P0, Q, R, alpha=0.5, beta=2.0, kappa=0.0):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        #self.P = np.eye(3) * 0.2 if P0 is None else np.array(P0, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def _sigma_points(self, x, P):
        n = len(x)
        lam = self.alpha ** 2 * (n + self.kappa) - n
        c = n + lam
        try:
            S = np.linalg.cholesky(c * P)
        except np.linalg.LinAlgError:
            S = np.linalg.cholesky(c * (P + 1e-6 * np.eye(n)))

        pts = [x]
        for i in range(n):
            pts.append(x + S[:, i])
            pts.append(x - S[:, i])

        Wm = np.full(2 * n + 1, 1.0 / (2.0 * c))
        Wc = np.full(2 * n + 1, 1.0 / (2.0 * c))
        Wm[0] = lam / c
        Wc[0] = lam / c + (1 - self.alpha ** 2 + self.beta)
        return np.array(pts), Wm, Wc

    def _mean_state(self, pts, Wm):
        x = np.sum(pts[:, 0] * Wm)
        y = np.sum(pts[:, 1] * Wm)
        sin_sum = np.sum(np.sin(pts[:, 2]) * Wm)
        cos_sum = np.sum(np.cos(pts[:, 2]) * Wm)
        th = math.atan2(sin_sum, cos_sum)
        return np.array([x, y, th], dtype=float)

    def predict(self, u, dt):
        v, omega = u
        pts, Wm, Wc = self._sigma_points(self.x, self.P)
        pred_pts = []
        for p in pts:
            x, y, th = p
            x2 = x + v * math.cos(th) * dt
            y2 = y + v * math.sin(th) * dt
            th2 = normalize_angle(th + omega * dt)
            pred_pts.append([x2, y2, th2])
        pred_pts = np.array(pred_pts, dtype=float)
        x_pred = self._mean_state(pred_pts, Wm)
        P_pred = self.Q.copy()
        for i in range(pred_pts.shape[0]):
            d = pred_pts[i] - x_pred
            d[2] = normalize_angle(d[2])
            P_pred += Wc[i] * np.outer(d, d)
        self.x = x_pred
        self.P = P_pred

    def update(self, z):
        if z is None:
            return
        z = np.array(z, dtype=float)
        pts, Wm, Wc = self._sigma_points(self.x, self.P)
        Z = pts.copy()
        z_mean = self._mean_state(Z, Wm)

        Pzz = self.R.copy()
        Pxz = np.zeros((3, 3), dtype=float)
        for i in range(Z.shape[0]):
            dz = Z[i] - z_mean
            dz[2] = normalize_angle(dz[2])
            dx = pts[i] - self.x
            dx[2] = normalize_angle(dx[2])
            Pzz += Wc[i] * np.outer(dz, dz)
            Pxz += Wc[i] * np.outer(dx, dz)

        y = z - z_mean
        y[2] = normalize_angle(y[2])
        K = Pxz @ np.linalg.inv(Pzz)
        self.x = self.x + K @ y
        self.x[2] = normalize_angle(self.x[2])
        self.P = self.P - K @ Pzz @ K.T


class ParticleFilter:
    def __init__(self, x0, num_particles, POS_STD, THETA_STD, MEAS_POS_STD, MEAS_THETA_STD):
        self.n = num_particles
        self.particles = np.zeros((self.n, 3), dtype=float)
        self.weights = np.ones(self.n, dtype=float) / self.n
        self.init_around(x0)

        self.POS_STD = POS_STD
        self.THETA_STD = THETA_STD
        self.MEAS_POS_STD = MEAS_POS_STD
        self.MEAS_THETA_STD = MEAS_THETA_STD

    def init_around(self, x0):
        x, y, th = x0
        self.particles[:, 0] = np.random.normal(x, 0.20, self.n)
        self.particles[:, 1] = np.random.normal(y, 0.20, self.n)
        self.particles[:, 2] = np.array([normalize_angle(t) for t in np.random.normal(th, 0.12, self.n)])
        self.weights[:] = 1.0 / self.n

    def predict(self, u, dt):
        v, omega = u
        noise_xy = np.random.normal(0.0, self.POS_STD, (self.n, 2))
        noise_th = np.random.normal(0.0, self.THETA_STD, self.n)
        for i in range(self.n):
            x, y, th = self.particles[i]
            x += v * math.cos(th) * dt + noise_xy[i, 0]
            y += v * math.sin(th) * dt + noise_xy[i, 1]
            th = normalize_angle(th + omega * dt + noise_th[i])
            self.particles[i] = [x, y, th]

    def update(self, z):
        if z is None:
            return

        z = np.array(z, dtype=float)

        dx = self.particles[:, 0] - z[0]
        dy = self.particles[:, 1] - z[1]
        dth = np.array([normalize_angle(p - z[2]) for p in self.particles[:, 2]])

        pos_var = self.MEAS_POS_STD ** 2
        th_var = self.MEAS_THETA_STD ** 2

        pos_ll = np.exp(-0.5 * (dx ** 2 + dy ** 2) / pos_var)
        th_ll = np.exp(-0.5 * (dth ** 2) / th_var)

        likelihood = pos_ll * th_ll

        self.weights *= likelihood + 1e-300

        s = np.sum(self.weights)
        if not np.isfinite(s) or s <= 0:
            self.weights[:] = 1.0 / self.n
        else:
            self.weights /= s

        self.resample_if_needed()

    def resample_if_needed(self):
        neff = 1.0 / np.sum(self.weights ** 2)
        if neff > self.n * 0.6:
            return
        positions = (np.arange(self.n) + random.random()) / self.n
        indexes = np.zeros(self.n, dtype=int)
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < self.n:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.particles = self.particles[indexes]
        self.weights[:] = 1.0 / self.n

    @property
    def x(self):
        w = self.weights
        x = np.sum(self.particles[:, 0] * w)
        y = np.sum(self.particles[:, 1] * w)
        sin_sum = np.sum(np.sin(self.particles[:, 2]) * w)
        cos_sum = np.sum(np.cos(self.particles[:, 2]) * w)
        th = math.atan2(sin_sum, cos_sum)
        return np.array([x, y, th], dtype=float)