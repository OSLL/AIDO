import numpy as np


class PID:

    def __init__(self, parameters):
        self.parameters = parameters
        self.d_I = 0.0
        self.phi_I = 0.0
        self.prev_d_err = 0.0
        self.prev_phi_err = 0.0
        self.r = 0.0318
        self.gain = 1
        self.base_line = 0.1
        self.limit = 1

    def update_parameters(self, parameters):

        self.parameters = parameters

    def compute_control_action(self, d_err, phi_err, dt, wheels_cmd_exec=[True, True], stop_line_distance=None):
        if dt is not None:
            self.integrate_errors(d_err, phi_err, dt)

        self.d_I = self.adjust_integral(
            d_err, self.d_I, self.parameters["~integral_bounds"]["d"], self.parameters["~d_resolution"]
        )
        self.phi_I = self.adjust_integral(
            phi_err,
            self.phi_I,
            self.parameters["~integral_bounds"]["phi"],
            self.parameters["~phi_resolution"],
        )

        self.reset_if_needed(d_err, phi_err, wheels_cmd_exec)

        omega = (
            self.parameters["~k_d"] * d_err
            + self.parameters["~k_theta"] * phi_err
            + self.parameters["~k_Id"] * self.d_I
            + self.parameters["~k_Iphi"] * self.phi_I
        )

        self.prev_d_err = d_err
        self.prev_phi_err = phi_err

        v = self.compute_velocity(stop_line_distance)

        v1 = ((v + 0.5 * omega * self.base_line) / self.r)/27
        v2 = ((v - 0.5 * omega * self.base_line) / self.r)/27
        u_r_limited = max(min(v1, self.limit), -self.limit)
        u_l_limited = max(min(v2, self.limit), -self.limit)
        return np.array([u_r_limited, u_l_limited])

    def compute_velocity(self, stop_line_distance):
        if stop_line_distance is None:
            return self.parameters["~v_bar"]
        else:

            d1, d2 = (
                self.parameters["~stop_line_slowdown"]["start"],
                self.parameters["~stop_line_slowdown"]["end"],
            )
            c = (0.5 * (d1 - stop_line_distance) + (stop_line_distance - d2)) / (d1 - d2)
            v_new = self.parameters["~v_bar"] * c
            v = np.max(
                [self.parameters["~v_bar"] / 2.0, np.min([self.parameters["~v_bar"], v_new])]
            )
            return v

    def integrate_errors(self, d_err, phi_err, dt):
        self.d_I += d_err * dt
        self.phi_I += phi_err * dt

    def reset_if_needed(self, d_err, phi_err, wheels_cmd_exec):
        if np.sign(d_err) != np.sign(self.prev_d_err):
            self.d_I = 0
        if np.sign(phi_err) != np.sign(self.prev_phi_err):
            self.phi_I = 0
        if wheels_cmd_exec[0] == 0 and wheels_cmd_exec[1] == 0:
            self.d_I = 0
            self.phi_I = 0

    @staticmethod
    def adjust_integral(error, integral, bounds, resolution):
        if integral > bounds["top"]:
            integral = bounds["top"]
        elif integral < bounds["bot"]:
            integral = bounds["bot"]
        elif abs(error) < resolution:
            integral = 0
        return integral