import time
from .PID import PID


class Controller:
    def __init__(self):
        self.last_t = None
        self.params = {
            "~v_bar": 0.19,
            "~k_d": -6.0,
            "~k_theta": -5,
            "~k_Id": -0.3,
            "~k_Iphi": 0.0,
            "~theta_thres": 0.523,
            "~d_thres": 0.2615,
            "~d_offset": 0.005,
            "~omega_ff": 0,
            "~integral_bounds": {
                "d": {
                    "top": 0.3,
                    "bot": -0.3
                },
                "phi": {
                    "top": 1.2,
                    "bot": -1.2
                }
            },
            "~d_resolution": 0.011,
            "~phi_resolution": 0.051,
            "~stop_line_slowdown":{
                "start": 0.6,
                "end": 0.15
            }
        }
        self.controller = PID(self.params)

    def compute_action(self, pose_msg):

        d, phi = pose_msg
        dt = None
        current_t = time.time()
        if self.last_t is not None:
            dt = current_t - self.last_t
        d_err = d - self.params["~d_offset"]
        phi_err = phi
        action = self.controller.compute_control_action(
            d_err, phi_err, dt)
        return action








