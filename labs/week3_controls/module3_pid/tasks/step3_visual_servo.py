"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 2/3 Lab — Step 3: Visual Servoing (Vision + PID)
Capstone: use a PID loop on the camera pixel error to keep a glowing
gate centered by yawing. Combines Week 2 vision with Week 3 control.
"""

import drone_core
import drone_utils as uav_utils
import cv2
import numpy as np

# -- Course setup: makes the shared `neo_lab` helper importable.
#    You don't need to read or change this block. --
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.realpath(__file__))
while _os.path.basename(_d) != "labs" and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
if _d not in _sys.path:
    _sys.path.insert(0, _d)
import neo_lab

# -- Constants --------------------------------------------------------------
V_MIN = 200
MIN_AREA = 500
COL_CENTER = 320
KP = 0.35
KI = 0.0
KD = 0.2
MAX_YAW = 0.25
SEARCH_YAW = 0.2
CENTER_TOL = 0.15    # normalized error considered centered
HOLD_TIME = 1.0

# -- Module-level state -----------------------------------------------------
_err_int = 0.0
_prev_err = 0.0
_target_col = None
_hold = 0.0
_done = False

def pid_control(err, err_int, err_dot, kp, ki, kd):
    """Return the PID controller output from the three gain terms (see README, Key terms)."""
    ##################################
    #### START PUT CODE HERE #########
    output = kp * err + ki * err_int + kd * err_dot
    ###### END PUT CODE HERE #########
    ##################################
    return output

def reset():
    global _err_int, _prev_err, _target_col, _hold, _done
    _err_int = 0.0
    _prev_err = 0.0
    _target_col = None
    _hold = 0.0
    _done = False


def update(drone):
    global _err_int, _prev_err, _target_col, _hold, _done
    if _done:
        return True
    ##################################
    #### START PUT CODE HERE #########

    dt = drone.get_delta_time()
    image = drone.camera.get_color_image()

    gate = neo_lab.detect_gate(image)

    if gate is None:
        drone.flight.send_pcmd(0, 0, SEARCH_YAW, 0)
        _err_int = 0.0
        _hold = 0.0
        return False

    col = gate.cx
    error = (col - COL_CENTER) / COL_CENTER

    _err_int = uav_utils.clamp(_err_int + error * dt, -1.0, 1.0)
    err_dot = (error - _prev_err) / dt if dt > 0 else 0.0
    _prev_err = error

    yaw = uav_utils.clamp(
        pid_control(error, _err_int, err_dot, KP, KI, KD),
        -MAX_YAW, MAX_YAW
    )
    drone.flight.send_pcmd(0, 0, yaw, 0)

    if abs(error) < CENTER_TOL:
        _hold += dt
    elif abs(error) > 2.0 * CENTER_TOL:
        _hold = 0.0
 
    if _hold >= HOLD_TIME:
        drone.flight.stop()
        print("[Step 3] Locked onto the gate")
        _done = True

    ###### END PUT CODE HERE #########
    ##################################
    return _done


if __name__ == "__main__":
    _drone = drone_core.create_drone()
    _launcher = neo_lab.Launcher(3.0)

    def start():
        _launcher.reset()
        reset()
        print("Step 3: Visual Servoing (Vision + PID)")

    def _update():
        if not _launcher.done:        # arm + climb to a safe height first
            _launcher.update(_drone)
            return
        if update(_drone):
            _drone.flight.land()

    _drone.set_start_update(start, _update)
    _drone.go()
