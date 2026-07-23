"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 2 · Module 6 — Step 2: Estimate Range and Approach
Turn the gate's apparent width into a distance using the pinhole camera model
(Module 1), then fly forward until you are STOP_DIST meters away.

This inverts the projection you wrote in Module 1: a known real width projects to a
pixel width that shrinks with distance, so distance is recoverable from FOCAL_PX,
REAL_GATE_WIDTH, and the measured pixel width. See the README (Key terms).
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
FOCAL_PX = 320.0          # focal length in pixels (~90 deg horizontal FOV, 640 wide)
REAL_GATE_WIDTH = 1.5     # meters: the gate's true outer width
MIN_AREA = 400
COL_CENTER = 320
STOP_DIST = 2.5           # meters: stop once this close
APPROACH_PITCH = 0.15     # forward speed while approaching
MAX_YAW = 0.2             # yaw authority to keep the gate centered
SEARCH_YAW = 0.2          # spin slowly when no gate is seen

# -- Module-level state -----------------------------------------------------
_done = False

def reset():
    global _done
    _done = False


def update(drone):
    global _done
    if _done:
        return True
    ##################################
    #### START PUT CODE HERE #########
    image = drone.camera.get_color_image()
    gate = neo_lab.detect_gate(image)
    if gate is None:
        drone.flight.send_pcmd(0, 0, SEARCH_YAW, 0)
        return False
    distance = FOCAL_PX * REAL_GATE_WIDTH / max(gate.tag_px, 1)
    gate_col = gate.cx
    err = (gate_col - COL_CENTER) / COL_CENTER
    yaw = uav_utils.clamp(err * MAX_YAW, -MAX_YAW, MAX_YAW)
    if distance <= STOP_DIST:
        drone.flight.stop()
        print(f"[Step 2] Reached gate, distance ~ {distance:.2f} m")
        _done = True
        return True
    drone.flight.send_pcmd(APPROACH_PITCH, 0, yaw, 0)

    ###### END PUT CODE HERE #########
    ##################################
    return _done


if __name__ == "__main__":
    _drone = drone_core.create_drone()
    _launcher = neo_lab.Launcher(3.0)

    def start():
        _launcher.reset()
        reset()
        print("Step 2: Estimate Range and Approach")

    def _update():
        if not _launcher.done:        # arm + climb to a safe height first
            _launcher.update(_drone)
            return
        if update(_drone):
            _drone.flight.land()

    _drone.set_start_update(start, _update)
    _drone.go()
