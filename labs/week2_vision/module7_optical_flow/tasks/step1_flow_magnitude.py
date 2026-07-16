"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 2 · Module 7 — Step 1: Optical Flow Magnitude
Optical flow measures how the image moved between two frames. With the downward
camera, that motion comes from the drone moving over the ground, so it is a
vision-only way to sense speed. We use SPARSE flow (track a few corner features,
not every pixel) and only process every SKIP-th frame, so the camera work stays
light and the simulator runs smoothly.
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
PROBE_PITCH = 0.10     # gentle forward drift so there is motion to measure
HOVER_TIME = 4.0
SKIP = 2               # do the vision work every Nth frame (image pull + flow are the cost)
MIN_PTS = 20           # re-find features when fewer than this survive
# OpenCV parameters for feature detection and tracking (given; you don't tune these).
FEATURE_PARAMS = dict(maxCorners=80, qualityLevel=0.01, minDistance=8, blockSize=7)
LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# -- Module-level state -----------------------------------------------------
_prev_gray = None
_prev_pts = None
_timer = 0.0
_frame = 0
_last_mag = 0.0
_done = False

def reset():
    global _prev_gray, _prev_pts, _timer, _frame, _last_mag, _done
    _prev_gray = None
    _prev_pts = None
    _timer = 0.0
    _frame = 0
    _last_mag = 0.0
    _done = False


def update(drone):
    global _prev_gray, _prev_pts, _timer, _frame, _last_mag, _done
    if _done:
        return True
    ##################################
    #### START PUT CODE HERE #########

    drone.flight.send_pcmd(PROBE_PITCH, 0, 0, 0)
    _timer += drone.get_delta_time()
    _frame += 1
    if _frame % SKIP == 0:
        gray = cv2.cvtColor(drone.camera.get_downward_image(), cv2.COLOR_BGR2GRAY)
        if _prev_gray is None or _prev_pts is None or len(_prev_pts) < MIN_PTS:
            _prev_pts = cv2.goodFeaturesToTrack(gray, **FEATURE_PARAMS)
        else:
            new_pts, status, _err = cv2.calcOpticalFlowPyrLK(_prev_gray, gray,
                                                         _prev_pts, None, **LK_PARAMS)
            if new_pts is not None and status is not None:
                keep = status.flatten() == 1
                good_new = new_pts[keep].reshape(-1, 2)
                good_old = _prev_pts[keep].reshape(-1, 2)
                if len(good_new) > 0:
                    disp = good_new - good_old
                    _last_mag = float(np.mean(np.sqrt(disp[:, 0] ** 2 + disp[:, 1] ** 2)))
                _prev_pts = good_new.reshape(-1, 1, 2)
        _prev_gray = gray
    if _timer >= HOVER_TIME:
        drone.flight.stop()
        print(f"[Step 1] mean sparse flow magnitude = {_last_mag:.3f} px/interval")
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
        print("Step 1: Optical Flow Magnitude")

    def _update():
        if not _launcher.done:        # arm + climb to a safe height first
            _launcher.update(_drone)
            return
        if update(_drone):
            _drone.flight.land()

    _drone.set_start_update(start, _update)
    _drone.go()
