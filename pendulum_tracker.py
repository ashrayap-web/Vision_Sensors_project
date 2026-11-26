#!/usr/bin/env python3
"""
Pendulum angle tracker for the Raspberry Pi HQ Camera (Picamera2 + OpenCV).

Two ways to find the pendulum:
  colour markers (default)  two dots of the same bright colour, one on the pivot, one on the bob
  printed tags (--aruco)    two ArUco or AprilTag tags: id 0 centred on the pivot axle, id 1 on
                            the arm, both mounted with their top edge towards the pivot
The script finds both in every frame and reports the angle of the line between them, measured
from the vertical:

    angle = 0    pendulum hanging straight down (after zeroing with the wind off)
    angle > 0    bob to the right of the pivot in the image

Usage
    python pendulum_tracker.py                          live camera on the CM5
    python pendulum_tracker.py --tune                   sliders to choose the marker colour range
    python pendulum_tracker.py --video clip.mp4         recorded or synthetic video (works on a laptop)
    python pendulum_tracker.py --pivot 640,120          fixed pivot pixel, track only the bob marker
    python pendulum_tracker.py --csv run1.csv           log every frame to a CSV file
    python pendulum_tracker.py --auto-zero 2            use the mean of the first 2 s as zero
    python pendulum_tracker.py --aruco --debug          show why detection is failing
    python pendulum_tracker.py --aruco --pivot-id 2 --arm-id 3 --exposure-us 1500   your 2 cm tags
    python pendulum_tracker.py --speed-k 9.0            also show v = k * sqrt(tan(angle))
    python pendulum_tracker.py --aruco                  track printed ArUco 4x4 tags instead of colour
    python pendulum_tracker.py --aruco --aruco-dict APRILTAG_36h11    AprilTag tags instead

With tags, the angle comes from the line between the two tag centres. If the arm tag is lost
(usually motion blur: use a shorter --exposure-us), that frame is skipped, unless the pivot tag
turns with the arm (--pivot-rotates): then its own rotation stands in for a few frames.
Readings that jump faster than a pendulum can move (--max-rate) are thrown away.

Keys in the video window
    z   set zero to the current averaged angle (do this with the wind OFF)
    e   measure exposure and white balance again (after the lighting or the scene changes)
    s   save a snapshot (PNG)
    q   quit (in --tune mode, also prints the chosen HSV range)

Camera notes for the HQ Camera (IMX477): 1280x720 at up to 50 fps uses the full-width sensor
mode. For 120 fps use --size 1332x990 --fps 120, which crops the centre of the sensor and gives a
narrower view.
"""

import argparse
import csv
import math
import time
from collections import deque

import cv2
import numpy as np


# ----------------------------------------------------------------------------- arguments

def parse_args():
    p = argparse.ArgumentParser(description="Track the angle of a pendulum with two colour markers.")
    p.add_argument("--video", help="read frames from this video file instead of the camera")
    p.add_argument("--size", default="1280x720", help="camera resolution WxH (default 1280x720)")
    p.add_argument("--fps", type=float, default=50, help="camera frame rate (default 50)")
    p.add_argument("--exposure-us", type=int,
                   help="fixed exposure time in microseconds (shorter = less motion blur)")
    p.add_argument("--hsv-low", default="35,80,60", help="lower HSV bound H,S,V (OpenCV: H 0-179)")
    p.add_argument("--hsv-high", default="85,255,255", help="upper HSV bound H,S,V")
    p.add_argument("--min-area", type=int, default=30, help="ignore blobs smaller than this (pixels)")
    p.add_argument("--pivot", help="fixed pivot pixel X,Y; then only the bob marker is tracked")
    p.add_argument("--window", type=float, default=1.0, help="averaging window in seconds (default 1)")
    p.add_argument("--zero", type=float, default=0.0, help="zero offset in degrees")
    p.add_argument("--auto-zero", type=float, metavar="SECONDS",
                   help="set the zero to the mean angle of the first SECONDS (wind off at start)")
    p.add_argument("--speed-k", type=float, help="calibration constant k in v = k*sqrt(tan(angle))")
    p.add_argument("--csv", help="write time and angles for every frame to this CSV file")
    p.add_argument("--tune", action="store_true", help="show sliders and the mask to tune the colour")
    p.add_argument("--aruco", action="store_true", help="track printed ArUco/AprilTag tags instead of colour")
    p.add_argument("--aruco-dict", default="4X4_50",
                   help="tag family: 4X4_50 (default), APRILTAG_36h11, 5X5_50, ...")
    p.add_argument("--pivot-id", type=int, default=0, help="id of the tag on the pivot (default 0)")
    p.add_argument("--arm-id", type=int, default=1, help="id of the tag on the arm (default 1)")
    p.add_argument("--pivot-rotates", action="store_true",
                   help="the pivot tag is fixed to the arm and turns with it, so its rotation can "
                        "stand in when the arm tag is lost (not for a tag fixed to the stand)")
    p.add_argument("--max-rate", type=float, default=600.0,
                   help="fastest believable change in angle, deg/s; bigger jumps are rejected (default 600)")
    p.add_argument("--no-display", action="store_true", help="no windows (for batch runs on videos)")
    p.add_argument("--debug", action="store_true",
                   help="show image brightness, the colour mask, and tag candidates that failed to decode")
    return p.parse_args()


def triple(text):
    return np.array([int(v) for v in text.split(",")], dtype=np.uint8)


# ----------------------------------------------------------------------------- frame sources

class CameraSource:
    """Live frames from the HQ Camera via Picamera2, with exposure and white balance locked."""

    def __init__(self, size, fps, exposure_us=None):
        from picamera2 import Picamera2  # imported here so --video works on a laptop

        self.cam = Picamera2()
        config = self.cam.create_video_configuration(
            main={"size": size, "format": "RGB888"},  # RGB888 arrives in BGR order, as OpenCV expects
            controls={"FrameRate": fps},
        )
        self.cam.configure(config)
        self.cam.start()
        self.exposure_us = exposure_us
        time.sleep(1.5)  # let auto exposure and auto white balance settle
        self.lock()

    def lock(self):
        """Lock exposure, gain and white balance at what auto mode chose for the current scene.
        If they keep changing, the colours change and detection stops working. Point the camera at
        the real scene BEFORE starting, or press e later to measure again."""
        md = self.cam.capture_metadata()
        exposure, gain = md["ExposureTime"], md["AnalogueGain"]
        if self.exposure_us:  # keep the same brightness: shorter exposure, more gain
            gain = float(np.clip(gain * exposure / self.exposure_us, 1.0, 16.0))
            exposure = self.exposure_us
        self.cam.set_controls({
            "AeEnable": False, "AwbEnable": False,
            "ExposureTime": int(exposure), "AnalogueGain": gain,
            "ColourGains": md["ColourGains"],
        })
        print(f"Camera locked: exposure {exposure} us, gain {gain:.2f}, colour gains {md['ColourGains']}")

    def relock(self):
        self.cam.set_controls({"AeEnable": True, "AwbEnable": True})
        time.sleep(1.5)
        self.lock()

    def read(self):
        request = self.cam.capture_request()
        frame = request.make_array("main")
        t = request.get_metadata()["SensorTimestamp"] * 1e-9  # time the frame was captured, seconds
        request.release()
        return frame, t

    def close(self):
        self.cam.stop()


class VideoSource:
    """Frames from a video file. Time comes from the frame number and the file's frame rate."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open video {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"Video {path}: {int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames at {self.fps:g} fps")
        self.count = 0

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        t = self.count / self.fps
        self.count += 1
        return frame, t

    def close(self):
        self.cap.release()


# ----------------------------------------------------------------------------- vision

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def colour_mask(frame_bgr, lo, hi):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    if lo[0] <= hi[0]:
        mask = cv2.inRange(hsv, lo, hi)
    else:  # hue range wraps around 0 (red): combine two ranges
        mask = cv2.inRange(hsv, lo, np.array([179, hi[1], hi[2]], np.uint8)) | \
               cv2.inRange(hsv, np.array([0, lo[1], lo[2]], np.uint8), hi)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)  # remove single-pixel noise


def find_blobs(mask, min_area):
    """Centroids of the colour blobs, largest first. Centroids are sub-pixel (mean of pixels)."""
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    blobs = [(stats[i, cv2.CC_STAT_AREA], centroids[i]) for i in range(1, n)
             if stats[i, cv2.CC_STAT_AREA] >= min_area]
    blobs.sort(key=lambda b: -b[0])
    return [c for _, c in blobs]


def pick_pivot_and_bob(blobs, fixed_pivot):
    if fixed_pivot is not None:
        return (fixed_pivot, blobs[0]) if blobs else (None, None)
    if len(blobs) < 2:
        return None, None
    a, b = blobs[0], blobs[1]
    return (a, b) if a[1] < b[1] else (b, a)  # the pivot is the higher marker (smaller y)


def angle_from_vertical(pivot, bob):
    dx = bob[0] - pivot[0]
    dy = bob[1] - pivot[1]  # image y points down, so a hanging pendulum has dy > 0
    return math.degrees(math.atan2(dx, dy))



def make_tag_detector(dict_name):
    """Return a function gray_image -> (corners, ids, rejected). Works with old and new OpenCV."""
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        raise SystemExit("This OpenCV has no ArUco module. On the Pi: sudo apt install python3-opencv. "
                         "With pip: pip install opencv-contrib-python")
    dict_id = getattr(aruco, "DICT_" + dict_name.upper(), None)
    if dict_id is None:
        raise SystemExit(f"Unknown tag family {dict_name}. Examples: 4X4_50, 5X5_100, APRILTAG_36h11")
    if hasattr(aruco, "ArucoDetector"):                       # OpenCV 4.7 and newer
        params = aruco.DetectorParameters()
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(dict_id), params)
        return lambda gray: detector.detectMarkers(gray)
    dictionary = aruco.Dictionary_get(dict_id)                # OpenCV 4.6 and older
    params = aruco.DetectorParameters_create()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return lambda gray: aruco.detectMarkers(gray, dictionary, parameters=params)


def tag_centre(c):
    """Centre of a tag = where its diagonals cross (correct even if the tag is seen at an angle)."""
    p0, p1, p2, p3 = c
    d1, d2 = p2 - p0, p3 - p1
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-9:
        return c.mean(axis=0)
    t = ((p1[0] - p0[0]) * d2[1] - (p1[1] - p0[1]) * d2[0]) / denom
    return p0 + t * d1


def tag_down(c):
    """Unit vector from the tag's top edge to its bottom edge (corners: TL, TR, BR, BL)."""
    v = (c[2] + c[3]) / 2 - (c[0] + c[1]) / 2
    return v / (np.linalg.norm(v) + 1e-9)


def wrap(deg):
    """Wrap an angle to the range -180..180 degrees."""
    return (deg + 180.0) % 360.0 - 180.0


def tag_rotation(c):
    """The tag's own rotation: angle of its top-to-bottom direction from vertical, degrees."""
    d = tag_down(c)
    return math.degrees(math.atan2(d[0], d[1]))


def read_tags(corners, ids):
    found = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            found[int(i)] = c.reshape(4, 2).astype(np.float64)
    return found


class RotationFallback:
    """While both tags are visible, learn how a tag's own rotation relates to the arm angle.
    Then, when the other tag is lost, that tag's rotation can stand in. The fallback switches
    itself off if the relation is not steady (a tag that does not really turn with the arm)."""

    def __init__(self, min_samples=25, max_spread_deg=3.0):
        self.diffs = {}
        self.min_samples, self.max_spread = min_samples, max_spread_deg

    def learn(self, tag_id, arm_angle, rotation):
        self.diffs.setdefault(tag_id, deque(maxlen=200)).append(math.radians(arm_angle - rotation))

    def offset(self, tag_id):
        d = self.diffs.get(tag_id)
        if not d or len(d) < self.min_samples:
            return None
        a = np.array(d)
        s, c = np.sin(a).mean(), np.cos(a).mean()             # circular mean, safe near +/-180
        spread = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(max(math.hypot(s, c), 1e-12)))))
        return math.degrees(math.atan2(s, c)) if spread <= self.max_spread else None


# ----------------------------------------------------------------------------- helpers

class RollingStats:
    def __init__(self, seconds):
        self.seconds = seconds
        self.items = deque()

    def add(self, t, value):
        self.items.append((t, value))
        while self.items and t - self.items[0][0] > self.seconds:
            self.items.popleft()

    def mean_std(self):
        if not self.items:
            return float("nan"), float("nan")
        values = np.array([v for _, v in self.items])
        return float(values.mean()), float(values.std())


def make_tuner(lo, hi):
    cv2.namedWindow("tune")
    for i, ch in enumerate("HSV"):
        top = 179 if ch == "H" else 255
        cv2.createTrackbar(f"{ch} low", "tune", int(lo[i]), top, lambda _: None)
        cv2.createTrackbar(f"{ch} high", "tune", int(hi[i]), top, lambda _: None)


def read_tuner():
    lo = np.array([cv2.getTrackbarPos(f"{c} low", "tune") for c in "HSV"], np.uint8)
    hi = np.array([cv2.getTrackbarPos(f"{c} high", "tune") for c in "HSV"], np.uint8)
    return lo, hi


def draw(frame, pivot, bob, zero, text_lines):
    if pivot is not None and bob is not None:
        p = tuple(int(round(v)) for v in pivot)
        b = tuple(int(round(v)) for v in bob)
        length = math.dist(pivot, bob)
        # the "zero" direction, so you can see the reference the angle is measured from
        z = math.radians(zero)
        ref = (int(p[0] + length * math.sin(z)), int(p[1] + length * math.cos(z)))
        cv2.line(frame, p, ref, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.line(frame, p, b, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, p, 6, (255, 0, 0), -1)
        cv2.circle(frame, b, 6, (0, 0, 255), -1)
    for i, line in enumerate(text_lines):
        y = 30 + 30 * i
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


# ----------------------------------------------------------------------------- main loop

def main():
    args = parse_args()
    lo, hi = triple(args.hsv_low), triple(args.hsv_high)
    fixed_pivot = tuple(float(v) for v in args.pivot.split(",")) if args.pivot else None
    display = not args.no_display

    if args.video:
        source = VideoSource(args.video)
    else:
        w, h = (int(v) for v in args.size.lower().split("x"))
        source = CameraSource((w, h), args.fps, args.exposure_us)

    detect_tags = make_tag_detector(args.aruco_dict) if args.aruco else None
    if args.aruco:
        args.tune = False

    if args.tune and display:
        make_tuner(lo, hi)

    writer, csv_file = None, None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["t_s", "raw_deg", "angle_deg", "mean_deg", "std_deg", "found", "source"])

    zero = args.zero
    zero_samples = [] if args.auto_zero else None
    fallback = RotationFallback()
    rotation_ids = [args.arm_id] + ([args.pivot_id] if args.pivot_rotates else [])
    last_good, jump_run, jumps, fallbacks = None, 0, 0, 0
    raw_stats = RollingStats(args.window)  # averaging is done on the raw angle; zero is subtracted after
    t0, last_tick, fps_est, frames, missed = None, time.time(), 0.0, 0, 0

    try:
        while True:
            frame, t = source.read()
            if frame is None:
                break
            if t0 is None:
                t0 = t
            t -= t0
            frames += 1

            if args.tune and display:
                lo, hi = read_tuner()

            corners = ids = rejected = mask = None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            measured, how, pivot, bob = float("nan"), "none", None, None
            if detect_tags:
                corners, ids, rejected = detect_tags(gray)
                found = read_tags(corners, ids)
                if args.arm_id in found and (args.pivot_id in found or fixed_pivot is not None):
                    pivot = tag_centre(found[args.pivot_id]) if args.pivot_id in found else np.array(fixed_pivot)
                    bob = tag_centre(found[args.arm_id])
                    measured = angle_from_vertical(pivot, bob)
                    how = "tags" if args.pivot_id in found else "fixed pivot"
                    for tid in rotation_ids:
                        if tid in found:
                            fallback.learn(tid, measured, tag_rotation(found[tid]))
                else:
                    for tid in rotation_ids:
                        offset = fallback.offset(tid) if tid in found else None
                        if offset is not None:
                            c = found[tid]
                            measured = wrap(tag_rotation(c) + offset)
                            pivot = tag_centre(c)
                            r = math.radians(measured)
                            bob = pivot + 4 * np.linalg.norm(c[1] - c[0]) * np.array([math.sin(r), math.cos(r)])
                            how = f"tag {tid} rotation"
                            fallbacks += 1
                            break
            else:
                mask = colour_mask(frame, lo, hi)
                pivot, bob = pick_pivot_and_bob(find_blobs(mask, args.min_area), fixed_pivot)
                if pivot is not None:
                    measured, how = angle_from_vertical(pivot, bob), "colour"

            # Throw away readings that jump faster than the pendulum can move. After 5 rejections
            # in a row the new value is accepted, so the tracker cannot get stuck.
            raw = float("nan")
            if not math.isnan(measured):
                if last_good is not None and jump_run < 5 and \
                        abs(wrap(measured - last_good[1])) > args.max_rate * (t - last_good[0]) + 3.0:
                    jump_run += 1
                    jumps += 1
                    how += ", rejected jump"
                else:
                    raw, last_good, jump_run = measured, (t, measured), 0

            if not math.isnan(raw):
                raw_stats.add(t, raw)
                if zero_samples is not None:
                    if t <= args.auto_zero:
                        zero_samples.append(raw)
                    else:
                        zero = float(np.mean(zero_samples)) if zero_samples else 0.0
                        print(f"Auto zero set to {zero:.2f} deg from {len(zero_samples)} frames")
                        zero_samples = None
            else:
                missed += 1
                pivot = bob = None

            mean_raw, std = raw_stats.mean_std()
            angle, mean = raw - zero, mean_raw - zero

            if writer:
                writer.writerow([f"{t:.4f}", f"{measured:.3f}", f"{angle:.3f}", f"{mean:.3f}", f"{std:.3f}",
                                 int(not math.isnan(raw)), how])

            now = time.time()
            fps_est = 0.9 * fps_est + 0.1 / max(now - last_tick, 1e-6)
            last_tick = now

            if display:
                lines = [f"angle {angle:+6.2f} deg ({how})" if not math.isnan(raw) else f"no angle this frame ({how})",
                         f"mean {mean:+6.2f} +/- {std:.2f} deg ({args.window:g} s)",
                         f"zero {zero:+.2f} deg   {fps_est:4.1f} fps"]
                if args.speed_k and not math.isnan(mean):
                    v = args.speed_k * math.sqrt(max(math.tan(math.radians(abs(mean))), 0.0))
                    lines.append(f"speed ~ {v:.2f} m/s")
                if args.debug:
                    clipped = float((gray >= 250).mean())
                    lines.append(f"brightness {gray.mean():.0f}/255, clipped {clipped:.0%}")
                    if detect_tags:
                        seen = [] if ids is None else sorted(int(i) for i in ids.flatten())
                        lines.append(f"tags seen {seen}, failed candidates {len(rejected)}")
                view = frame.copy()
                if args.debug and rejected is not None and len(rejected):
                    cv2.aruco.drawDetectedMarkers(view, rejected, borderColor=(0, 0, 255))
                if ids is not None and len(ids):
                    cv2.aruco.drawDetectedMarkers(view, corners, ids)
                draw(view, pivot, bob, zero, lines)
                cv2.imshow("pendulum", view)
                if args.tune:
                    cv2.imshow("tune", mask)
                elif args.debug and mask is not None:
                    cv2.imshow("mask", mask)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("z") and not math.isnan(mean_raw):
                    zero = mean_raw
                    print(f"Zero set to {zero:.2f} deg (pass --zero {zero:.2f} next time)")
                if key == ord("e") and hasattr(source, "relock"):
                    print("Measuring exposure again...")
                    source.relock()
                if key == ord("s"):
                    name = f"snapshot_{int(time.time())}.png"
                    cv2.imwrite(name, view)
                    print(f"Saved {name}")
    finally:
        source.close()
        if csv_file:
            csv_file.close()
        if display:
            cv2.destroyAllWindows()
        if args.tune:
            print(f"--hsv-low {lo[0]},{lo[1]},{lo[2]} --hsv-high {hi[0]},{hi[1]},{hi[2]}")
        print(f"{frames} frames, no angle in {missed}"
              + (f", single-tag rotation used in {fallbacks}" if detect_tags else "")
              + f", jumps rejected {jumps}")
        if frames == 0:
            print("No frames were read. If this was a video file, check that it plays and is not empty.")


if __name__ == "__main__":
    main()