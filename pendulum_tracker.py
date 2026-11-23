#!/usr/bin/env python3
"""
Pendulum angle tracker for the Raspberry Pi HQ Camera (Picamera2 + OpenCV).

Put two markers of the same bright colour on the pendulum: one on the pivot and one on the
bob (or near the end of the arm). The script finds both markers in every frame and reports the
angle of the line between them, measured from the vertical:

    angle = 0    pendulum hanging straight down (after zeroing with the wind off)
    angle > 0    bob to the right of the pivot in the image

Usage
    python pendulum_tracker.py                          live camera on the CM5
    python pendulum_tracker.py --tune                   sliders to choose the marker colour range
    python pendulum_tracker.py --video clip.mp4         recorded or synthetic video (works on a laptop)
    python pendulum_tracker.py --pivot 640,120          fixed pivot pixel, track only the bob marker
    python pendulum_tracker.py --csv run1.csv           log every frame to a CSV file
    python pendulum_tracker.py --auto-zero 2            use the mean of the first 2 s as zero
    python pendulum_tracker.py --speed-k 9.0            also show v = k * sqrt(tan(angle))

Keys in the video window
    z   set zero to the current averaged angle (do this with the wind OFF)
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
    p.add_argument("--no-display", action="store_true", help="no windows (for batch runs on videos)")
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
        time.sleep(1.5)  # let auto exposure and auto white balance settle

        # Lock exposure, gain and white balance. If they keep changing, the colours change and the
        # HSV thresholds stop working (and the windows of a wind tunnel change the light a lot).
        md = self.cam.capture_metadata()
        exposure, gain = md["ExposureTime"], md["AnalogueGain"]
        if exposure_us:  # keep the same brightness: shorter exposure, more gain
            gain = float(np.clip(gain * exposure / exposure_us, 1.0, 16.0))
            exposure = exposure_us
        self.cam.set_controls({
            "AeEnable": False, "AwbEnable": False,
            "ExposureTime": int(exposure), "AnalogueGain": gain,
            "ColourGains": md["ColourGains"],
        })
        print(f"Camera locked: exposure {exposure} us, gain {gain:.2f}, colour gains {md['ColourGains']}")

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

    if args.tune and display:
        make_tuner(lo, hi)

    writer, csv_file = None, None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["t_s", "raw_deg", "angle_deg", "mean_deg", "std_deg", "found"])

    zero = args.zero
    zero_samples = [] if args.auto_zero else None
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

            mask = colour_mask(frame, lo, hi)
            pivot, bob = pick_pivot_and_bob(find_blobs(mask, args.min_area), fixed_pivot)

            raw = float("nan")
            if pivot is not None:
                raw = angle_from_vertical(pivot, bob)
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

            mean_raw, std = raw_stats.mean_std()
            angle, mean = raw - zero, mean_raw - zero

            if writer:
                writer.writerow([f"{t:.4f}", f"{raw:.3f}", f"{angle:.3f}", f"{mean:.3f}", f"{std:.3f}",
                                 int(pivot is not None)])

            now = time.time()
            fps_est = 0.9 * fps_est + 0.1 / max(now - last_tick, 1e-6)
            last_tick = now

            if display:
                lines = [f"angle {angle:+6.2f} deg" if pivot is not None else "markers not found",
                         f"mean {mean:+6.2f} +/- {std:.2f} deg ({args.window:g} s)",
                         f"zero {zero:+.2f} deg   {fps_est:4.1f} fps"]
                if args.speed_k and not math.isnan(mean):
                    v = args.speed_k * math.sqrt(max(math.tan(math.radians(abs(mean))), 0.0))
                    lines.append(f"speed ~ {v:.2f} m/s")
                view = frame.copy()
                draw(view, pivot, bob, zero, lines)
                cv2.imshow("pendulum", view)
                if args.tune:
                    cv2.imshow("tune", mask)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("z") and not math.isnan(mean_raw):
                    zero = mean_raw
                    print(f"Zero set to {zero:.2f} deg (pass --zero {zero:.2f} next time)")
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
        print(f"{frames} frames, markers missed in {missed}")
        if frames == 0:
            print("No frames were read. If this was a video file, check that it plays and is not empty.")


if __name__ == "__main__":
    main()