#!/usr/bin/env python3
"""
Make a synthetic pendulum video with known angles, so you can test pendulum_tracker.py before the
real pendulum exists (and on a laptop, without the camera).

The video has: the wind off for a few seconds, a swing (after a push) that dies away, then steady
"wind" steps at increasing angles with turbulent wobble on top. The camera is slightly tilted on
purpose, to check that zeroing works. The true angle of every frame goes to a CSV file.

    python make_test_video.py
    python pendulum_tracker.py --video synthetic_pendulum.mp4 --auto-zero 2 --csv tracked.csv
    python make_test_video.py --compare tracked.csv     # error of the tracker against the truth
"""

import argparse
import csv
import math

import cv2
import numpy as np

SHIFT = 4          # draw with 1/16-pixel precision so positions are not rounded to whole pixels
SCALE = 1 << SHIFT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="synthetic_pendulum.mp4")
    p.add_argument("--truth", default="synthetic_truth.csv")
    p.add_argument("--size", default="1280x720")
    p.add_argument("--fps", type=float, default=50)
    p.add_argument("--length", type=float, help="pivot to bob distance in pixels (default: 58%% of height)")
    p.add_argument("--roll", type=float, default=2.0, help="camera tilt in degrees")
    p.add_argument("--noise", type=float, default=4.0, help="sensor noise (grey levels)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--compare", metavar="TRACKED_CSV", help="compare a tracker CSV with the truth CSV")
    return p.parse_args()


def true_angle_schedule(fps, rng):
    """True angle (degrees) for every frame, plus a label for each part of the test."""
    schedule = [("rest", 3.0, 0.0),                                   # wind off, hanging still
                ("swing", 5.0, 0.0)]                                  # given a push, wind off
    schedule += [(f"wind {a}", 5.0, a) for a in (10, 20, 35, 50, 65)]  # steady wind steps

    targets, labels, swing = [], [], []
    for label, seconds, mean in schedule:
        n = int(seconds * fps)
        t = np.arange(n) / fps
        targets.extend([mean] * n)
        labels.extend([label] * n)
        swing.extend(20.0 * np.exp(-t / 1.5) * np.sin(2 * math.pi * t / 1.2) if label == "swing"
                     else np.zeros(n))
    targets = np.array(targets, float)

    # A real pendulum does not jump between angles: let it follow each new wind speed with a lag.
    mean = np.empty_like(targets)
    alpha = 1.0 / (0.4 * fps)
    mean[0] = targets[0]
    for i in range(1, len(targets)):
        mean[i] = mean[i - 1] + alpha * (targets[i] - mean[i - 1])

    # Turbulence: random noise smoothed to a few Hz, larger at higher angles; none with the wind off.
    n = len(targets)
    wobble = np.convolve(rng.normal(0, 1, n + 50), np.ones(25) / 25, mode="same")[25:25 + n]
    wobble *= 1.0 / wobble.std()
    wind_on = np.array([lab.startswith("wind") for lab in labels])
    wobble *= np.where(wind_on, 0.3 + 0.04 * targets, 0.0)

    return mean + np.array(swing) + wobble, labels


def background(w, h, rng):
    """Grey wall with some texture, a darker band (tunnel wall) and a few distractors."""
    yy = np.linspace(0, 1, h)[:, None]
    img = (120 + 40 * yy + rng.normal(0, 6, (h, w))).clip(0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img, (0, h - 60), (w, h), (70, 70, 80), -1)
    cv2.rectangle(img, (60, 400), (160, 470), (40, 90, 200), -1)     # orange box: not the marker colour
    cv2.circle(img, (1150, 200), 25, (200, 160, 60), -1)             # blue-ish disc
    return img


def render(bg, pivot, length, angle_deg, roll_deg):
    img = bg.copy()
    a = math.radians(angle_deg + roll_deg)
    bob = (pivot[0] + length * math.sin(a), pivot[1] + length * math.cos(a))
    P = (int(pivot[0] * SCALE), int(pivot[1] * SCALE))
    B = (int(bob[0] * SCALE), int(bob[1] * SCALE))
    cv2.line(img, P, B, (40, 40, 40), 4, cv2.LINE_AA, SHIFT)                      # rod
    cv2.circle(img, P, 14 * SCALE, (90, 90, 90), -1, cv2.LINE_AA, SHIFT)          # pivot block
    cv2.circle(img, P, 9 * SCALE, (60, 220, 60), -1, cv2.LINE_AA, SHIFT)          # green pivot marker
    cv2.circle(img, B, 26 * SCALE, (235, 235, 235), -1, cv2.LINE_AA, SHIFT)       # white ball
    cv2.circle(img, B, 12 * SCALE, (60, 220, 60), -1, cv2.LINE_AA, SHIFT)         # green bob marker
    return img


def open_writer(path, fps, size):
    """Open a video writer, trying codecs that are usually available. Never fail silently."""
    stem = path.rsplit(".", 1)[0]
    for out_path, codec in ((path, "mp4v"), (path, "avc1"), (stem + ".avi", "MJPG"), (stem + ".avi", "XVID")):
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            print(f"Writing {out_path} with codec {codec}")
            return writer, out_path
    raise SystemExit("OpenCV could not open any video writer. Check the 'Video I/O' section of\n"
                     "  python3 -c \"import cv2; print(cv2.getBuildInformation())\"")


def make_video(args):
    w, h = (int(v) for v in args.size.lower().split("x"))
    length = args.length or 0.58 * h
    rng = np.random.default_rng(args.seed)
    cv2.setRNGSeed(args.seed)
    angles, labels = true_angle_schedule(args.fps, rng)
    bg = background(w, h, rng)
    pivot = (w / 2, 0.14 * h)

    out, out_path = open_writer(args.out, args.fps, (w, h))
    sub = 4  # motion blur: average 4 renders spread over half a frame period
    acc = np.empty((h, w, 3), np.float32)
    noise = np.empty((h, w, 3), np.float32)
    n = len(angles)
    with open(args.truth, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "true_deg", "part", "roll_deg"])
        for i, angle in enumerate(angles):
            nxt = angles[min(i + 1, n - 1)]
            acc[:] = 0
            for k in range(sub):
                acc += render(bg, pivot, length, angle + (nxt - angle) * 0.5 * k / sub, args.roll)
            cv2.randn(noise, (0, 0, 0), (args.noise,) * 3)
            out.write(cv2.convertScaleAbs(cv2.scaleAdd(acc, 1.0 / sub, noise)))
            # the true angle is the mean over the exposure
            true = angle + (nxt - angle) * 0.5 * (sub - 1) / (2 * sub)
            writer.writerow([f"{i / args.fps:.4f}", f"{true:.4f}", labels[i], args.roll])
            if i % 50 == 0 or i == n - 1:
                f.flush()
                print(f"\r  frame {i + 1}/{n}", end="", flush=True)
    out.release()  # the video file is only playable after this line
    print()

    # Read the file back, so an empty or broken video is reported here and not later.
    check = cv2.VideoCapture(out_path)
    readable = 0
    while check.read()[0]:
        readable += 1
    check.release()
    if readable == 0:
        raise SystemExit(f"{out_path} was written but OpenCV cannot read it back. Try --out test.avi")
    print(f"Done: {out_path} ({readable} frames, {n / args.fps:.0f} s) and {args.truth}")
    if out_path != args.out:
        print(f"Note: use --video {out_path} with the tracker.")


def compare(args):
    import pandas as pd
    truth = pd.read_csv(args.truth)
    tracked = pd.read_csv(args.compare)
    n = min(len(truth), len(tracked))
    df = truth.iloc[:n].copy()
    df["tracked"] = tracked["angle_deg"].iloc[:n].values
    df["mean"] = tracked["mean_deg"].iloc[:n].values
    df["err"] = df["tracked"] - df["true_deg"]
    print(f"Frames compared: {n}, markers missed: {(tracked['found'].iloc[:n] == 0).sum()}")
    print("Per-frame error by part of the test (degrees); 'rest' includes the frames used for zeroing:")
    print(df.groupby("part", sort=False)["err"].agg(["mean", "std", lambda e: e.abs().max()])
          .rename(columns={"<lambda_0>": "max_abs"}).round(3))
    print("\nWind steps: true mean over the last second vs the tracker's 1-s mean at the end:")
    for part, g in df[df["part"].str.startswith("wind")].groupby("part", sort=False):
        last = g[g["t_s"] > g["t_s"].iloc[-1] - 1.0]
        print(f"  {part:8s} true {last['true_deg'].mean():6.2f}   tracked {g['mean'].iloc[-1]:6.2f}")

if __name__ == "__main__":
    a = parse_args()
    compare(a) if a.compare else make_video(a)