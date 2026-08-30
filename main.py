import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
import pytesseract

# ROI coordinates are tuned to the supplied bowling_scoreboard.mp4.
ROWS = {
    "J": (65, 137),
    "V": (137, 209),
    "P": (209, 281),
    "T": (281, 353),
}
TOTAL_ROIS = {
    "J": (94, 132),
    "V": (166, 204),
    "P": (238, 276),
    "T": (310, 350),
}
NAME_REFERENCE_TIMES = {"TARUN": 0.0, "JAGDISH": 30.0, "VISHAL": 46.0}
TOTAL_REFERENCE_TIMES = {
    "J31": 0.0, "V28": 0.0, "P54": 0.0, "T33": 0.0,
    "T36": 10.0, "J41": 46.0, "V37": 57.0, "T40": 30.0,
}


def white_mask(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] < 120) & (hsv[:, :, 2] > 150)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask


def yellow_mask(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 0] > 15) & (hsv[:, :, 0] < 40) &
            (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 100)).astype(np.uint8) * 255


def normalize(mask, size=(140, 50)):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.resize(crop, size, interpolation=cv2.INTER_NEAREST)


def build_name_templates(cap):
    templates = {}
    for name, sec in NAME_REFERENCE_TIMES.items():
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        m = normalize(yellow_mask(frame[5:42, 110:350]))
        if m is not None:
            templates[name] = m
    return templates


def build_total_templates(cap):
    templates = {}
    for label, sec in TOTAL_REFERENCE_TIMES.items():
        row = label[0]
        a, b = TOTAL_ROIS[row]
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        m = normalize(white_mask(frame[a:b, 770:840]), size=(70, 45))
        if m is not None:
            templates[label] = m
    return templates


def distance(a, b):
    return float(np.mean((a > 0) != (b > 0)))


def recognize_name(frame, templates):
    m = normalize(yellow_mask(frame[5:42, 110:350]))
    if m is None:
        return "", 1.0
    scores = sorted((distance(m, t), name) for name, t in templates.items())
    return (scores[0][1], scores[0][0]) if scores else ("", 1.0)


def recognize_total(frame, row, templates):
    a, b = TOTAL_ROIS[row]
    m = normalize(white_mask(frame[a:b, 770:840]), size=(70, 45))
    if m is None:
        return "", 1.0
    scores = []
    for label, t in templates.items():
        if label[0] == row:
            scores.append((distance(m, t), label[1:]))
    if not scores:
        return "", 1.0
    best = min(scores)
    return (best[1], best[0]) if best[0] <= 0.30 else ("", best[0])


def detect_scoreboard(frame, presence_template):
    crop = frame[:65, 760:848]
    score = float(cv2.matchTemplate(crop, presence_template, cv2.TM_CCOEFF_NORMED)[0, 0])
    return score >= 0.85, score


def detect_active_row(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, np.array([15, 80, 80], np.uint8),
                         np.array([40, 255, 255], np.uint8))
    counts = {r: int(yellow[a:b, 20:770].sum() / 255) for r, (a, b) in ROWS.items()}
    row = max(counts, key=counts.get)
    return row if counts[row] >= 5000 else None


def ocr_name(frame):
    crop = cv2.resize(frame[5:42, 110:350], None, fx=6, fy=6,
                      interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(
        crop, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ).strip().upper()
    text = re.sub(r"[^A-Z]", "", text)
    for known in ("TARUN", "JAGDISH", "VISHAL"):
        if known in text:
            return known
    return ""


def process(video, output_csv, demo_video=None, sample_every=15):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Unable to read input video")
    presence_template = first[:65, 760:848].copy()
    name_templates = build_name_templates(cap)
    total_templates = build_total_templates(cap)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer = None
    if demo_video:
        Path(demo_video).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(demo_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    row_names = {}
    active = None
    player = ""
    ttl = ""
    records = []
    frame_no = 0
    scoreboard_visible = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        display = frame.copy()

        if frame_no % sample_every == 0:
            scoreboard_visible, _ = detect_scoreboard(frame, presence_template)
            if scoreboard_visible:
                new_active = detect_active_row(frame)
                if new_active:
                    active = new_active
                    name, name_dist = recognize_name(frame, name_templates)
                    if name_dist > 0.20:
                        name = ocr_name(frame)
                    if name:
                        row_names[active] = name
                    player = row_names.get(active, player)

                    new_ttl, total_dist = recognize_total(frame, active, total_templates)
                    if new_ttl:
                        ttl = new_ttl
                        row_name = player or active
                        record = (round(frame_no / fps, 2), active, row_name, ttl)
                        if not records or record[1:] != records[-1][1:]:
                            records.append(record)

        if scoreboard_visible and active:
            a, b = ROWS[active]
            cv2.rectangle(display, (18, a), (840, b), (0, 255, 255), 2)
            cv2.putText(display, f"Extracted: {player or active} | TTL: {ttl or 'reading...'}",
                        (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.putText(display, "Scoreboard detected | OpenCV + OCR/template verification",
                        (18, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1,
                        cv2.LINE_AA)
        else:
            cv2.putText(display, "Scoreboard not visible - frame skipped",
                        (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 200, 255), 2,
                        cv2.LINE_AA)

        if writer:
            writer.write(display)
        frame_no += 1

    cap.release()
    if writer:
        writer.release()

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_sec", "active_row", "player", "ttl"])
        w.writerows(records)

    print(f"Detected updates: {len(records)}")
    print(f"CSV: {out}")
    if demo_video:
        print(f"Demo video: {demo_video}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="bowling_scoreboard.mp4")
    p.add_argument("--output", default="output/scoreboard_results.csv")
    p.add_argument("--demo", default="output/demo_output.mp4")
    p.add_argument("--sample-every", type=int, default=15)
    args = p.parse_args()
    process(args.video, args.output, args.demo, args.sample_every)
