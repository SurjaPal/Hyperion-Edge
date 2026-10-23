#!/usr/bin/env python3
"""
Fetch third-party assets that are deliberately NOT bundled in the submission (see
THIRD_PARTY_NOTICES.md for why): the SKAB sensor dataset the live demo replays, the YOLOX-nano
model if missing, and a few sample frames for smoke-testing the vision module.
Safe to run repeatedly; skips anything already present. Requires internet access once.
"""
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKAB_BASE = "https://raw.githubusercontent.com/waico/SKAB/master/data"

# The exact set that downloaded successfully and was used for every result in docs/RESULTS.md.
SKAB_FILES = (
    [f"valve1/{i}.csv" for i in range(1, 16)]
    + [f"valve2/{i}.csv" for i in range(1, 4)]
    + [f"other/{i}.csv" for i in range(1, 15)]
    + ["anomaly-free/anomaly-free.csv"]
)

OTHER_ASSETS = [
    ("models/yolox_nano.onnx",
     "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx"),
    ("data/frames/sample_bus.jpg",
     "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"),
    ("data/frames/sample_zidane.jpg",
     "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg"),
    ("data/frames/sample_dog.jpg",
     "https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/assets/dog.jpg"),
]


def _fetch(rel, url):
    dest = os.path.join(ROOT, rel)
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        print(f"downloading {rel} ...")
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"  could not fetch {rel}: {e}")


def main():
    for rel in SKAB_FILES:
        _fetch(os.path.join("data", "skab", rel), f"{SKAB_BASE}/{rel}")
    for rel, url in OTHER_ASSETS:
        _fetch(rel, url)
    missing_skab = [f for f in SKAB_FILES if not os.path.exists(os.path.join(ROOT, "data", "skab", f))]
    if missing_skab:
        print(f"WARNING: {len(missing_skab)} SKAB file(s) could not be fetched; "
              "the live demo needs at least valve1/2.csv, valve2/2.csv, other/3.csv.")


if __name__ == "__main__":
    main()
