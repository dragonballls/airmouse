"""26-minute Windows runtime soak for the packaged Airmouse dependency graph.

This deliberately avoids synthetic mouse clicks and keyboard events. It stresses
the MediaPipe VIDEO tracker, RGB-buffer reuse, gesture routing, keyboard state,
and repeated processing while watching for crashes or runaway memory.
"""
from __future__ import annotations

import argparse
import gc
import os
import time

import numpy as np
import psutil

from core.actuator import MouseActuator
from core.display import build_trackpad_zone, build_virtual_desktop
from core.gestures import GestureOrchestrator
from core.tracker import HandTracker, HandsResult
from core.virtual_keyboard import VirtualKeyboard


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=26.0)
    args = parser.parse_args()

    os.environ["AIRMOUSE_ENABLE_GESTURE_AI"] = "0"
    process = psutil.Process()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[95:265, 205:435] = 8

    desktop = build_virtual_desktop()
    trackpad = build_trackpad_zone()
    actuator = MouseActuator(
        desktop.total_width,
        desktop.total_height,
        origin_x=desktop.origin_x,
        origin_y=desktop.origin_y,
    )
    orchestrator = GestureOrchestrator(actuator, desktop, trackpad)
    keyboard = VirtualKeyboard(640, 280, key_width=54, key_height=42, key_margin=5, compact=True)
    tracker = HandTracker()

    baseline_rss = process.memory_info().rss
    peak_rss = baseline_rss
    frames = 0
    started = time.perf_counter()
    deadline = started + max(25.0 * 60.0, args.minutes * 60.0)
    last_report = started

    try:
        while True:
            now = time.perf_counter()
            if now >= deadline:
                break

            hands = tracker.process(frame)
            assert isinstance(hands, HandsResult)
            orchestrator.process(hands)
            keyboard.update_hover(-1, -1)
            keyboard.update_gesture(None)

            frames += 1
            rss = process.memory_info().rss
            peak_rss = max(peak_rss, rss)

            if now - last_report >= 30.0:
                elapsed = now - started
                fps = frames / max(elapsed, 0.001)
                delta_mb = (rss - baseline_rss) / (1024 * 1024)
                print(
                    f"[SOAK] {elapsed/60:.1f} min | frames={frames} | "
                    f"tracker_fps={fps:.1f} | rss_delta={delta_mb:+.1f} MB",
                    flush=True,
                )
                if delta_mb > 300.0:
                    raise MemoryError(f"RSS grew by {delta_mb:.1f} MB")
                last_report = now

            if frames % 300 == 0:
                gc.collect()
    finally:
        tracker.close()
        keyboard.close()
        orchestrator.reset()

    elapsed = time.perf_counter() - started
    final_rss = process.memory_info().rss
    print(
        f"[SOAK PASS] elapsed={elapsed/60:.2f} min | frames={frames} | "
        f"peak_rss_delta={(peak_rss-baseline_rss)/(1024*1024):+.1f} MB | "
        f"final_rss_delta={(final_rss-baseline_rss)/(1024*1024):+.1f} MB",
        flush=True,
    )
    assert elapsed >= 25.0 * 60.0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
