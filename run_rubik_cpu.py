import csv
import json
import time
from datetime import datetime
from pathlib import Path

import common

DELEGATE = "cpu"
SAVE_VIDEO = False

COOLDOWN_TOLERANCE_C = 3.0
COOLDOWN_POLL_S = 30
COOLDOWN_TIMEOUT_S = 900

BASE_DIR = Path(__file__).parent.resolve()
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
RESULTS_DIR = BASE_DIR / "results"
BASELINE_FILE = BASE_DIR / "npu_baseline.json"
MODEL_PERSON = BASE_DIR / "models" / "foot_track_net-tflite-w8a8" / "foot_track_net.tflite"
MODEL_PPE = BASE_DIR / "models" / "gear_guard_net-tflite-w8a8" / "gear_guard_net.tflite"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    if SAVE_VIDEO:
        OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Device: Rubik Pi, CPU only (no Hexagon) | "
          f"Save video: {SAVE_VIDEO}")
    interp_person = common.load_interpreter(MODEL_PERSON, DELEGATE)
    interp_ppe = common.load_interpreter(MODEL_PPE, DELEGATE)

    config = {"device": "rubik-cpu", "delegate": DELEGATE,
              "save_video": SAVE_VIDEO}
    common.write_env(RESULTS_DIR / "env.json", config, [MODEL_PERSON, MODEL_PPE])

    npu_start_temp = common.read_baseline_temp(BASELINE_FILE)
    if npu_start_temp is not None:
        print(f"NPU started at {npu_start_temp:.1f} C - cooling CPU run to match "
              f"(+/-{COOLDOWN_TOLERANCE_C:.0f} C) before benchmarking")
        common.wait_until_cooled(npu_start_temp, COOLDOWN_TOLERANCE_C,
                                 COOLDOWN_POLL_S, COOLDOWN_TIMEOUT_S)
    else:
        print("No NPU baseline found - skipping cooldown")
    start_temp = common.read_temp_c()
    print(f"CPU run start temp: {start_temp:.1f} C")

    csv_path = RESULTS_DIR / "inference_metrics.csv"
    total_frames = 0
    total_seconds = 0.0
    started_unix = time.time()
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(common.CSV_HEADER)
        for video in sorted(INPUT_DIR.glob("*.mp4")):
            print(f"Processing: {video.name}")
            frames, seconds = common.process_video(
                video, OUTPUT_DIR / video.name,
                interp_person, interp_ppe, writer, SAVE_VIDEO)
            total_frames += frames
            total_seconds += seconds
    finished_unix = time.time()

    summary = {
        "device": "rubik-cpu",
        "delegate": DELEGATE,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "frames": total_frames,
        "duration_s": round(total_seconds, 2),
        "fps": round(total_frames / total_seconds, 2) if total_seconds else 0.0,
        "start_temp_c": round(start_temp, 1),
        "npu_baseline_temp_c": round(npu_start_temp, 1) if npu_start_temp is not None else None,
        "started_unix": round(started_unix, 3),
        "finished_unix": round(finished_unix, 3),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done: {summary['frames']} frames, {summary['fps']} FPS")
    print(f"Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
