"""
Thread-count sweep for the CPU pipeline - companion experiment to the main
benchmark (article 01). The headline article runs a single configuration per
board; here we re-run the *same* two-stage pipeline at a range of CPU thread
counts to map how throughput, temperature and (externally measured) power scale
with the number of threads.

This deliberately covers only the CPU path (XNNPACK): the Hexagon/NPU run is
not affected by `num_threads`, so sweeping it would be meaningless.

Power is NOT measured in software. As in the main experiment it is logged
externally with the ChargerLAB Power-Z meter. To make the trace sliceable, this
script prints a wall-clock MARKER (and records ISO timestamps in the summary
CSV) around each thread-count segment - align each [start_iso, end_iso] window
with the .db afterwards to get mean power per thread count.

Used by the two thin entry points in this directory:
    run_rpi_threads.py     (RPi 5  - 1..4 threads, 4 cores)
    run_rubik_threads.py   (Rubik Pi 3 CPU - 1..8 threads, 8 cores)
"""

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

# Put the published 01-cpu-vs-npu/ package dir on the path so we reuse the exact
# same pipeline code (process_video, read_temp_c, write_env, ...) - the sweep must
# measure the same pipeline the article does, not a fork of it. The interpreter is
# built HERE though: the article's common.load_interpreter always uses the LiteRT
# default (1 thread), whereas the whole point of this sweep is to VARY num_threads.
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import common  # noqa: E402
from ai_edge_litert import interpreter as litert  # noqa: E402

INPUT_DIR = BASE_DIR / "input"
MODEL_PERSON = BASE_DIR / "models" / "foot_track_net-tflite-w8a8" / "foot_track_net.tflite"
MODEL_PPE = BASE_DIR / "models" / "gear_guard_net-tflite-w8a8" / "gear_guard_net.tflite"
SAVE_VIDEO = False


def cpu_interpreter(model_path, num_threads):
    # CPU (XNNPACK) path via LiteRT, with num_threads varied by the sweep. No
    # Hexagon delegate: the NPU does not scale with num_threads, so sweeping it
    # would be meaningless (see module docstring).
    interp = litert.Interpreter(model_path=str(model_path), num_threads=num_threads)
    interp.allocate_tensors()
    return interp


def run_sweep(device, thread_counts):
    """Run the full 6-video CPU pipeline once per thread count.

    device:        short tag used for the output sub-directory ("rpi", "rubik-cpu").
    thread_counts: iterable of `num_threads` values to sweep, e.g. range(1, 5).
    """
    thread_counts = list(thread_counts)
    results_dir = Path(__file__).resolve().parent / "results_threads" / device
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== CPU thread sweep on '{device}' | counts={thread_counts} ===")
    print("Power is measured externally (Power-Z). Start the recording now and "
          "keep it running for the whole sweep.\n")

    rows = []
    for tc in thread_counts:
        interp_person = cpu_interpreter(MODEL_PERSON, tc)
        interp_ppe = cpu_interpreter(MODEL_PPE, tc)

        if tc == thread_counts[0]:
            common.write_env(
                results_dir / "env.json",
                {"device": device, "delegate": "cpu",
                 "thread_counts": thread_counts, "save_video": SAVE_VIDEO},
                [MODEL_PERSON, MODEL_PPE])

        per_frame_csv = results_dir / f"threads_{tc}.csv"
        start_iso = datetime.now().isoformat(timespec="seconds")
        t0 = time.perf_counter()
        print(f"--- threads={tc} | POWER-Z MARKER START {start_iso} ---")

        total_frames, total_seconds = 0, 0.0
        with open(per_frame_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(common.CSV_HEADER)
            for video in sorted(INPUT_DIR.glob("*.mp4")):
                frames, seconds = common.process_video(
                    video, None, interp_person, interp_ppe, w, SAVE_VIDEO)
                total_frames += frames
                total_seconds += seconds

        wall_s = time.perf_counter() - t0
        end_iso = datetime.now().isoformat(timespec="seconds")
        fps = round(total_frames / total_seconds, 2) if total_seconds else 0.0
        print(f"--- threads={tc} | POWER-Z MARKER END   {end_iso} "
              f"| {total_frames} frames, {fps} FPS ---\n")

        with open(per_frame_csv) as f:
            temps = [float(r["temp_c"]) for r in csv.DictReader(f)]
        rows.append({
            "threads": tc,
            "frames": total_frames,
            "duration_s": round(total_seconds, 2),
            "wall_s": round(wall_s, 2),
            "fps": fps,
            "max_temp_c": round(max(temps), 1) if temps else 0.0,
            "avg_temp_c": round(sum(temps) / len(temps), 1) if temps else 0.0,
            "start_iso": start_iso,
            "end_iso": end_iso,
        })

    summary_path = results_dir / "sweep_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Sweep summary: {summary_path}")
    print("Per-thread-count window [start_iso, end_iso] → slice the Power-Z .db "
          "to get mean power / energy-per-frame at each thread count.")
    return rows
