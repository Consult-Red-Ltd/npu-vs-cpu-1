"""Rubik Pi 3 CPU thread sweep: 1..8 threads (QCS6490 = 8× Kryo 670).

CPU only - the Hexagon/NPU delegate is not used here (it does not scale with
`num_threads`). Run on the Rubik with the Power-Z meter recording:
    python run_rubik_threads.py
Output: results_threads/rubik-cpu/  (per-thread CSVs + sweep_summary.csv + env.json)
"""

import sweep

if __name__ == "__main__":
    sweep.run_sweep("rubik-cpu", range(1, 9))  # 1..8
