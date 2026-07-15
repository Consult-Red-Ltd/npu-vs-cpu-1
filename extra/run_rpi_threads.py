"""Raspberry Pi 5 CPU thread sweep: 1..4 threads (BCM2712 = 4× Cortex-A76).

Run on the Pi with the Power-Z meter recording:
    python run_rpi_threads.py
Output: results_threads/rpi/  (per-thread CSVs + sweep_summary.csv + env.json)
"""

import sweep

if __name__ == "__main__":
    sweep.run_sweep("rpi", range(1, 5))  # 1, 2, 3, 4
