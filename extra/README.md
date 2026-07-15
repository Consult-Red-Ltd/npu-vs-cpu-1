# extra - CPU thread-count sweep

Companion experiment to the main benchmark in this repo's root. The main run
uses LiteRT's default thread count (1 thread - `common.load_interpreter` sets no
`num_threads`); these scripts re-run the **same two-stage pipeline** at a range
of CPU thread counts to see how throughput, temperature and power scale with the
number of threads.

CPU path only (XNNPACK). The NPU is not swept - the Hexagon delegate does not
scale with `num_threads`, so it would be meaningless.

| Script | Device | Threads swept |
|--------|--------|---------------|
| `run_rpi_threads.py`   | Raspberry Pi 5            | 1–4 (4 cores) |
| `run_rubik_threads.py` | Rubik Pi 3 (CPU only)    | 1–8 (8 cores) |

Both reuse the published pipeline code in the parent directory (`common.py`) so
the measured pipeline is identical to the article's - `sweep.py` just loops it
over thread counts.

## Power measurement

Power is **not** measured in software (same as the main experiment): record it
externally with the ChargerLAB Power-Z meter. Start the recording **before**
launching the script and keep it running for the whole sweep. The script prints
a wall-clock `POWER-Z MARKER START/END` line for each thread count and stores
those ISO timestamps in `sweep_summary.csv`; slice the `.db` by each
`[start_iso, end_iso]` window to get mean power / energy-per-frame per thread
count.

## Run

```bash
python run_rpi_threads.py     # on the Raspberry Pi
python run_rubik_threads.py   # on the Rubik Pi (CPU)
```

## Output (`results_threads/<device>/`)

- `threads_<n>.csv` - per-frame rows (same schema as the main benchmark) for n threads
- `sweep_summary.csv` - one row per thread count: `threads, frames, duration_s,
  wall_s, fps, max_temp_c, avg_temp_c, start_iso, end_iso`
- `env.json` - environment + run config (written once)
