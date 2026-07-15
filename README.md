# CPU vs NPU - TFLite inference benchmark

The same two-stage AI pipeline (person detection + per-person PPE detection,
INT8 TFLite models) executed single-threaded in Python on two boards:

| Script | Device | Inference |
|--------|--------|-----------|
| `run_rpi.py` | Raspberry Pi 5 | CPU (XNNPack) |
| `run_rubik.py` | Rubik Pi 3 (Qualcomm QCS6490) | NPU (Hexagon HTP via QNN delegate) |
| `run_rubik_cpu.py` | Rubik Pi 3 (Qualcomm QCS6490) | CPU only (Hexagon disabled) |

The third script is the control run: the same board as `run_rubik.py` with the
NPU taken out of the equation, so the CPU-vs-NPU gap cannot be explained away
by the Rubik simply having a stronger CPU than the Raspberry Pi.

Compared metrics: per-frame stage timings, FPS, SoC temperature (sampled every
frame from `/sys/class/thermal`). Power draw is measured externally with a USB
power meter (e.g. ChargerLAB Power-Z) - the meter cable is the only power
source, so the recording covers exactly one benchmark run.

## Repository layout

```
common.py        shared helpers (TFLite I/O, postprocessing, temperature, env dump)
run_rpi.py       benchmark entry point for the Raspberry Pi
run_rubik.py     benchmark entry point for the Rubik Pi
models/          INT8 .tflite models (person/foot detection + PPE detection)
input/           source .mp4 videos
results/         benchmark output (plain text: CSV + JSON), overwritten per run
output/          annotated videos (only when SAVE_VIDEO = True; gitignored)
```

## Setup (on the device)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On the Rubik Pi the Hexagon delegate (`libQnnTFLiteDelegate.so` from the
Qualcomm QAIRT/QNN SDK) must be resolvable via `LD_LIBRARY_PATH`. If it fails
to load, the script prints a warning and falls back to CPU.

## Run

No CLI arguments - every parameter is a global variable at the top of the
script, so runs are repeatable by design:

```bash
python run_rpi.py        # on the Raspberry Pi
python run_rubik.py      # on the Rubik Pi (NPU)
python run_rubik_cpu.py  # on the Rubik Pi (CPU only)
```

All three write to the same `results/` directory - archive or pull the results
between runs on the same device.

Set `SAVE_VIDEO = True` to also write annotated videos to `output/` (off by
default: mp4 encoding loads the CPU and skews the benchmark).

## Output (plain text, overwritten on every run)

- `results/inference_metrics.csv` - one row per frame:

  | Column | Description |
  |--------|-------------|
  | `video`, `frame_id` | source video and frame index |
  | `num_persons` | persons detected |
  | `frame_ms` | wall time of the whole frame (decode + inference + postprocess) |
  | `temp_c` | SoC temperature |

  This experiment deliberately reports only whole-frame numbers (FPS, power,
  temperature) - the per-stage breakdown of a frame is the subject of the
  single-vs-pipeline experiment.

- `results/summary.json` - total frames, duration, average FPS
- `results/env.json` - Python and package versions, OS release, kernel,
  device model, CPU governors/frequencies, SHA-256 of the models, run config
