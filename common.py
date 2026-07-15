import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert import interpreter as litert

PERSON_SCORE_THRESH = 0.76
PERSON_IOU_THRESH = 0.5
PPE_SCORE_THRESH = 0.40
PPE_IOU_THRESH = 0.5
STRIDE = 4
TOP_K = 1000

COLOR_PERSON = (255, 200, 0)
COLOR_HELMET = (0, 165, 255)
COLOR_VEST = (0, 255, 0)

THERMAL_ZONE_PRIORITY = ("cpu0-thermal", "cpuss0-thermal", "cpu-thermal")


def load_interpreter(model_path, delegate):
    # CPU path (delegate != "hexagon"): no explicit delegate object is passed.
    # LiteRT then runs inference on the CPU through its built-in XNNPACK delegate
    # (enabled by default), which accelerates the INT8 (w8a8) ops. XNNPACK is thus
    # the CPU baseline for both the RPi 5 and the Rubik CPU control run - there is
    # no Python flag to add it explicitly; it is the default.
    delegates = []
    if delegate == "hexagon":
        # https://docs.qualcomm.com/nav/home/options.html?product=1601111740010412
        options = {"backend_type": "htp", "htp_performance_mode": "2"}
        try:
            delegates.append(litert.load_delegate("libQnnTFLiteDelegate.so", options=options))
        except Exception as e:
            print(f"WARN: Hexagon delegate unavailable ({e}), falling back to CPU.")
    interpreter = litert.Interpreter(
        model_path=str(model_path),
        experimental_delegates=delegates or None,
    )
    interpreter.allocate_tensors()
    return interpreter


def get_output(interpreter, name, dequantize=True):
    detail = next(d for d in interpreter.get_output_details() if d["name"] == name)
    tensor = interpreter.get_tensor(detail["index"])
    scale, zero_point = detail["quantization"]
    if dequantize and scale > 0.0:
        tensor = (tensor.astype(np.float32) - zero_point) * scale
    return tensor


def iou(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-5)


def nms(boxes, iou_thresh):
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    for box in boxes:
        if all(iou(box, kept) <= iou_thresh for kept in keep):
            keep.append(box)
    return keep


def detect_persons(interpreter):
    heatmap = get_output(interpreter, "heatmap")
    tlrb = get_output(interpreter, "bbox")
    scores = heatmap[0, :, :, 1]
    ys, xs = np.where(scores > PERSON_SCORE_THRESH)
    if len(ys) > TOP_K:
        order = np.argsort(-scores[ys, xs])[:TOP_K]
        ys, xs = ys[order], xs[order]
    boxes = []
    for cy, cx in zip(ys, xs):
        left, top, right, bottom = tlrb[0, cy, cx, 4:8]
        boxes.append((
            (cx - left) * STRIDE,
            (cy - top) * STRIDE,
            (cx + right) * STRIDE,
            (cy + bottom) * STRIDE,
            float(scores[cy, cx]),
        ))
    return nms(boxes, PERSON_IOU_THRESH)


def detect_ppe(interpreter):
    boxes = get_output(interpreter, "boxes")[0]
    scores = get_output(interpreter, "scores")[0]
    classes = get_output(interpreter, "class_idx", dequantize=False)[0]
    valid = scores > PPE_SCORE_THRESH
    boxes, scores, classes = boxes[valid], scores[valid], classes[valid]
    if len(boxes) == 0:
        return []
    rects = [
        [int(x1), int(y1), int(max(0, x2 - x1)), int(max(0, y2 - y1))]
        for x1, y1, x2, y2 in boxes
    ]
    keep = cv2.dnn.NMSBoxes(rects, scores.astype(float).tolist(), PPE_SCORE_THRESH, PPE_IOU_THRESH)
    if len(keep) == 0:
        return []
    return [
        (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], int(classes[i]), float(scores[i]))
        for i in np.asarray(keep).flatten()
    ]


def resize_and_pad(image, target_h, target_w):
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return canvas, scale, pad_w, pad_h


def draw_box(frame, x1, y1, x2, y2, color, label):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    cv2.putText(frame, label, (x1 + 4, max(14, y1 - 6)), cv2.FONT_HERSHEY_DUPLEX,
                0.45, color, 1, cv2.LINE_AA)


def read_temp_c():
    zones = {}
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            zones[(zone / "type").read_text().strip()] = zone / "temp"
        except OSError:
            pass
    for name in THERMAL_ZONE_PRIORITY:
        if name in zones:
            try:
                return int(zones[name].read_text().strip()) / 1000.0
            except (OSError, ValueError):
                pass
    return 0.0


def _read_file(path):
    try:
        return Path(path).read_text(errors="ignore").strip().strip("\x00")
    except OSError:
        return ""


def _cpu_freq_info():
    info = {}
    for cpu_dir in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
        governor = _read_file(cpu_dir / "cpufreq" / "scaling_governor")
        max_freq = _read_file(cpu_dir / "cpufreq" / "cpuinfo_max_freq")
        if governor or max_freq:
            info[cpu_dir.name] = {"governor": governor, "max_freq_khz": max_freq}
    return info


def write_env(path, config, model_paths):
    env = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "libc": "-".join(platform.libc_ver()),
        "device_model": _read_file("/proc/device-tree/model"),
        "os_release": _read_file("/etc/os-release"),
        "cpu": _cpu_freq_info(),
        "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()},
        "models": {
            Path(m).name: hashlib.sha256(Path(m).read_bytes()).hexdigest()
            for m in model_paths
        },
        "config": config,
    }
    Path(path).write_text(json.dumps(env, indent=2), encoding="utf-8")


def process_video(video_path, output_path, interp_person, interp_ppe, csv_writer, save_video):
    person_input = interp_person.get_input_details()[0]
    ppe_input = interp_ppe.get_input_details()[0]
    h1, w1 = person_input["shape"][1], person_input["shape"][2]
    h2, w2 = ppe_input["shape"][1], ppe_input["shape"][2]

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save_video:
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (orig_w, orig_h))

    frame_id = 0
    t_video_start = time.perf_counter()
    while True:
        t_frame_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(frame_rgb, (w1, h1))
        interp_person.set_tensor(person_input["index"],
                                 np.expand_dims(resized, 0).astype(np.uint8))
        interp_person.invoke()
        persons = detect_persons(interp_person)

        scale_x = orig_w / w1
        scale_y = orig_h / h1

        for px1, py1, px2, py2, person_score in persons:
            x1 = max(0, int(px1 * scale_x))
            y1 = max(0, int(py1 * scale_y))
            x2 = min(orig_w, int(px2 * scale_x))
            y2 = min(orig_h, int(py2 * scale_y))

            if save_video:
                draw_box(frame, x1, y1, x2, y2, COLOR_PERSON, f"Person {person_score:.0%}")

            crop = frame_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            padded, crop_scale, pad_w, pad_h = resize_and_pad(crop, h2, w2)
            interp_ppe.set_tensor(ppe_input["index"],
                                  np.expand_dims(padded, 0).astype(np.uint8))
            interp_ppe.invoke()
            detections = detect_ppe(interp_ppe)

            if save_video:
                for bx1, by1, bx2, by2, class_id, _score in detections:
                    gx1 = int(x1 + (bx1 - pad_w) / crop_scale)
                    gy1 = int(y1 + (by1 - pad_h) / crop_scale)
                    gx2 = int(x1 + (bx2 - pad_w) / crop_scale)
                    gy2 = int(y1 + (by2 - pad_h) / crop_scale)
                    color = COLOR_HELMET if class_id == 0 else COLOR_VEST
                    name = "Helmet" if class_id == 0 else "Vest"
                    draw_box(frame, max(0, gx1), max(0, gy1),
                             min(orig_w, gx2), min(orig_h, gy2), color, name)

        # frame_ms = the pipeline cost of one frame: decode + colour convert +
        # resize + both cascade inferences + NMS + post-processing. The frame
        # stopwatch stops HERE, before the two instrumentation calls below
        # (per-frame temperature read + CSV row write), so frame_ms is the clean
        # per-frame pipeline time and is NOT polluted by the benchmark's own
        # logging.
        frame_ms = (time.perf_counter() - t_frame_start) * 1000.0
        csv_writer.writerow([
            Path(video_path).name, frame_id, len(persons),
            f"{frame_ms:.3f}", f"{read_temp_c():.1f}", f"{time.time():.3f}",
        ])

        if writer is not None:
            writer.write(frame)
        frame_id += 1

    cap.release()
    if writer is not None:
        writer.release()

    # elapsed_s = wall-clock for the whole video loop. It is slightly LARGER
    # than the sum of frame_ms because it also contains the per-frame
    # instrumentation (read_temp_c() globs /sys/class/thermal, plus the CSV
    # write) that sits outside the frame stopwatch. The reported end-to-end FPS
    # (frames / elapsed_s) therefore carries this fixed logging overhead, which
    # makes the headline throughput a touch conservative - never optimistic.
    # (Model loading and VideoCapture open happen before t_video_start and are
    # NOT counted.)
    elapsed_s = time.perf_counter() - t_video_start
    print(f"  {Path(video_path).name}: {frame_id} frames in {elapsed_s:.1f}s "
          f"({frame_id / elapsed_s:.2f} FPS)")
    return frame_id, elapsed_s


CSV_HEADER = ["video", "frame_id", "num_persons", "frame_ms", "temp_c", "ts_unix"]


def write_baseline_temp(path, temp_c):
    Path(path).write_text(json.dumps({"start_temp_c": round(temp_c, 1)}), encoding="utf-8")


def read_baseline_temp(path):
    try:
        return float(json.loads(Path(path).read_text(encoding="utf-8"))["start_temp_c"])
    except (OSError, ValueError, KeyError):
        return None


def wait_until_cooled(target_c, tolerance_c, poll_s, timeout_s):
    deadline = time.time() + timeout_s
    while True:
        current = read_temp_c()
        if current <= target_c + tolerance_c:
            print(f"  cooled to {current:.1f} C (target {target_c:.1f} +/- {tolerance_c:.1f}) - starting")
            return current
        if time.time() >= deadline:
            print(f"  cooldown timeout after {timeout_s}s at {current:.1f} C - starting anyway")
            return current
        print(f"  waiting to cool: {current:.1f} C > {target_c + tolerance_c:.1f} C, sleeping {poll_s}s")
        time.sleep(poll_s)
