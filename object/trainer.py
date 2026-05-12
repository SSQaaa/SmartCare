import argparse
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from camera.depth_camera import open_required_depth_camera
from .memory import activate_object, init_db, upsert_object


SMART_CARE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SMART_CARE_ROOT.parent
YOLOV5_DIR = PROJECT_ROOT / "yolov5"
PERSONAL_ROOT = SMART_CARE_ROOT / "personal_objects"

START_KEYS = {ord("s"), ord("S")}
END_KEYS = {ord("e"), ord("E")}
QUIT_KEYS = {ord("q"), ord("Q")}

MIN_MANUAL_KEYFRAMES = 5
MAX_MANUAL_KEYFRAMES = 20
MANUAL_KEYFRAMES_PER_SECOND = 1.0

AUG_OUTPUTS_PER_EXAMPLE = 3
VAL_RATIO = 0.10

TRAIN_EPOCHS = 150
TRAIN_BATCH_SIZE = 4
TRAIN_PATIENCE = 10
TRAIN_IMG_SIZE = 640
TRAIN_WORKERS = 2


def make_tracker():
    tracker_names = (
        "TrackerCSRT_create",
        "TrackerKCF_create",
        "TrackerMOSSE_create",
        "TrackerMIL_create",
    )
    for name in tracker_names:
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, name):
            return getattr(cv2.legacy, name)()
        if hasattr(cv2, name):
            return getattr(cv2, name)()
    raise RuntimeError(
        "No OpenCV tracker is available. Install opencv-contrib-python, "
        "or use an environment that provides CSRT/KCF/MOSSE/MIL trackers."
    )


def normalize_object_name(name):
    normalized = []
    for ch in name.strip().lower():
        if ch.isalnum():
            normalized.append(ch)
        else:
            normalized.append("_")
    value = "".join(normalized).strip("_")
    while "__" in value:
        value = value.replace("__", "_")
    return value or "personal_object"


def xyxy_to_yolo(box, width, height):
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return (
        cx / width,
        cy / height,
        w / width,
        h / height,
    )


def yolo_to_xyxy(box, width, height):
    cx, cy, w, h = box
    w_px = w * width
    h_px = h * height
    cx_px = cx * width
    cy_px = cy * height
    x1 = cx_px - w_px / 2.0
    y1 = cy_px - h_px / 2.0
    x2 = cx_px + w_px / 2.0
    y2 = cy_px + h_px / 2.0
    return [x1, y1, x2, y2]


def clip_box(box, width, height):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width - 1, int(round(x2))))
    y2 = max(0, min(height - 1, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def to_tracker_box(box, width, height):
    clipped = clip_box(box, width, height)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None
    return (int(x1), int(y1), int(w), int(h))


def ensure_dirs(object_name):
    object_dir = PERSONAL_ROOT / object_name
    dataset_dir = object_dir / "dataset"
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    train_images = images_dir / "train"
    val_images = images_dir / "val"
    train_labels = labels_dir / "train"
    val_labels = labels_dir / "val"
    for path in [train_images, val_images, train_labels, val_labels]:
        path.mkdir(parents=True, exist_ok=True)
    return {
        "object_dir": object_dir,
        "dataset_dir": dataset_dir,
        "train_images": train_images,
        "val_images": val_images,
        "train_labels": train_labels,
        "val_labels": val_labels,
    }


def reset_dataset_dir(dataset_dir):
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)


def save_sample(image_path, label_path, frame, box):
    cv2.imwrite(str(image_path), frame)
    if box is None:
        label_path.write_text("", encoding="utf-8")
        return
    h, w = frame.shape[:2]
    cx, cy, bw, bh = xyxy_to_yolo(box, w, h)
    label_path.write_text(
        f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n",
        encoding="utf-8",
    )


def get_video_writer(video_path, frame_or_shape, fps):
    if hasattr(frame_or_shape, "shape"):
        height, width = frame_or_shape.shape[:2]
    else:
        height, width = frame_or_shape[:2]
    width = int(width)
    height = int(height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(video_path), fourcc, float(fps), (width, height))


def draw_record_ui(frame, object_name, recording, elapsed_seconds):
    canvas = frame.copy()
    line1 = "Object capture"
    line2 = "Press S to start, E to stop, Q to quit"
    line3 = f"Recording: {'YES' if recording else 'NO'}  Time: {elapsed_seconds:.1f}s"
    color = (0, 0, 255) if recording else (0, 255, 0)
    cv2.putText(canvas, line1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(canvas, line2, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, line3, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return canvas


def record_object_video(object_name, camera_index):
    object_dir = PERSONAL_ROOT / object_name / "captures"
    object_dir.mkdir(parents=True, exist_ok=True)
    video_path = object_dir / f"{object_name}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

    cap = open_required_depth_camera()
    if not cap.isOpened():
        raise RuntimeError("Unable to open Orbbec depth camera.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 20.0

    writer = None
    recording = False
    started_at = 0.0
    missed_frames = 0
    max_missed_frames = 100

    print("Please place the object near the center, press S to start, E to stop.")
    cv2.namedWindow("Record Object", cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            missed_frames += 1
            if missed_frames == 1:
                print("Waiting for camera frames...")
            if missed_frames >= max_missed_frames:
                break
            cv2.waitKey(30)
            continue
        missed_frames = 0

        elapsed = time.time() - started_at if recording else 0.0
        preview = draw_record_ui(frame, object_name, recording, elapsed)
        cv2.imshow("Record Object", preview)
        key = cv2.waitKey(1) & 0xFF

        if key in START_KEYS and not recording:
            writer = get_video_writer(video_path, frame, fps)
            recording = True
            started_at = time.time()
            print(f"Recording started: {video_path}")
        elif key in END_KEYS and recording:
            print("Recording stopped.")
            break
        elif key in QUIT_KEYS:
            if writer is not None:
                writer.release()
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError("Recording cancelled.")

        if recording and writer is not None:
            writer.write(frame)

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

    if not video_path.exists() or video_path.stat().st_size == 0:
        raise RuntimeError("No video was recorded.")
    return video_path


def load_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 20.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("The recorded video contains no readable frames.")
    return frames, float(fps)


def choose_even_indices(count, total):
    if total <= 0 or count <= 0:
        return []
    count = min(count, total)
    if count == 1:
        return [total // 2]
    values = np.linspace(0, total - 1, count)
    indices = sorted({int(round(value)) for value in values})
    while len(indices) < count:
        for idx in range(total):
            if idx not in indices:
                indices.append(idx)
                break
        indices.sort()
    return indices[:count]


def get_adaptive_count(frame_count, fps, rate, min_count, max_count):
    duration = frame_count / max(fps, 1.0)
    count = int(round(duration * rate))
    count = max(min_count, count)
    count = min(max_count, count)
    count = min(count, frame_count)
    return max(1, count)


def prompt_empty_frame_choice():
    print("No ROI selected. Enter n for no object, r to redraw, q to quit.")
    while True:
        answer = input("Choice [n/r/q]: ").strip().lower()
        if answer in {"n", "r", "q"}:
            return answer


def manual_label_frame(frame, title):
    while True:
        roi = cv2.selectROI(title, frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(title)
        x, y, w, h = [int(v) for v in roi]
        if w > 0 and h > 0:
            return [x, y, x + w, y + h]
        choice = prompt_empty_frame_choice()
        if choice == "n":
            return None
        if choice == "q":
            raise RuntimeError("Annotation cancelled by user.")


def collect_manual_keyframe_annotations(frames, fps, object_name):
    frame_count = len(frames)
    keyframe_count = get_adaptive_count(
        frame_count,
        fps,
        MANUAL_KEYFRAMES_PER_SECOND,
        MIN_MANUAL_KEYFRAMES,
        MAX_MANUAL_KEYFRAMES,
    )
    indices = choose_even_indices(keyframe_count, frame_count)
    annotations = []
    print(f"Manual labeling {len(indices)} keyframes based on video duration.")
    for idx in indices:
        frame = frames[idx]
        preview = frame.copy()
        cv2.putText(
            preview,
            f"Frame {idx + 1}/{frame_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
        box = manual_label_frame(preview, "Keyframe Label")
        annotations.append({"frame_index": idx, "box": box, "source": "manual"})
    return annotations


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def interpolate_size(start_box, end_box, alpha):
    sw = start_box[2] - start_box[0]
    sh = start_box[3] - start_box[1]
    ew = end_box[2] - end_box[0]
    eh = end_box[3] - end_box[1]
    width = sw + (ew - sw) * alpha
    height = sh + (eh - sh) * alpha
    return max(2.0, width), max(2.0, height)


def track_segment(frames, start_idx, end_idx, start_box, end_box):
    height, width = frames[start_idx].shape[:2]
    start_box = clip_box(start_box, width, height)
    end_box = clip_box(end_box, width, height)
    if start_box is None or end_box is None:
        return {}

    tracked = {start_idx: start_box}
    if end_idx <= start_idx + 1:
        return tracked

    tracker = make_tracker()
    init_box = to_tracker_box(start_box, width, height)
    if init_box is None:
        return tracked
    try:
        tracker.init(frames[start_idx], init_box)
    except cv2.error as exc:
        print(f"Tracker init failed for frame {start_idx + 1}: {exc}")
        return tracked
    total_steps = end_idx - start_idx

    for idx in range(start_idx + 1, end_idx):
        try:
            ok, tracked_box = tracker.update(frames[idx])
        except cv2.error:
            continue
        if not ok:
            continue
        tx, ty, tw, th = tracked_box
        center_x = tx + tw / 2.0
        center_y = ty + th / 2.0
        alpha = (idx - start_idx) / max(total_steps, 1)
        interp_w, interp_h = interpolate_size(start_box, end_box, alpha)
        box = [
            center_x - interp_w / 2.0,
            center_y - interp_h / 2.0,
            center_x + interp_w / 2.0,
            center_y + interp_h / 2.0,
        ]
        height, width = frames[idx].shape[:2]
        clipped = clip_box(box, width, height)
        if clipped is not None:
            tracked[idx] = clipped
    return tracked


def merge_boxes_with_anchor_labels(frames, anchor_annotations):
    positive = sorted(
        [ann for ann in anchor_annotations if ann["box"] is not None],
        key=lambda item: item["frame_index"],
    )
    tracked = {}
    for ann in positive:
        tracked[ann["frame_index"]] = ann["box"]
    for left, right in zip(positive, positive[1:]):
        segment = track_segment(
            frames,
            left["frame_index"],
            right["frame_index"],
            left["box"],
            right["box"],
        )
        tracked.update(segment)
        tracked[right["frame_index"]] = right["box"]
    return tracked


def apply_affine_to_box(box, matrix, width, height):
    points = np.array(
        [
            [box[0], box[1], 1.0],
            [box[2], box[1], 1.0],
            [box[2], box[3], 1.0],
            [box[0], box[3], 1.0],
        ],
        dtype=np.float32,
    )
    warped = points @ matrix.T
    x1 = float(np.min(warped[:, 0]))
    y1 = float(np.min(warped[:, 1]))
    x2 = float(np.max(warped[:, 0]))
    y2 = float(np.max(warped[:, 1]))
    return clip_box([x1, y1, x2, y2], width, height)


def maybe_flip(frame, box, rng):
    height, width = frame.shape[:2]
    if rng.random() < 0.5:
        frame = cv2.flip(frame, 1)
        if box is not None:
            x1, y1, x2, y2 = box
            box = [width - x2, y1, width - x1, y2]
    if rng.random() < 0.5:
        frame = cv2.flip(frame, 0)
        if box is not None:
            x1, y1, x2, y2 = box
            box = [x1, height - y2, x2, height - y1]
    return frame, clip_box(box, width, height) if box is not None else None


def apply_geometric_aug(frame, box, rng):
    height, width = frame.shape[:2]
    center = (width / 2.0, height / 2.0)
    angle = rng.uniform(-15.0, 15.0)
    scale = rng.uniform(0.74, 1.0)
    shear_x = math.tan(math.radians(rng.uniform(-15.0, 15.0)))
    shear_y = math.tan(math.radians(rng.uniform(-15.0, 15.0)))
    tx = rng.uniform(-0.05, 0.05) * width
    ty = rng.uniform(-0.05, 0.05) * height

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    affine = np.vstack([matrix, [0.0, 0.0, 1.0]])
    shear = np.array(
        [
            [1.0, shear_x, 0.0],
            [shear_y, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    translate = np.array(
        [
            [1.0, 0.0, tx],
            [0.0, 1.0, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    final = translate @ shear @ affine
    warped = cv2.warpAffine(
        frame,
        final[:2],
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    new_box = apply_affine_to_box(box, final[:2], width, height) if box is not None else None
    return warped, new_box


def apply_color_aug(frame, rng):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_shift = rng.uniform(-17.0, 17.0)
    sat_scale = 1.0 + rng.uniform(-0.22, 0.22)
    val_scale = 1.0 + rng.uniform(-0.17, 0.17)
    exposure_scale = 1.0 + rng.uniform(-0.15, 0.15)

    hsv[..., 0] = (hsv[..., 0] + hue_shift / 2.0) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_scale, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    out = np.clip(out * exposure_scale, 0, 255).astype(np.uint8)
    return out


def apply_blur_and_noise(frame, rng):
    out = frame.copy()
    blur_sigma = rng.uniform(0.0, 4.1)
    if blur_sigma > 0.2:
        out = cv2.GaussianBlur(out, (0, 0), blur_sigma)
    noise_ratio = rng.uniform(0.0, 0.0038)
    noise_pixels = int(noise_ratio * out.shape[0] * out.shape[1])
    if noise_pixels > 0:
        ys = rng.integers(0, out.shape[0], size=noise_pixels)
        xs = rng.integers(0, out.shape[1], size=noise_pixels)
        values = rng.integers(0, 256, size=(noise_pixels, 3), dtype=np.uint8)
        out[ys, xs] = values
    return out


def build_augmented_variant(frame, box, seed):
    rng = np.random.default_rng(seed)
    aug_frame = frame.copy()
    aug_box = box[:] if box is not None else None
    aug_frame, aug_box = maybe_flip(aug_frame, aug_box, rng)
    aug_frame, aug_box = apply_geometric_aug(aug_frame, aug_box, rng)
    aug_frame = apply_color_aug(aug_frame, rng)
    aug_frame = apply_blur_and_noise(aug_frame, rng)
    if aug_box is not None:
        aug_box = clip_box(aug_box, aug_frame.shape[1], aug_frame.shape[0])
    return aug_frame, aug_box


def save_augmented_variants(image_dir, label_dir, stem, frame, box):
    if box is None:
        return
    for idx in range(AUG_OUTPUTS_PER_EXAMPLE):
        aug_frame, aug_box = build_augmented_variant(frame, box, seed=hash((stem, idx)) & 0xFFFFFFFF)
        if aug_box is None:
            continue
        save_sample(
            image_dir / f"{stem}_aug{idx + 1}.jpg",
            label_dir / f"{stem}_aug{idx + 1}.txt",
            aug_frame,
            aug_box,
        )


def collect_review_samples(frames, tracked_boxes, sample_stride):
    samples = []
    for frame_index in sorted(tracked_boxes.keys()):
        if frame_index % sample_stride != 0:
            continue
        samples.append(
            {
                "frame_index": frame_index,
                "frame": frames[frame_index].copy(),
                "box": tracked_boxes[frame_index],
                "source": "tracked",
            }
        )
    return samples


def collect_negative_review_samples(frames, annotations, sample_stride):
    negatives = []
    for ann in annotations:
        if ann["box"] is not None:
            continue
        if ann["frame_index"] % sample_stride != 0 and ann["source"] != "manual":
            continue
        negatives.append(
            {
                "frame_index": ann["frame_index"],
                "frame": frames[ann["frame_index"]].copy(),
                "box": None,
                "source": ann["source"],
            }
        )
    return negatives


def draw_review_sample(sample, object_name, draft_box=None):
    frame = sample["frame"].copy()
    box = draft_box if draft_box is not None else sample["box"]
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    line1 = f"Object sample  Frame: {sample['frame_index']}"
    line2 = "Drag mouse to draw bbox directly. A/D prev next, C no-object, Space confirm, Q quit"
    line3 = f"Source: {sample['source']}"
    if sample["box"] is None and draft_box is None:
        line3 += "  Negative sample"
    cv2.putText(frame, line1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(frame, line2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, line3, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def review_all_samples(samples, object_name):
    if not samples:
        return []

    state = {
        "drawing": False,
        "start": None,
        "draft_box": None,
    }
    window_name = "Review All Samples"

    def mouse_callback(event, x, y, _flags, _userdata):
        current = samples[index_holder[0]]
        width = current["frame"].shape[1]
        height = current["frame"].shape[0]
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["start"] = (x, y)
            state["draft_box"] = [x, y, x, y]
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            x0, y0 = state["start"]
            state["draft_box"] = clip_box([min(x0, x), min(y0, y), max(x0, x), max(y0, y)], width, height)
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            state["drawing"] = False
            x0, y0 = state["start"]
            current["box"] = clip_box([min(x0, x), min(y0, y), max(x0, x), max(y0, y)], width, height)
            state["draft_box"] = None
            state["start"] = None

    index_holder = [0]
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        current = samples[index_holder[0]]
        preview = draw_review_sample(current, object_name, state["draft_box"])
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("a"):
            state["draft_box"] = None
            index_holder[0] = max(0, index_holder[0] - 1)
        elif key == ord("d"):
            state["draft_box"] = None
            index_holder[0] = min(len(samples) - 1, index_holder[0] + 1)
        elif key == ord("c") or key == ord("x"):
            current["box"] = None
            state["draft_box"] = None
        elif key in (13, 32):
            break
        elif key == ord("q"):
            cv2.destroyWindow(window_name)
            raise RuntimeError("Review cancelled by user.")

    cv2.destroyWindow(window_name)
    return samples


def write_dataset_yaml(dataset_dir, object_name):
    data_yaml = dataset_dir / f"{object_name}.yaml"
    payload = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {0: object_name},
        "nc": 1,
    }
    data_yaml.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return data_yaml


def image_files(path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts])


def paired_label_path(image_path, label_dir):
    return label_dir / f"{image_path.stem}.txt"


def move_sample_pair(image_path, src_label_dir, dst_image_dir, dst_label_dir):
    label_path = paired_label_path(image_path, src_label_dir)
    dst_image = dst_image_dir / image_path.name
    dst_label = dst_label_dir / label_path.name
    if dst_image.exists():
        dst_image.unlink()
    if dst_label.exists():
        dst_label.unlink()
    shutil.move(str(image_path), str(dst_image))
    if label_path.exists():
        shutil.move(str(label_path), str(dst_label))


def rebalance_dataset_split(dirs, val_ratio=VAL_RATIO):
    train_images = image_files(dirs["train_images"])
    val_images = image_files(dirs["val_images"])
    total = len(train_images) + len(val_images)
    if total <= 1:
        return len(train_images), len(val_images)

    target_val = int(round(total * val_ratio))
    target_val = max(1, min(total - 1, target_val))

    if len(val_images) < target_val:
        need = target_val - len(val_images)
        candidates = [p for p in train_images if "_aug" not in p.stem]
        if len(candidates) < need:
            candidates = train_images
        for image_path in candidates[:need]:
            move_sample_pair(image_path, dirs["train_labels"], dirs["val_images"], dirs["val_labels"])
    elif len(val_images) > target_val:
        need = len(val_images) - target_val
        for image_path in val_images[:need]:
            move_sample_pair(image_path, dirs["val_labels"], dirs["train_images"], dirs["train_labels"])

    train_count = len(image_files(dirs["train_images"]))
    val_count = len(image_files(dirs["val_images"]))
    print(f"Final dataset split: train={train_count}, val={val_count}, target val_ratio≈{val_ratio:.0%}")
    return train_count, val_count


def split_samples_roughly(samples, val_ratio=VAL_RATIO):
    positives = [sample for sample in samples if sample["box"] is not None]
    negatives = [sample for sample in samples if sample["box"] is None]
    if not positives:
        return [], []

    rng = random.Random(20260506)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    val_count = int(round(len(positives) * val_ratio))
    if len(positives) > 1:
        val_count = max(1, min(len(positives) - 1, val_count))
    else:
        val_count = 0

    val_samples = positives[:val_count]
    train_samples = positives[val_count:]

    neg_val_count = int(round(len(negatives) * val_ratio))
    val_samples.extend(negatives[:neg_val_count])
    train_samples.extend(negatives[neg_val_count:])

    train_samples.sort(key=lambda item: item["frame_index"])
    val_samples.sort(key=lambda item: item["frame_index"])
    return train_samples, val_samples


def save_reviewed_samples_to_dataset(reviewed_samples, dirs, object_name, val_ratio=VAL_RATIO):
    train_samples, val_samples = split_samples_roughly(reviewed_samples, val_ratio)
    sample_count = 0

    for target_split, samples in (("train", train_samples), ("val", val_samples)):
        image_dir = dirs["val_images"] if target_split == "val" else dirs["train_images"]
        label_dir = dirs["val_labels"] if target_split == "val" else dirs["train_labels"]
        for sample in samples:
            frame = sample["frame"]
            box = sample["box"]
            frame_index = sample["frame_index"]
            stem = f"{object_name}_{frame_index:04d}"
            save_sample(image_dir / f"{stem}.jpg", label_dir / f"{stem}.txt", frame, box)
            sample_count += 1
            if target_split == "train":
                save_augmented_variants(image_dir, label_dir, stem, frame, box)

    print(
        f"Dataset split: train originals={len(train_samples)}, "
        f"val originals={len(val_samples)}, val_ratio≈{val_ratio:.0%}"
    )
    rebalance_dataset_split(dirs, val_ratio)
    return sample_count


def build_dataset_from_video(video_path, object_name, display_name, sample_stride=3, val_ratio=VAL_RATIO):
    frames, fps = load_video_frames(video_path)
    dirs = ensure_dirs(object_name)
    reset_dataset_dir(dirs["dataset_dir"])
    dirs = ensure_dirs(object_name)

    manual_annotations = collect_manual_keyframe_annotations(frames, fps, object_name)
    anchor_annotations = manual_annotations
    positive_anchors = [ann for ann in anchor_annotations if ann["box"] is not None]
    tracked_boxes = merge_boxes_with_anchor_labels(frames, anchor_annotations) if positive_anchors else {}

    review_samples = collect_review_samples(frames, tracked_boxes, sample_stride)
    review_samples.extend(collect_negative_review_samples(frames, anchor_annotations, sample_stride))
    review_samples.sort(key=lambda item: item["frame_index"])
    reviewed_samples = review_all_samples(review_samples, object_name)

    if not any(sample["box"] is not None for sample in reviewed_samples):
        raise RuntimeError("At least one positive sample is required for training.")

    sample_count = save_reviewed_samples_to_dataset(reviewed_samples, dirs, object_name, val_ratio)

    data_yaml = write_dataset_yaml(dirs["dataset_dir"], object_name)
    return dirs["dataset_dir"], data_yaml, sample_count


def train_personal_model(
    object_name,
    data_yaml,
    epochs=TRAIN_EPOCHS,
    batch_size=TRAIN_BATCH_SIZE,
    img_size=TRAIN_IMG_SIZE,
    patience=TRAIN_PATIENCE,
    workers=TRAIN_WORKERS,
):
    object_dir = PERSONAL_ROOT / object_name
    run_dir = object_dir / "runs" / "finetune"
    run_dir.mkdir(parents=True, exist_ok=True)
    weights = YOLOV5_DIR / "yolov5s.pt"
    train_script = YOLOV5_DIR / "train.py"

    cmd = [
        sys.executable,
        str(train_script),
        "--img",
        str(img_size),
        "--batch",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--workers",
        str(workers),
        "--data",
        str(data_yaml),
        "--weights",
        str(weights),
        "--project",
        str(object_dir / "runs"),
        "--name",
        "finetune",
        "--exist-ok",
    ]
    print(
        "Starting YOLOv5 fine-tuning. "
        f"img={img_size}, batch={batch_size}, workers={workers}. This can take several minutes..."
    )
    subprocess.run(cmd, cwd=str(YOLOV5_DIR), check=True)
    best_weights = object_dir / "runs" / "finetune" / "weights" / "best.pt"
    if not best_weights.exists():
        raise RuntimeError("Training finished but best.pt was not found.")
    return best_weights


def train_existing_dataset(
    object_name,
    display_name=None,
    epochs=TRAIN_EPOCHS,
    batch_size=TRAIN_BATCH_SIZE,
    img_size=TRAIN_IMG_SIZE,
    patience=TRAIN_PATIENCE,
    workers=TRAIN_WORKERS,
):
    object_dir = PERSONAL_ROOT / object_name
    dataset_dir = object_dir / "dataset"
    data_yaml = dataset_dir / f"{object_name}.yaml"
    if not data_yaml.exists():
        raise RuntimeError(f"Existing dataset yaml not found: {data_yaml}")

    dirs = {
        "train_images": dataset_dir / "images" / "train",
        "val_images": dataset_dir / "images" / "val",
        "train_labels": dataset_dir / "labels" / "train",
        "val_labels": dataset_dir / "labels" / "val",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    rebalance_dataset_split(dirs, VAL_RATIO)

    display_name = display_name or object_name
    weights_path = train_personal_model(
        object_name,
        data_yaml,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        patience=patience,
        workers=workers,
    )
    init_db()
    upsert_object(
        object_name=object_name,
        display_name=display_name,
        dataset_dir=str(dataset_dir),
        data_yaml=str(data_yaml),
        weights_path=str(weights_path),
        status="trained",
    )
    activate_object(object_name)
    print(f"[OK] Object: {object_name}")
    print(f"[OK] Dataset: {dataset_dir}")
    print(f"[OK] Weights: {weights_path}")
    print("[OK] This personal object is now active.")
    return weights_path


def run_pipeline(
    video_path,
    object_name,
    display_name,
    sample_stride=3,
    epochs=TRAIN_EPOCHS,
    batch_size=TRAIN_BATCH_SIZE,
    img_size=TRAIN_IMG_SIZE,
    patience=TRAIN_PATIENCE,
    workers=TRAIN_WORKERS,
):
    dataset_dir, data_yaml, sample_count = build_dataset_from_video(
        video_path,
        object_name,
        display_name,
        sample_stride=sample_stride,
    )
    weights_path = train_personal_model(
        object_name,
        data_yaml,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        patience=patience,
        workers=workers,
    )
    init_db()
    upsert_object(
        object_name=object_name,
        display_name=display_name,
        video_path=str(video_path),
        dataset_dir=str(dataset_dir),
        data_yaml=str(data_yaml),
        weights_path=str(weights_path),
        status="trained",
    )
    activate_object(object_name)
    print(f"[OK] Object: {object_name}")
    print(f"[OK] Dataset: {dataset_dir}")
    print(f"[OK] Weights: {weights_path}")
    print("[OK] This personal object is now active.")
    print(f"[OK] Reviewed samples used: {sample_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--object-name", type=str, default="")
    parser.add_argument("--display-name", type=str, default="")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--img-size", type=int, default=TRAIN_IMG_SIZE)
    parser.add_argument("--patience", type=int, default=TRAIN_PATIENCE)
    args = parser.parse_args()

    raw_name = args.object_name.strip() if args.object_name else input("Enter object name: ").strip()
    if not raw_name:
        raise RuntimeError("Object name is required.")
    object_name = normalize_object_name(raw_name)
    display_name = args.display_name.strip()
    if not display_name:
        display_name = input("Enter display object name / 请输入物品显示名: ").strip() or raw_name

    if args.video:
        video_path = Path(args.video).resolve()
    else:
        video_path = record_object_video(object_name, args.camera_index)

    run_pipeline(
        video_path=video_path,
        object_name=object_name,
        display_name=display_name,
        sample_stride=max(1, args.sample_stride),
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
