import atexit
import io
import os
import re
import sys
import threading
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch

from camera.depth_camera import open_required_depth_camera
from .memory import get_active_object, init_db

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
SMART_CARE_ROOT = FILE.parents[1]
ULTRALYTICS_CONFIG_DIR = SMART_CARE_ROOT / ".ultralytics"
ULTRALYTICS_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))

# 运行开关区：把 USE_DEPTH_CAMERA 改成 True 就会打开 Orbbec 深度相机。
# 如果 USE_DEPTH_CAMERA 为 False，则使用普通摄像头/视频源 VIDEO_SOURCE。
USE_DEPTH_CAMERA = True
VIDEO_SOURCE = 0
USE_PERSONAL_MODEL = True

YOLOV5_ROOT = ROOT / "yolov5"
YOLOV5_MODEL_PATH = YOLOV5_ROOT / "yolov5s.pt"
YOLOV5_DATA_PATH = YOLOV5_ROOT / "data" / "coco.yaml"
CN_NAME_MAP_PATH = SMART_CARE_ROOT / "data" / "coco80_cn_names.txt"
ORBBEC_CPP_SDK_ROOT = SMART_CARE_ROOT / "third_party" / "OrbbecSDK_C_C++_v1.5.7_win_x64_release" / "SDK"
ORBBEC_CPP_DLL_DIR = ORBBEC_CPP_SDK_ROOT / "lib"
CAMERA_ROOT = SMART_CARE_ROOT / "camera"

ORBBEC_SDK_DIR = Path(
    r"E:\SSQ\Sophomore\RobotStar\dog\奥比中光系列相机资料\奥比中光系列相机资料_2024.01.02"
    r"\6.Orbbec SDK使用【Linux、Windows】\Windows_SDK\OrbbecSDK_Python_v1.1.4_win_x64_release"
    r"\OrbbecSDK_Python_v1.1.4_win_x64_release\python3.9\Samples"
)

if str(YOLOV5_ROOT) not in sys.path:
    sys.path.append(str(YOLOV5_ROOT))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device

IMG_SIZE = (640, 640)
COCO_CONF_THRES = 0.60
PERSONAL_CONF_THRES = 0.40
IOU_THRES = 0.45
MAX_DET = 30
DEVICE = ""
HALF = False

# Only detect these names. Empty means all COCO/personal classes are allowed.
TARGET_NAMES = []

# Large objects that can support smaller objects, used for "on the table/bed/chair" relations.
SUPPORT_NAMES = {"dining table", "bed", "couch", "chair"}

# Small objects for which relative position descriptions are useful.
RELATION_OBJECT_NAMES = {
    "cup", "bottle", "cell phone", "remote", "book",
    "bowl", "fork", "knife", "spoon", "mouse"
}

# Detected but not displayed or spoken.
IGNORE_NAMES = {"person"}

MAX_DEPTH_MM = 5000
MIN_VALID_DEPTH_COUNT = 20
DEPTH_VIS_MAX_MM = 3000
DEPTH_HISTORY_LEN = 5
INNER_BOX_SCALE = 0.5

BROADCAST_INTERVAL_FRAMES = 30
MAX_LINES = 8
ESC = 27
Q_KEY = ord("q")
V_KEY = ord("v")

ENABLE_VL_OBJECT_FINDER = True
VL_INITIAL_QUERY = ""
VL_FIND_INTERVAL_FRAMES = 60
# Safe default for live camera use. Set to "qwen" after the Qwen3-VL environment
# and model cache are ready; loading a 4B VL model inside the camera loop can stall.
VL_BACKEND = "opencv"
VL_MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
VL_FALLBACK_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
SUPPRESS_ORBBEC_ENDPOINT_LOGS = True

depth_history_map = defaultdict(lambda: deque(maxlen=DEPTH_HISTORY_LEN))
PERSONAL_NAME_MAP = {}
_endpoint_log_filter_installed = False
_endpoint_log_filter_restore_fd = None
_endpoint_log_filter_original_stdout = None


def install_orbbec_endpoint_log_filter():
    global _endpoint_log_filter_installed, _endpoint_log_filter_restore_fd, _endpoint_log_filter_original_stdout

    if _endpoint_log_filter_installed or not SUPPRESS_ORBBEC_ENDPOINT_LOGS:
        return
    if not hasattr(os, "dup2"):
        return

    try:
        _endpoint_log_filter_original_stdout = sys.stdout
        original_stdout_fd = os.dup(1)
        _endpoint_log_filter_restore_fd = os.dup(1)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 1)
        os.close(write_fd)
        stdout_encoding = getattr(_endpoint_log_filter_original_stdout, "encoding", None) or "utf-8"
        sys.stdout = io.TextIOWrapper(
            os.fdopen(os.dup(1), "wb", buffering=0),
            encoding=stdout_encoding,
            errors="replace",
            line_buffering=True,
            write_through=True,
        )
    except OSError:
        return

    _endpoint_log_filter_installed = True
    endpoint_pattern = re.compile(rb"Endpoint 0x[0-9A-Fa-f]+ bandwidth:")

    def restore_stdout():
        global _endpoint_log_filter_restore_fd, _endpoint_log_filter_original_stdout

        if _endpoint_log_filter_restore_fd is None:
            return
        try:
            try:
                sys.stdout.flush()
            except Exception:
                pass
            os.dup2(_endpoint_log_filter_restore_fd, 1)
            os.close(_endpoint_log_filter_restore_fd)
            if _endpoint_log_filter_original_stdout is not None:
                sys.stdout = _endpoint_log_filter_original_stdout
        except OSError:
            pass
        _endpoint_log_filter_restore_fd = None
        _endpoint_log_filter_original_stdout = None

    def pump_stdout():
        pending = b""
        with os.fdopen(read_fd, "rb", buffering=0) as reader, os.fdopen(original_stdout_fd, "wb", buffering=0) as writer:
            while True:
                chunk = reader.read(1024)
                if not chunk:
                    if pending and not endpoint_pattern.search(pending):
                        writer.write(pending)
                        writer.flush()
                    break

                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if endpoint_pattern.search(line):
                        continue
                    writer.write(line + b"\n")
                    writer.flush()

    atexit.register(restore_stdout)
    threading.Thread(target=pump_stdout, name="orbbec-endpoint-log-filter", daemon=True).start()


def load_cn_name_map(path):
    name_map = {}
    if not path.exists():
        legacy_path = SMART_CARE_ROOT / "coco80_cn_names.txt"
        if legacy_path.exists():
            path = legacy_path
        else:
            print(f"Chinese name map file not found: {path}")
            return name_map

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            print(f"Skip invalid name map line {line_no}: {raw_line}")
            continue
        english_name, chinese_name = [part.strip() for part in line.split("=", 1)]
        if english_name and chinese_name:
            name_map[english_name] = chinese_name
    return name_map


CN_NAME_MAP = load_cn_name_map(CN_NAME_MAP_PATH)


def get_cn_name(name):
    if name in PERSONAL_NAME_MAP:
        return PERSONAL_NAME_MAP[name]
    return CN_NAME_MAP.get(name, name)


def load_detector(weights_path, data_path):
    print(f"Loading YOLO detector: {weights_path}")
    device = select_device(DEVICE)
    model = DetectMultiBackend(weights_path, device=device, data=data_path, fp16=HALF)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(IMG_SIZE, s=stride)
    model.warmup(imgsz=(1, 3, *imgsz))
    print(f"YOLO detector ready: {weights_path}")
    return model, device, stride, names, pt, imgsz


def load_model():
    return load_detector(YOLOV5_MODEL_PATH, YOLOV5_DATA_PATH)


def load_active_personal_model():
    init_db()
    active = get_active_object()
    if not active:
        print("No active personal object found in database.")
        return None

    object_name = active["object_name"]
    weights_path = active["weights_path"]
    data_yaml = active["data_yaml"]
    object_root = SMART_CARE_ROOT / "personal_objects" / object_name

    if not weights_path:
        fallback_weights = object_root / "runs" / "finetune" / "weights" / "best.pt"
        if fallback_weights.exists():
            weights_path = str(fallback_weights)
            print(f"Personal weights_path missing in DB, fallback to {fallback_weights}")

    if not data_yaml:
        fallback_yaml = object_root / "dataset" / f"{object_name}.yaml"
        if fallback_yaml.exists():
            data_yaml = str(fallback_yaml)
            print(f"Personal data_yaml missing in DB, fallback to {fallback_yaml}")

    if not weights_path or not data_yaml:
        print("Active personal object has no valid weights or data yaml.")
        return None

    weights_path = Path(weights_path)
    data_yaml = Path(data_yaml)
    if not weights_path.exists() or not data_yaml.exists():
        print("Active personal object paths do not exist.")
        return None

    PERSONAL_NAME_MAP[active["object_name"]] = active["display_name"] or active["object_name"]
    model, device, stride, names, pt, imgsz = load_detector(weights_path, data_yaml)
    print(f"Loaded personal object model: {active['object_name']} -> {weights_path}")
    return {
        "model": model,
        "device": device,
        "stride": stride,
        "names": names,
        "pt": pt,
        "imgsz": imgsz,
        "object_name": active["object_name"],
    }


def detect_objects(model, frame, device, pt, stride, names, imgsz, conf_thres, source):
    im = letterbox(frame, imgsz, stride=stride, auto=pt)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)

    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()
    im /= 255.0
    im = im.unsqueeze(0)

    pred = model(im)
    pred = non_max_suppression(pred, conf_thres, IOU_THRES, max_det=MAX_DET)

    detections = []
    det = pred[0]
    if len(det):
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
        for *xyxy, conf, cls in det:
            conf_value = float(conf.item())
            if conf_value < conf_thres:
                continue

            class_id = int(cls)
            class_name = names[class_id]
            if TARGET_NAMES and class_name not in TARGET_NAMES:
                continue

            x1, y1, x2, y2 = [int(v.item()) for v in xyxy]
            detections.append({
                "box": [x1, y1, x2, y2],
                "conf": conf_value,
                "class_id": class_id,
                "class_name": class_name,
                "source": source,
                "depth_mm": None,
                "valid_count": 0,
                "relation_text": "",
            })

    return detections


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def prioritize_personal_detections(detections, iou_thres=0.3):
    personal = [d for d in detections if d.get("source") == "personal"]
    if not personal:
        return detections

    filtered = []
    for det in detections:
        if det.get("source") == "personal":
            filtered.append(det)
            continue

        overlapped = any(compute_iou(det["box"], p["box"]) >= iou_thres for p in personal)
        if not overlapped:
            filtered.append(det)

    return filtered


def get_box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def describe_direction(box, image_shape):
    cx, _ = get_box_center(box)
    width = image_shape[1]

    if cx < width / 3:
        return "左前方"
    if cx > width * 2 / 3:
        return "右前方"
    return "正前方"


def get_relative_position(box_a, box_b):
    ax, ay = get_box_center(box_a)
    bx, by = get_box_center(box_b)
    dx = ax - bx
    dy = ay - by

    if abs(dx) >= abs(dy):
        return "右边" if dx > 0 else "左边"
    if abs(dy) < 40:
        return "旁边"
    return "前面" if dy > 0 else "后面"


def compute_iou_horizontal(box_a, box_b):
    ax1, _, ax2, _ = box_a
    bx1, _, bx2, _ = box_b
    inter = max(0, min(ax2, bx2) - max(ax1, bx1))
    union = max(ax2, bx2) - min(ax1, bx1)
    if union <= 0:
        return 0.0
    return inter / union


def assign_support_relations(detections):
    support_dets = [d for d in detections if d["class_name"] in SUPPORT_NAMES]

    for det in detections:
        if det["class_name"] in IGNORE_NAMES:
            det["relation_text"] = ""
            continue
        det["relation_text"] = ""
        if det["class_name"] not in RELATION_OBJECT_NAMES:
            continue

        best_support = None
        best_score = -1.0
        _, _, _, y2 = det["box"]
        obj_bottom = y2

        for support in support_dets:
            _, sy1, _, sy2 = support["box"]
            horizontal_overlap = compute_iou_horizontal(det["box"], support["box"])
            vertical_ok = sy1 - 80 <= obj_bottom <= sy2 + 40

            score = horizontal_overlap
            if det["depth_mm"] is not None and support["depth_mm"] is not None:
                if abs(det["depth_mm"] - support["depth_mm"]) <= 400:
                    score += 0.5

            if horizontal_overlap > 0.3 and vertical_ok and score > best_score:
                best_score = score
                best_support = support

        if best_support is not None:
            det["relation_text"] = f"在{get_cn_name(best_support['class_name'])}上面"
            det["relation_target_name"] = best_support["class_name"]


def assign_pair_relations(detections):
    for det in detections:
        if det["class_name"] in IGNORE_NAMES:
            det["relation_text"] = ""
            continue
        if det["relation_text"]:
            continue

        best_other = None
        best_distance = float("inf")
        ax, ay = get_box_center(det["box"])

        for other in detections:
            if other is det:
                continue
            if other["class_name"] in IGNORE_NAMES:
                continue
            bx, by = get_box_center(other["box"])
            distance = (ax - bx) ** 2 + (ay - by) ** 2
            if distance < best_distance:
                best_distance = distance
                best_other = other

        if best_other is not None:
            if best_other.get("relation_target_name") == det["class_name"]:
                continue
            pos = get_relative_position(det["box"], best_other["box"])
            other_name = get_cn_name(best_other["class_name"])
            if pos == "旁边":
                det["relation_text"] = f"在{other_name}旁边"
            else:
                det["relation_text"] = f"在{other_name}{pos}"
            det["relation_target_name"] = best_other["class_name"]


def build_object_sentence(det, image_shape):
    if det["class_name"] in IGNORE_NAMES:
        return ""

    class_name = get_cn_name(det["class_name"])
    direction = describe_direction(det["box"], image_shape)

    if USE_DEPTH_CAMERA and det["depth_mm"] is not None:
        depth_m = det["depth_mm"] / 1000.0
        if det["relation_text"]:
            return f"{class_name}{det['relation_text']}，在我{direction}，距离我约{depth_m:.2f}米"
        return f"{class_name}在我{direction}，距离我约{depth_m:.2f}米"

    if det["relation_text"]:
        return f"{class_name}{det['relation_text']}，在我{direction}"
    return f"{class_name}在我{direction}"


def draw_detection(frame, det):
    x1, y1, x2, y2 = det["box"]
    color = (0, 255, 0)
    label = f"{det['class_name']} {det['conf']:.2f}"
    if USE_DEPTH_CAMERA and det["depth_mm"] is not None:
        label += f" {det['depth_mm'] / 1000.0:.2f}m"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_text_panel(frame, sentences):
    text = f"Detected objects: {len(sentences)}"
    cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def draw_vl_result(frame, result):
    if result is None:
        return

    x1, y1, x2, y2 = result.box
    color = (255, 0, 255)
    label = "VL target"
    if result.depth_m is not None:
        label += f" {result.depth_m:.2f}m from camera"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(frame, label, (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def decode_color_frame(color_frame, ob_types):
    color_data = color_frame.data()
    color_width = color_frame.width()
    color_height = color_frame.height()
    color_format = color_frame.format()

    if color_format == ob_types["OB_PY_FORMAT_MJPG"]:
        color_img = cv2.imdecode(color_data, 1)
        if color_img is None:
            return None
        color_img = np.resize(color_img, (color_height, color_width, 3))
    elif color_format == ob_types["OB_PY_FORMAT_RGB888"]:
        color_img = np.resize(color_data, (color_height, color_width, 3))
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    elif color_format == ob_types["OB_PY_FORMAT_YUYV"]:
        color_img = np.resize(color_data, (color_height, color_width, 2))
        color_img = cv2.cvtColor(color_img, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == ob_types["OB_PY_FORMAT_UYVY"]:
        color_img = np.resize(color_data, (color_height, color_width, 2))
        color_img = cv2.cvtColor(color_img, cv2.COLOR_YUV2BGR_UYVY)
    elif color_format == ob_types["OB_PY_FORMAT_I420"]:
        color_img = color_data.reshape((color_height * 3 // 2, color_width))
        color_img = cv2.cvtColor(color_img, cv2.COLOR_YUV2BGR_I420)
        color_img = cv2.resize(color_img, (color_width, color_height))
    else:
        return None

    return color_img


def decode_depth_frame(depth_frame):
    depth_data = depth_frame.data()
    depth_width = depth_frame.width()
    depth_height = depth_frame.height()
    value_scale = depth_frame.getValueScale()

    depth_data = np.resize(depth_data, (depth_height, depth_width, 2))
    depth_u16 = depth_data[:, :, 0].astype(np.uint16) + depth_data[:, :, 1].astype(np.uint16) * 256
    depth_mm = (depth_u16 * value_scale).astype(np.uint16)
    return depth_mm


def scale_box_to_depth(box, color_shape, depth_shape):
    x1, y1, x2, y2 = box
    color_h, color_w = color_shape[:2]
    depth_h, depth_w = depth_shape[:2]

    sx = depth_w / float(color_w)
    sy = depth_h / float(color_h)

    dx1 = max(0, min(depth_w - 1, int(round(x1 * sx))))
    dy1 = max(0, min(depth_h - 1, int(round(y1 * sy))))
    dx2 = max(0, min(depth_w - 1, int(round(x2 * sx))))
    dy2 = max(0, min(depth_h - 1, int(round(y2 * sy))))
    return [dx1, dy1, dx2, dy2]


def get_inner_box(box, scale=0.5):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale

    nx1 = int(round(cx - w / 2.0))
    ny1 = int(round(cy - h / 2.0))
    nx2 = int(round(cx + w / 2.0))
    ny2 = int(round(cy + h / 2.0))
    return [nx1, ny1, nx2, ny2]


def get_stable_depth_from_box(box, depth_img, color_shape):
    depth_box = scale_box_to_depth(box, color_shape, depth_img.shape)
    depth_box = get_inner_box(depth_box, INNER_BOX_SCALE)

    x1, y1, x2, y2 = depth_box
    if x2 <= x1 or y2 <= y1:
        return None, 0

    roi = depth_img[y1:y2, x1:x2]
    valid = roi[(roi > 0) & (roi < MAX_DEPTH_MM)]
    if len(valid) < MIN_VALID_DEPTH_COUNT:
        return None, len(valid)

    return int(np.median(valid)), len(valid)


def smooth_depth(track_key, depth_value):
    history = depth_history_map[track_key]
    history.append(depth_value)
    return int(np.median(history))


def make_depth_view(depth_img):
    valid_mask = (depth_img > 0) & (depth_img < DEPTH_VIS_MAX_MM)
    clipped_depth = np.clip(depth_img, 0, DEPTH_VIS_MAX_MM)
    norm_depth = cv2.convertScaleAbs(clipped_depth, alpha=255.0 / DEPTH_VIS_MAX_MM)
    depth_color = cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET)
    depth_color[~valid_mask] = (0, 0, 0)
    return depth_color


def load_orbbec_modules():
    if str(ORBBEC_SDK_DIR) not in sys.path:
        sys.path.append(str(ORBBEC_SDK_DIR))
    os.environ["PATH"] = str(ORBBEC_SDK_DIR) + os.pathsep + os.environ.get("PATH", "")

    from ObTypes import (
        OB_PY_ALIGN_D2C_SW_MODE,
        OB_PY_FORMAT_I420,
        OB_PY_FORMAT_MJPG,
        OB_PY_FORMAT_RGB888,
        OB_PY_FORMAT_UYVY,
        OB_PY_FORMAT_YUYV,
        OB_PY_PERMISSION_WRITE,
        OB_PY_PROP_COLOR_MIRROR_BOOL,
        OB_PY_PROP_DEPTH_MIRROR_BOOL,
        OB_PY_SENSOR_COLOR,
        OB_PY_SENSOR_DEPTH,
        OB_PY_STREAM_VIDEO,
    )
    import Pipeline
    from Error import ObException

    ob_types = {
        "OB_PY_ALIGN_D2C_SW_MODE": OB_PY_ALIGN_D2C_SW_MODE,
        "OB_PY_FORMAT_I420": OB_PY_FORMAT_I420,
        "OB_PY_FORMAT_MJPG": OB_PY_FORMAT_MJPG,
        "OB_PY_FORMAT_RGB888": OB_PY_FORMAT_RGB888,
        "OB_PY_FORMAT_UYVY": OB_PY_FORMAT_UYVY,
        "OB_PY_FORMAT_YUYV": OB_PY_FORMAT_YUYV,
        "OB_PY_PERMISSION_WRITE": OB_PY_PERMISSION_WRITE,
        "OB_PY_PROP_COLOR_MIRROR_BOOL": OB_PY_PROP_COLOR_MIRROR_BOOL,
        "OB_PY_PROP_DEPTH_MIRROR_BOOL": OB_PY_PROP_DEPTH_MIRROR_BOOL,
        "OB_PY_SENSOR_COLOR": OB_PY_SENSOR_COLOR,
        "OB_PY_SENSOR_DEPTH": OB_PY_SENSOR_DEPTH,
        "OB_PY_STREAM_VIDEO": OB_PY_STREAM_VIDEO,
    }
    return Pipeline, ObException, ob_types


def set_mirror_if_supported(device, prop, permission, enabled, name):
    if device.isPropertySupported(prop, permission):
        device.setBoolProperty(prop, enabled)
        print(f"{name} mirror set to {enabled}")


def load_orbbec_cpp_camera():
    if hasattr(os, "add_dll_directory") and ORBBEC_CPP_DLL_DIR.exists():
        os.add_dll_directory(str(ORBBEC_CPP_DLL_DIR))
    if hasattr(os, "add_dll_directory") and CAMERA_ROOT.exists():
        os.add_dll_directory(str(CAMERA_ROOT))
    for import_dir in (CAMERA_ROOT, SMART_CARE_ROOT, ROOT):
        if str(import_dir) not in sys.path:
            sys.path.append(str(import_dir))

    try:
        from orbbec_cpp_camera import OrbbecCppCamera
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import orbbec_cpp_camera. Build it first with: "
            "python smart_care/camera/setup_orbbec_cpp.py build_ext --inplace"
        ) from exc

    return OrbbecCppCamera


def load_vl_object_finder():
    if str(SMART_CARE_ROOT) not in sys.path:
        sys.path.append(str(SMART_CARE_ROOT))

    from vl.object_finder_legacy import create_vl_finder

    print(f"Preparing VL finder backend: {VL_BACKEND}. Heavy models may take time on first use...")
    return create_vl_finder(
        backend=VL_BACKEND,
        model_name=VL_MODEL_NAME,
        fallback_model_name=VL_FALLBACK_MODEL_NAME,
        device=DEVICE,
    )


def run_2d_mode():
    model, device, stride, names, pt, imgsz = load_model()
    personal_detector = load_active_personal_model() if USE_PERSONAL_MODEL else None
    cap = open_required_depth_camera()

    if not cap.isOpened():
        raise RuntimeError("Cannot open Orbbec depth camera.")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_objects(model, frame, device, pt, stride, names, imgsz, COCO_CONF_THRES, "coco")
        if personal_detector is not None:
            personal_detections = detect_objects(
                personal_detector["model"],
                frame,
                personal_detector["device"],
                personal_detector["pt"],
                personal_detector["stride"],
                personal_detector["names"],
                personal_detector["imgsz"],
                PERSONAL_CONF_THRES,
                "personal",
            )
            if personal_detections and frame_idx % BROADCAST_INTERVAL_FRAMES == 0:
                print(f"Personal detections: {[(d['class_name'], round(d['conf'], 3)) for d in personal_detections]}")
            detections.extend(personal_detections)
        detections = prioritize_personal_detections(detections)
        assign_support_relations(detections)
        assign_pair_relations(detections)

        sentences = []
        for det in detections:
            draw_detection(frame, det)
            sentence = build_object_sentence(det, frame.shape)
            if sentence:
                sentences.append(sentence)

        draw_text_panel(frame, sentences)

        if detections and frame_idx % BROADCAST_INTERVAL_FRAMES == 0:
            print(" | ".join(sentences[:3]))

        cv2.imshow("Object Locator 2D", frame)
        key = cv2.waitKey(1)
        if key == ESC or key == Q_KEY:
            cv2.destroyAllWindows()
            break

        frame_idx += 1

    cap.release()


def run_depth_mode():
    model, device, stride, names, pt, imgsz = load_model()
    personal_detector = load_active_personal_model() if USE_PERSONAL_MODEL else None
    OrbbecCppCamera = load_orbbec_cpp_camera()
    vl_finder = load_vl_object_finder() if ENABLE_VL_OBJECT_FINDER else None
    vl_query_text = VL_INITIAL_QUERY.strip()
    vl_result = None
    vl_last_run_frame = -VL_FIND_INTERVAL_FRAMES
    vl_error = ""

    printed_frame_info = False
    frame_idx = 0

    camera = OrbbecCppCamera(align_depth_to_color=True, mirror=False)
    install_orbbec_endpoint_log_filter()
    print("Opening Orbbec depth camera...")
    camera.start()
    print(f"Orbbec C++ camera info: {camera.info()}")

    try:
        while True:
            frame_set = camera.read(100)
            if frame_set is None:
                continue

            color_img = frame_set["color"]
            depth_img = frame_set["depth"]
            depth_view = make_depth_view(depth_img)

            if not printed_frame_info:
                print(f"Color image size: {color_img.shape[1]}x{color_img.shape[0]}")
                print(f"Depth image size: {depth_img.shape[1]}x{depth_img.shape[0]}")
                if ENABLE_VL_OBJECT_FINDER:
                    model_text = VL_MODEL_NAME if VL_BACKEND.lower() == "qwen" else "no model loaded"
                    print(f"VL finder: backend={VL_BACKEND}, model={model_text}")
                    print("VL finder: press V in the image window, then type a target such as 灰色鼠标.")
                printed_frame_info = True

            if vl_finder is not None and vl_query_text:
                should_run_vl = frame_idx - vl_last_run_frame >= VL_FIND_INTERVAL_FRAMES
                if should_run_vl:
                    try:
                        from vl.object_finder_legacy import build_vl_depth_result, parse_user_query

                        parsed_query = parse_user_query(vl_query_text)
                        candidates = vl_finder.find_target_box(color_img, parsed_query)
                        if candidates:
                            vl_result = build_vl_depth_result(
                                parsed_query,
                                candidates[0],
                                depth_img,
                                color_img.shape,
                                get_stable_depth_from_box,
                                describe_direction,
                            )
                            vl_error = ""
                            print(vl_result.description)
                        else:
                            vl_result = None
                            vl_error = "VL target not found"
                            print(vl_error)
                    except Exception as e:
                        vl_result = None
                        vl_error = f"VL finder failed: {e}"
                        print(vl_error)
                    vl_last_run_frame = frame_idx

            detections = detect_objects(model, color_img, device, pt, stride, names, imgsz, COCO_CONF_THRES, "coco")
            if personal_detector is not None:
                personal_detections = detect_objects(
                    personal_detector["model"],
                    color_img,
                    personal_detector["device"],
                    personal_detector["pt"],
                    personal_detector["stride"],
                    personal_detector["names"],
                    personal_detector["imgsz"],
                    PERSONAL_CONF_THRES,
                    "personal",
                )
                if personal_detections and frame_idx % BROADCAST_INTERVAL_FRAMES == 0:
                    print(f"Personal detections: {[(d['class_name'], round(d['conf'], 3)) for d in personal_detections]}")
                detections.extend(personal_detections)
            detections = prioritize_personal_detections(detections)

            for det in detections:
                depth_mm, valid_count = get_stable_depth_from_box(det["box"], depth_img, color_img.shape)
                if depth_mm is not None:
                    track_key = f"{det['class_name']}_{det['class_id']}"
                    det["depth_mm"] = smooth_depth(track_key, depth_mm)
                det["valid_count"] = valid_count

            assign_support_relations(detections)
            assign_pair_relations(detections)

            sentences = []
            for det in detections:
                draw_detection(color_img, det)
                sentence = build_object_sentence(det, color_img.shape)
                if sentence:
                    sentences.append(sentence)

                depth_box = scale_box_to_depth(det["box"], color_img.shape, depth_img.shape)
                dx1, dy1, dx2, dy2 = depth_box
                cv2.rectangle(depth_view, (dx1, dy1), (dx2, dy2), (255, 255, 255), 2)

            draw_text_panel(color_img, sentences)
            draw_vl_result(color_img, vl_result)
            if vl_query_text:
                status = "VL target set"
                cv2.putText(color_img, status, (20, color_img.shape[0] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
            if vl_result is not None:
                depth_text = "VL found target"
                if vl_result.depth_m is not None:
                    depth_text += f", {vl_result.depth_m:.2f}m from camera"
                cv2.putText(color_img, depth_text, (20, color_img.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
            elif vl_error:
                cv2.putText(color_img, "VL finder error or target not found", (20, color_img.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            if detections and frame_idx % BROADCAST_INTERVAL_FRAMES == 0:
                print(" | ".join(sentences[:3]))

            cv2.imshow("Object Locator", color_img)
            cv2.imshow("Depth View", depth_view)

            key = cv2.waitKey(1)
            if key == ESC or key == Q_KEY:
                cv2.destroyAllWindows()
                break
            if key == V_KEY and vl_finder is not None:
                print("Enter target description, e.g. 灰色鼠标: ", end="", file=sys.stderr, flush=True)
                new_query = input().strip()
                if new_query:
                    vl_query_text = new_query
                    vl_result = None
                    vl_error = ""
                    vl_last_run_frame = frame_idx - VL_FIND_INTERVAL_FRAMES

            frame_idx += 1
    finally:
        camera.stop()


def main():
    if USE_DEPTH_CAMERA:
        run_depth_mode()
    else:
        run_2d_mode()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Run failed:", e)
        if USE_DEPTH_CAMERA:
            print("Depth mode uses the Orbbec C++ pybind module.")
            print("Build it with: python smart_care/camera/setup_orbbec_cpp.py build_ext --inplace")
