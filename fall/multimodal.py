import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from collections import deque
from fall.features import PerFrameFeature
import joblib

# ===================== 参数 =====================
STATIC_MODEL_PATH = r"E:\SSQ\Sophomore\zqzb\rgzndl\yolov5\runs\train\exp6\weights\best.onnx"
POSE_MODEL_PATH = r"E:\SSQ\Sophomore\zqzb\rgzndl\myyolov8\myproject\yolov8n-pose.pt"
DYNAMIC_MODEL_PATH = r"E:\SSQ\Sophomore\zqzb\rgzndl\myyolov8\myproject\scripts\xgb_fall_detector.pkl"
VIDEO_PATH = r"E:\SSQ\Sophomore\zqzb\rgzndl\mypose\test.mp4"
OUT_VIDEO_PATH = "fall_multimodal_result.mp4"

IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
WINDOW_SIZE = 20
STATIC_CLEAR_SECONDS = 2.0


CLASS_NAMES = ['fall', 'stand', 'sit', 'squat', 'run']

SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]


# ===================== 静态分支预处理 =====================
def letterbox(img, new_shape=640, color=(114, 114, 114)):
    shape = img.shape[:2]
    ratio = min(new_shape / shape[0], new_shape / shape[1])

    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw /= 2
    dh /= 2

    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img_padded = cv2.copyMakeBorder(
        img_resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return img_padded, ratio, (dw, dh)


def preprocess(img):
    img, ratio, (dw, dh) = letterbox(img, IMG_SIZE)

    img = img[:, :, ::-1]
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0

    return np.expand_dims(img, 0), ratio, (dw, dh)


def nms(boxes, scores, iou_thres):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return keep


def postprocess(pred, ratio, dwdh):
    pred = pred[0]

    boxes = pred[:, :4]
    obj_conf = pred[:, 4]
    class_scores = pred[:, 5:]

    class_ids = np.argmax(class_scores, axis=1)
    class_conf = class_scores[np.arange(len(class_scores)), class_ids]
    scores = obj_conf * class_conf

    mask = scores > CONF_THRES
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    dw, dh = dwdh
    boxes_xyxy[:, [0, 2]] -= dw
    boxes_xyxy[:, [1, 3]] -= dh
    boxes_xyxy /= ratio

    results = []
    for cls in np.unique(class_ids):
        idxs = np.where(class_ids == cls)[0]
        keep = nms(boxes_xyxy[idxs], scores[idxs], IOU_THRES)

        for i in keep:
            results.append({
                "box": boxes_xyxy[idxs][i],
                "score": scores[idxs][i],
                "class_id": int(cls)
            })

    return results


def get_static_result(results):
    if not results:
        return "none", 0.0

    best_result = max(results, key=lambda x: x["score"])
    best_label = CLASS_NAMES[best_result["class_id"]]

    fall_scores = [r["score"] for r in results if r["class_id"] == 0]
    fall_score = max(fall_scores) if fall_scores else 0.0

    return best_label, float(fall_score)


# ===================== 动态分支 =====================
def draw_pose(frame, kpts_xy, kpts_conf, conf_thr=0.3):
    for i, (x, y) in enumerate(kpts_xy):
        if kpts_conf[i] > conf_thr:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

    for i, j in SKELETON:
        if kpts_conf[i] > conf_thr and kpts_conf[j] > conf_thr:
            x1, y1 = kpts_xy[i]
            x2, y2 = kpts_xy[j]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)


def get_dynamic_score(model, window_input):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(window_input)[0]
        if len(proba) > 1:
            return float(proba[1])
        return float(proba[0])

    pred = model.predict(window_input)[0]
    return float(pred)


def get_pose_result(results, frame):
    if len(results.boxes) == 0:
        raw_kpts = [None] * 17
        bbox = [0, 0, 1, 1]
        return raw_kpts, bbox

    bbox_xywh = results.boxes.xywh[0].cpu().numpy()
    bbox = bbox_xywh.tolist()
    kpts_xy = results.keypoints.xy[0].cpu().numpy()
    kpts_conf = results.keypoints.conf[0].cpu().numpy()
    raw_kpts = [(float(x), float(y), float(c)) for (x, y), c in zip(kpts_xy, kpts_conf)]

    draw_pose(frame, kpts_xy, kpts_conf)
    return raw_kpts, bbox



# ===================== 主推理 =====================
def infer_video():
    print(f"Loading static fall ONNX model: {STATIC_MODEL_PATH}")
    static_session = ort.InferenceSession(STATIC_MODEL_PATH, providers=['CPUExecutionProvider'])
    print(f"Loading YOLO pose model: {POSE_MODEL_PATH}")
    pose_model = YOLO(POSE_MODEL_PATH)
    print(f"Loading dynamic fall model: {DYNAMIC_MODEL_PATH}")
    dynamic_model = joblib.load(DYNAMIC_MODEL_PATH)
    print("Multimodal fall models ready.")

    feat_window = deque(maxlen=WINDOW_SIZE)
    pf = None
    flag = 0
    static_miss_frames = 0

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    static_clear_frames = max(1, int(fps * STATIC_CLEAR_SECONDS))

    writer = cv2.VideoWriter(
        OUT_VIDEO_PATH,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if pf is None:
            pf = PerFrameFeature(frame.shape[1], frame.shape[0], fps=25.0, conf_thr=0.2)

        input_tensor, ratio, dwdh = preprocess(frame)
        static_outputs = static_session.run(None, {"images": input_tensor})
        static_results = postprocess(static_outputs[0], ratio, dwdh)
        static_label, static_score = get_static_result(static_results)

        pose_results = pose_model(frame, verbose=False)[0]
        raw_kpts, bbox = get_pose_result(pose_results, frame)
        feat = pf.compute_frame_feat(raw_kpts, bbox)
        feat_window.append(feat)

        dynamic_score = 0.0
        dynamic_pred = 0
        if len(feat_window) == WINDOW_SIZE:
            window_input = np.array(feat_window).flatten().reshape(1, -1)
            dynamic_score = get_dynamic_score(dynamic_model, window_input)
            dynamic_pred = 1 if dynamic_score >= 0.5 else 0

        static_pred = 1 if static_label == "fall" else 0
        if dynamic_pred == 1:
            flag = 1
            static_miss_frames = 0

        if flag == 1:
            if static_pred == 1:
                result = "FALL"
                static_miss_frames = 0
            else:
                result = "SUSPECT"
                static_miss_frames += 1
                if static_miss_frames >= static_clear_frames:
                    flag = 0
                    static_miss_frames = 0
                    result = "SAFE"
        else:
            result = "SAFE"

        for r in static_results:
            x1, y1, x2, y2 = r["box"].astype(int)
            cls = r["class_id"]
            score = r["score"]
            label = f"{CLASS_NAMES[cls]} {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if result == "FALL":
            result_color = (0, 0, 255)
        elif result == "SUSPECT":
            result_color = (0, 165, 255)
        else:
            result_color = (0, 255, 0)
        cv2.putText(frame, f"Result: {result}", (30, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 2)
        cv2.putText(frame, f"Static: {static_label} {static_score:.2f}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"Dynamic: {dynamic_pred} {dynamic_score:.2f}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        writer.write(frame)
        cv2.imshow("Multimodal Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    infer_video()
