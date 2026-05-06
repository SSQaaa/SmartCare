import cv2
import numpy as np
import onnxruntime as ort

# ===================== 参数 =====================
MODEL_PATH = r"E:\SSQ\Sophomore\zqzb\rgzndl\yolov5\runs\train\exp6\weights\best.onnx"
IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45

CLASS_NAMES = ['fall','stand','sit','squat','run']


# ===================== 预处理 =====================
def letterbox(img, new_shape=640, color=(114, 114, 114)):
    shape = img.shape[:2]  # h, w
    ratio = min(new_shape / shape[0], new_shape / shape[1])

    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw /= 2
    dh /= 2

    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)

    return img_padded, ratio, (dw, dh)


def preprocess(img):
    img, ratio, (dw, dh) = letterbox(img, IMG_SIZE)

    img = img[:, :, ::-1]  # BGR → RGB
    img = img.transpose(2, 0, 1)  # HWC → CHW
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0

    return np.expand_dims(img, 0), ratio, (dw, dh)


# ===================== NMS（纯 numpy） =====================
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


# ===================== 后处理 =====================
def postprocess(pred, ratio, dwdh):
    pred = pred[0]  # (25200, 13)

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

    # xywh → xyxy
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    # 反 letterbox
    dw, dh = dwdh
    boxes_xyxy[:, [0, 2]] -= dw
    boxes_xyxy[:, [1, 3]] -= dh
    boxes_xyxy /= ratio

    # NMS（按类别）
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


# ===================== 主推理 =====================
def infer_video():
    print(f"Loading static fall ONNX model: {MODEL_PATH}")
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    print("Static fall model ready.")

    cap = cv2.VideoCapture(r"E:\SSQ\Sophomore\zqzb\rgzndl\myyolov8\test.mp4")  # 摄像头 / 或视频路径

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor, ratio, dwdh = preprocess(frame)
        outputs = session.run(None, {"images": input_tensor})
        pred = outputs[0]

        results = postprocess(pred, ratio, dwdh)

        for r in results:
            x1, y1, x2, y2 = r["box"].astype(int)
            cls = r["class_id"]
            score = r["score"]

            label = f"{CLASS_NAMES[cls]} {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("result", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    infer_video()
