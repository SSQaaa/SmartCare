import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from fall.features import PerFrameFeature
import joblib

WINDOW_SIZE = 20
POSE_MODEL = r"E:\SSQ\Sophomore\zqzb\rgzndl\myyolov8\myproject\yolov8n-pose.pt"
XGB_MODEL = r"E:\SSQ\Sophomore\zqzb\rgzndl\myyolov8\myproject\scripts\xgb_fall_detector.pkl"

print(f"Loading YOLO pose model: {POSE_MODEL}")
pose_model = YOLO(POSE_MODEL)
print(f"Loading dynamic fall model: {XGB_MODEL}")
xgb_model = joblib.load(XGB_MODEL)
print("Dynamic fall models ready.")
feat_window = deque(maxlen=WINDOW_SIZE)

pf = None  # 后面初始化 PerFrameFeature

cap = cv2.VideoCapture(r"C:\Users\ssqaaa\Downloads\老人日常动作视频生成.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    "fall_dynamic_result.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# COCO skeleton
SKELETON = [
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (5, 6),              # shoulders
    (5, 11), (6, 12),    # torso
    (11, 12),
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16)   # right leg
]

def draw_pose(frame, kpts_xy, kpts_conf, conf_thr=0.3):
    # 画关键点
    for i, (x, y) in enumerate(kpts_xy):
        if kpts_conf[i] > conf_thr:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

    # 画骨架连线
    for i, j in SKELETON:
        if kpts_conf[i] > conf_thr and kpts_conf[j] > conf_thr:
            x1, y1 = kpts_xy[i]
            x2, y2 = kpts_xy[j]
            cv2.line(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 255),
                2
            )



while True:
    ret, frame = cap.read()
    if not ret: break

    if pf is None:
        # 初始化帧尺寸
        pf = PerFrameFeature(frame.shape[1], frame.shape[0], fps=25.0, conf_thr=0.2)

    results = pose_model(frame, verbose=False)[0]

    if len(results.boxes) == 0:
        raw_kpts = [None]*17
        bbox = [0,0,1,1]
    else:
        bbox_xywh = results.boxes.xywh[0].cpu().numpy()
        bbox = bbox_xywh.tolist()
        kpts_xy = results.keypoints.xy[0].cpu().numpy()
        kpts_conf = results.keypoints.conf[0].cpu().numpy()
        raw_kpts = [(float(x), float(y), float(c)) for (x,y),c in zip(kpts_xy, kpts_conf)]

        draw_pose(frame, kpts_xy, kpts_conf)


    feat = pf.compute_frame_feat(raw_kpts, bbox)
    feat_window.append(feat)

    if len(feat_window) == WINDOW_SIZE:
        window_input = np.array(feat_window).flatten().reshape(1,-1)
        pred = xgb_model.predict(window_input)[0]
        cv2.putText(frame, f"Fall: {pred}", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255),8)

    writer.write(frame)
    cv2.imshow("Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
