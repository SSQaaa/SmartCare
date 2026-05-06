import os
import cv2
import json
import csv
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ----------------- 配置 -----------------
VIDEOS_JSON_DIR = Path("../kpt_raw")       # YOLOv8 keypoints JSON 文件夹
GT_DIR = Path("../annocations")       # Le2i 标注 txt 文件夹
OUT_DIR = Path("../features_csv")          # 输出 CSV 文件夹
POSE_MODEL = "../yolov8n-pose.pt"          # YOLOv8 pose 模型路径
CONF_THR = 0.2
FPS = 25.0
DT = 1.0 / FPS
OUT_DIR.mkdir(exist_ok=True)
# ----------------------------------------

model = YOLO(POSE_MODEL)

# ----------------- Le2i 解析 -----------------
def read_le2i_gt(gt_path):
    """读取 Le2i 标注，返回 fall_start, fall_end, per-frame bbox"""
    if not gt_path.exists():
        return None, None, []
    with open(gt_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    # 前两行是 fall start/end
    try:
        fall_start = int(lines[0].split(',')[0])
        fall_end = int(lines[1].split(',')[0])
    except:
        fall_start, fall_end = 0, 0
    # 之后每行 bbox: h,w,cx,cy等，根据逗号分割
    bboxes = []
    for l in lines[2:]:
        vals = list(map(float, l.split(',')))
        if len(vals) < 4:
            continue
        h, w, cx, cy = vals[-4:]
        bboxes.append([cx, cy, w, h])
    return fall_start, fall_end, bboxes

# ----------------- 特征计算 -----------------
class PerFrameFeature:
    def __init__(self, frame_w, frame_h, fps=25.0, conf_thr=0.2):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.prev_center_y = None
        self.prev_tilt = None
        self.FPS = fps
        self.DT = 1.0 / fps
        self.CONF_THR = conf_thr

    def safe_kp(self, kp):
        if kp is None: return None
        x,y,c = float(kp[0]), float(kp[1]), float(kp[2])
        if c < self.CONF_THR:
            return None
        return (x,y,c)

    def compute_tilt(self, kpts):
        try:
            l_sh, r_sh = kpts[5], kpts[6]
            l_hip, r_hip = kpts[11], kpts[12]
            if None in (l_sh, r_sh, l_hip, r_hip):
                return np.nan
            sh_c = ((l_sh[0]+r_sh[0])/2, (l_sh[1]+r_sh[1])/2)
            hip_c = ((l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2)
            dx = hip_c[0] - sh_c[0]
            dy = hip_c[1] - sh_c[1]
            return float(np.degrees(np.arctan2(dy, dx)))
        except:
            return np.nan

    def compute_center_y(self, kpts):
        ys = [kp[1] for kp in kpts if kp is not None]
        if not ys: return np.nan
        return float(np.mean(ys) / (self.frame_h + 1e-6))

    def compute_hw(self, bbox_xywh):
        w, h = bbox_xywh[2], bbox_xywh[3]
        if w <= 0: return np.nan
        return float(h / (w + 1e-6))

    def compute_mean_conf(self, kpts):
        confs = [kp[2] for kp in kpts if kp is not None]
        if not confs: return 0.0
        return float(np.mean(confs))

    def compute_head_ankle(self, kpts):
        head = kpts[0]
        ankles = []
        if len(kpts) > 15 and kpts[15] is not None: ankles.append(kpts[15])
        if len(kpts) > 16 and kpts[16] is not None: ankles.append(kpts[16])
        if head is None or not ankles:
            return np.nan
        head_y = head[1]
        ankle_y = np.mean([a[1] for a in ankles])
        return float((ankle_y - head_y) / (self.frame_h + 1e-6))

    def compute_frame_feat(self, raw_kpts, bbox_xywh):
        if raw_kpts is None:
            raw_kpts = [None]*17
        kpts = [self.safe_kp(k) for k in raw_kpts]

        tilt = self.compute_tilt(kpts)
        center_y = self.compute_center_y(kpts)
        hw = self.compute_hw(bbox_xywh)
        mean_conf = self.compute_mean_conf(kpts)
        head_ankle = self.compute_head_ankle(kpts)

        # 速度
        center_speed = 0.0 if self.prev_center_y is None or np.isnan(center_y) else (center_y - self.prev_center_y) / self.DT
        delta = 0.0
        if self.prev_tilt is not None and not np.isnan(tilt):
            delta = tilt - self.prev_tilt
            if delta > 180: delta -= 360
            if delta < -180: delta += 360
        tilt_speed = delta / self.DT

        if not np.isnan(center_y): self.prev_center_y = center_y
        if not np.isnan(tilt): self.prev_tilt = tilt

        return [
            float(tilt) if not np.isnan(tilt) else 0.0,
            float(center_y) if not np.isnan(center_y) else 0.0,
            float(hw) if not np.isnan(hw) else 0.0,
            float(mean_conf),
            float(center_speed),
            float(tilt_speed),
            float(head_ankle) if not np.isnan(head_ankle) else 0.0
        ]

# ----------------- 主处理 -----------------
def process_video(json_file, gt_file, out_dir):
    out_csv = out_dir / (json_file.stem + ".csv")
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    fall_start, fall_end, bboxes_gt = read_le2i_gt(gt_file)

    frame_h = 240  # 可以用实际视频尺寸替换
    frame_w = 320

    pf = PerFrameFeature(frame_w, frame_h, fps=FPS, conf_thr=CONF_THR)
    rows = []

    for i, frame in enumerate(data["frames"]):
        raw_kpts = frame["keypoints"]
        bbox_xywh = frame["bbox"] if frame["bbox"] is not None else [0,0,1,1]

        feat7 = pf.compute_frame_feat(raw_kpts, bbox_xywh)

        # label
        if fall_start is None or fall_end is None:
            label = 0
        else:
            label = 1 if fall_start <= i <= fall_end else 0

        rows.append(feat7 + [label])

    # 保存 CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tilt","center_y","hw","mean_conf","center_speed","tilt_speed","head_ankle","label"])
        writer.writerows(rows)

    print(f"[OK] Saved {out_csv}")


def main():
    json_files = list(VIDEOS_JSON_DIR.glob("*.json"))
    if not json_files:
        print("[ERROR] No JSON files found!")
        return

    for jf in json_files:
        gt_file = GT_DIR / (jf.stem + ".txt")
        if not gt_file.exists():
            print(f"[WARN] No GT file for {jf.name}, will use label=0")
        process_video(jf, gt_file, OUT_DIR)

if __name__ == "__main__":
    main()
