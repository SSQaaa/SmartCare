import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SMART_CARE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SMART_CARE_ROOT.parent


@dataclass
class ActionResult:
    success: bool
    spoken_text: str
    debug_text: str = ""


class VisionActions:
    def __init__(self, vl_backend="qwen", vl_model="", vl_fallback_model="", fall_mode="multimodal"):
        self.vl_backend = vl_backend
        self.vl_model = vl_model
        self.vl_fallback_model = vl_fallback_model
        self.fall_mode = fall_mode
        self.current_process = None
        self.current_process_name = ""
        self._vl_assistant = None
        self._vl_finder = None
        self._face_app = None

    def close(self):
        self.stop_current()

    def stop_current(self):
        if self.current_process is None:
            return ActionResult(False, "当前没有正在运行的长时间功能。")
        name = self.current_process_name or "当前功能"
        self.current_process.terminate()
        try:
            self.current_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.current_process.kill()
        self.current_process = None
        self.current_process_name = ""
        return ActionResult(True, f"已停止{name}。")

    def start_fall_monitor(self):
        self.stop_current()
        cmd = [sys.executable, str(SMART_CARE_ROOT / "main_fall.py"), self.fall_mode]
        self.current_process = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
        self.current_process_name = "跌倒检测"
        return ActionResult(True, f"已启动{self.fall_mode}跌倒检测。")

    def stop_fall_monitor(self):
        if self.current_process_name != "跌倒检测":
            return ActionResult(False, "跌倒检测当前没有运行。")
        return self.stop_current()

    def fall_status(self):
        if self.current_process is None or self.current_process_name != "跌倒检测":
            return ActionResult(True, "跌倒检测当前没有运行。")
        if self.current_process.poll() is None:
            return ActionResult(True, "跌倒检测正在运行。")
        code = self.current_process.returncode
        self.current_process = None
        self.current_process_name = ""
        return ActionResult(False, f"跌倒检测已经退出，退出码是 {code}。")

    def read_color_frame(self, retries=30, delay=0.1):
        from camera.depth_camera import open_required_depth_camera

        cap = open_required_depth_camera()
        try:
            if not cap.isOpened():
                raise RuntimeError("相机没有打开。")
            for _ in range(retries):
                ok, frame = cap.read()
                if ok:
                    return frame
                time.sleep(delay)
            raise RuntimeError("相机已打开，但没有读到彩色画面。")
        finally:
            cap.release()

    def read_rgbd_frame(self, retries=30, timeout_ms=200):
        from camera.depth_camera import load_orbbec_cpp_camera

        OrbbecCppCamera = load_orbbec_cpp_camera()
        camera = OrbbecCppCamera(align_depth_to_color=True, mirror=False)
        camera.start()
        try:
            for _ in range(retries):
                frame_set = camera.read(timeout_ms)
                if frame_set is not None and "color" in frame_set and "depth" in frame_set:
                    return frame_set["color"], frame_set["depth"]
            raise RuntimeError("相机已打开，但没有读到同步的彩色和深度画面。")
        finally:
            camera.stop()

    @staticmethod
    def describe_direction(box, image_shape):
        x1, _y1, x2, _y2 = box
        center_x = (x1 + x2) / 2.0
        width = image_shape[1]
        if center_x < width / 3:
            return "左前方"
        if center_x > width * 2 / 3:
            return "右前方"
        return "正前方"

    @staticmethod
    def get_stable_depth_from_box(box, depth_img, color_shape):
        x1, y1, x2, y2 = box
        color_h, color_w = color_shape[:2]
        depth_h, depth_w = depth_img.shape[:2]
        sx = depth_w / float(color_w)
        sy = depth_h / float(color_h)
        dx1 = max(0, min(depth_w - 1, int(round(x1 * sx))))
        dy1 = max(0, min(depth_h - 1, int(round(y1 * sy))))
        dx2 = max(0, min(depth_w - 1, int(round(x2 * sx))))
        dy2 = max(0, min(depth_h - 1, int(round(y2 * sy))))
        if dx2 <= dx1 or dy2 <= dy1:
            return None, 0

        cx = (dx1 + dx2) / 2.0
        cy = (dy1 + dy2) / 2.0
        w = (dx2 - dx1) * 0.5
        h = (dy2 - dy1) * 0.5
        ix1 = max(0, int(round(cx - w / 2.0)))
        iy1 = max(0, int(round(cy - h / 2.0)))
        ix2 = min(depth_w - 1, int(round(cx + w / 2.0)))
        iy2 = min(depth_h - 1, int(round(cy + h / 2.0)))
        roi = depth_img[iy1:iy2, ix1:ix2]
        valid = roi[(roi > 0) & (roi < 5000)]
        if len(valid) < 20:
            return None, len(valid)
        return int(np.median(valid)), len(valid)

    def _load_vl_assistant(self):
        if self._vl_assistant is None:
            from config import existing_model_path
            from vl import create_vl_assistant

            self._vl_assistant = create_vl_assistant(
                backend=self.vl_backend,
                model_name=existing_model_path(self.vl_model) if self.vl_model else "",
                fallback_model_name=existing_model_path(self.vl_fallback_model) if self.vl_fallback_model else "",
            )
        return self._vl_assistant

    def _load_vl_finder(self):
        if self._vl_finder is None:
            from config import existing_model_path
            from vl.object_finder_legacy import create_vl_finder

            self._vl_finder = create_vl_finder(
                backend=self.vl_backend,
                model_name=existing_model_path(self.vl_model) if self.vl_model else "",
                fallback_model_name=existing_model_path(self.vl_fallback_model) if self.vl_fallback_model else "",
            )
        return self._vl_finder

    def describe_scene(self):
        from vl.types import VLRequest

        frame = self.read_color_frame()
        assistant = self._load_vl_assistant()
        response = assistant.run(
            frame,
            VLRequest(intent="describe_scene", response_mode="answer", max_new_tokens=80),
        )
        text = response.answer.strip() if response.answer else "我没有得到画面描述。"
        return ActionResult(bool(response.answer), text, response.raw_text)

    def find_object(self, target):
        target = str(target or "").strip()
        if not target:
            return ActionResult(False, "你想找什么？请说，比如帮我找杯子。")

        from object import locator
        color_img, depth_img = self.read_rgbd_frame()

        query_texts = {target.lower()}
        try:
            from vl.object_finder_legacy import parse_user_query

            query = parse_user_query(target)
            query_texts.update(
                value.lower()
                for value in (query.raw_text, query.target_name, query.object_name, query.grounding_prompt)
                if value
            )
        except Exception:
            query = None

        def matches_detection(det):
            names = {
                str(det.get("class_name", "")).lower(),
                str(locator.get_cn_name(det.get("class_name", ""))).lower(),
            }
            return any(
                query_text == name or query_text in name or name in query_text
                for query_text in query_texts
                for name in names
                if query_text and name
            )

        detections = []
        try:
            model, device, stride, names, pt, imgsz = locator.load_model()
            detections.extend(locator.detect_objects(
                model,
                color_img,
                device,
                pt,
                stride,
                names,
                imgsz,
                locator.COCO_CONF_THRES,
                "coco",
            ))
        except Exception as exc:
            print(f"YOLO 常见物体检测失败：{exc}")

        if locator.USE_PERSONAL_MODEL:
            personal_detector = locator.load_active_personal_model()
            if personal_detector is not None:
                detections.extend(locator.detect_objects(
                    personal_detector["model"],
                    color_img,
                    personal_detector["device"],
                    personal_detector["pt"],
                    personal_detector["stride"],
                    personal_detector["names"],
                    personal_detector["imgsz"],
                    locator.PERSONAL_CONF_THRES,
                    "personal",
                ))

        detections = locator.prioritize_personal_detections(detections)
        for det in detections:
            depth_mm, valid_count = locator.get_stable_depth_from_box(det["box"], depth_img, color_img.shape)
            det["depth_mm"] = depth_mm
            det["valid_count"] = valid_count

        for det in detections:
            if matches_detection(det):
                sentence = locator.build_object_sentence(det, color_img.shape)
                return ActionResult(True, sentence or f"我找到了{target}。", f"yolo {det['class_name']} {det['box']}")

        from vl.object_finder_legacy import build_vl_depth_result, parse_user_query
        query = query or parse_user_query(target)
        finder = self._load_vl_finder()
        candidates = finder.find_target_box(color_img, query)
        if not candidates:
            return ActionResult(False, f"没有找到{target}。")
        result = build_vl_depth_result(
            query,
            candidates[0],
            depth_img,
            color_img.shape,
            self.get_stable_depth_from_box,
            self.describe_direction,
        )
        return ActionResult(True, result.description, f"{result.source} {result.label} {result.box}")

    def summarize_objects(self, max_items=5):
        from object import locator

        color_img, depth_img = self.read_rgbd_frame()
        model, device, stride, names, pt, imgsz = locator.load_model()
        detections = locator.detect_objects(
            model,
            color_img,
            device,
            pt,
            stride,
            names,
            imgsz,
            locator.COCO_CONF_THRES,
            "coco",
        )
        personal_detector = locator.load_active_personal_model() if locator.USE_PERSONAL_MODEL else None
        if personal_detector is not None:
            detections.extend(locator.detect_objects(
                personal_detector["model"],
                color_img,
                personal_detector["device"],
                personal_detector["pt"],
                personal_detector["stride"],
                personal_detector["names"],
                personal_detector["imgsz"],
                locator.PERSONAL_CONF_THRES,
                "personal",
            ))
        detections = locator.prioritize_personal_detections(detections)

        for det in detections:
            depth_mm, valid_count = locator.get_stable_depth_from_box(det["box"], depth_img, color_img.shape)
            det["depth_mm"] = depth_mm
            det["valid_count"] = valid_count

        locator.assign_support_relations(detections)
        locator.assign_pair_relations(detections)
        sentences = [
            locator.build_object_sentence(det, color_img.shape)
            for det in detections
        ]
        sentences = [sentence for sentence in sentences if sentence]
        if not sentences:
            return ActionResult(False, "我没有检测到明显的常见物品。")
        return ActionResult(True, "；".join(sentences[:max_items]) + "。", f"objects={len(sentences)}")

    def recognize_face(self):
        from face import recognition

        model_path = recognition.DEFAULT_MODEL_PATH
        if not model_path.exists():
            return ActionResult(False, "人脸库还没有训练。请先录入并训练人脸库。")

        vectors, labels, metadata = recognition.load_model(model_path)
        app = self._face_app
        if app is None:
            app = recognition.load_face_app(
                model_name=metadata.get("model_name", recognition.INSIGHTFACE_MODEL_NAME),
                det_size=recognition.DET_SIZE,
            )
            self._face_app = app

        frame = self.read_color_frame()
        faces = recognition.detect_faces(app, frame)
        if not faces:
            return ActionResult(False, "画面里没有检测到人脸。")
        if len(faces) > 1:
            return ActionResult(False, f"画面里检测到 {len(faces)} 张人脸，请一次只识别一个人。")

        threshold = float(metadata.get("threshold", recognition.DEFAULT_THRESHOLD))
        label, distance, similarity = recognition.predict_embedding(faces[0].embedding, vectors, labels, threshold)
        if label == "unknown":
            return ActionResult(False, f"没有识别出这个人，相似度 {similarity:.2f}。")
        name = metadata.get("display_names", {}).get(label, label)
        return ActionResult(True, f"这是{name}，相似度 {similarity:.2f}。", f"dist={distance:.3f}")
