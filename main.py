import argparse
import atexit
import io
import os
import queue
import re
import sys
import threading
import time
import warnings
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from config import VL_BACKEND, VL_FALLBACK_MODEL_PATH, VL_MODEL_PATH, existing_model_path
from voice.sherpa_asr import SherpaOnnxRecognizer, VoiceDependencyError
from voice.commands import HELP_TEXT, is_safety_confirmation, parse_command
from voice.tts import DEFAULT_TTS_MODEL_DIR, SpeechSpeaker

from voice.sherpa_asr import DEFAULT_MODEL_DIR


SMART_CARE_ROOT = Path(__file__).resolve().parent

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"insightface\.utils\.face_align",
    message=r".*estimate.*deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*`estimate` is deprecated.*",
)

_endpoint_filter_installed = False
_endpoint_restore_fd = None
_endpoint_original_stdout = None


def install_endpoint_log_filter():
    global _endpoint_filter_installed, _endpoint_restore_fd, _endpoint_original_stdout

    if _endpoint_filter_installed or not hasattr(os, "dup2"):
        return

    try:
        _endpoint_original_stdout = sys.stdout
        original_stdout_fd = os.dup(1)
        _endpoint_restore_fd = os.dup(1)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 1)
        os.close(write_fd)
        stdout_encoding = getattr(_endpoint_original_stdout, "encoding", None) or "utf-8"
        sys.stdout = io.TextIOWrapper(
            os.fdopen(os.dup(1), "wb", buffering=0),
            encoding=stdout_encoding,
            errors="replace",
            line_buffering=True,
            write_through=True,
        )
    except OSError:
        return

    _endpoint_filter_installed = True
    endpoint_pattern = re.compile(rb"Endpoint 0x[0-9A-Fa-f]+ bandwidth:")

    def restore_stdout():
        global _endpoint_restore_fd, _endpoint_original_stdout
        if _endpoint_restore_fd is None:
            return
        try:
            try:
                sys.stdout.flush()
            except Exception:
                pass
            os.dup2(_endpoint_restore_fd, 1)
            os.close(_endpoint_restore_fd)
            if _endpoint_original_stdout is not None:
                sys.stdout = _endpoint_original_stdout
        except OSError:
            pass
        _endpoint_restore_fd = None
        _endpoint_original_stdout = None

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
    threading.Thread(target=pump_stdout, name="endpoint-log-filter", daemon=True).start()


class SmartCareMainApp:
    def __init__(self, args):
        self.args = args
        self.speaker = SpeechSpeaker(
            enabled=not args.no_tts,
            rate=args.tts_rate,
            model_dir=args.tts_model_dir,
            sid=args.tts_sid,
            num_threads=args.tts_num_threads,
            provider=args.tts_provider,
        )
        self.recognizer = SherpaOnnxRecognizer(
            model_dir=args.model_dir,
            sample_rate=args.sample_rate,
            record_seconds=args.record_seconds,
            input_device=args.input_device,
            debug_audio=args.debug_audio,
        )
        self.voice_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.camera = None
        self.object_detector = None
        self.personal_detector = None
        self.vl_assistant = None
        self.vl_finder = None
        self.face_app = None
        self.face_db = None
        self.static_session = None
        self.pose_model = None
        self.dynamic_model = None
        self.pose_feature = None
        self.fall_features = deque(maxlen=20)

        self.last_objects = []
        self.last_faces = []
        self.last_vl_candidates = []
        self.last_frame_shape = (480, 640, 3)
        self.last_fall_result = "SAFE"
        self.last_fall_static = "none"
        self.last_fall_static_score = 0.0
        self.last_fall_dynamic = 0
        self.last_fall_dynamic_score = 0.0
        self.fall_flag = False
        self.fall_flag_started_at = 0.0
        self.static_fall_times = deque()
        self.last_status = "starting"
        self.danger = False
        self.last_danger_prompt = 0.0

    def speak(self, text):
        self.speaker.speak(text)

    def load_all(self):
        self._load_camera()
        self._load_object_models()
        self._load_vl_models()
        self._load_face_models()
        self._load_fall_models()
        try:
            self.recognizer.load()
            self._start_voice_thread()
        except VoiceDependencyError as exc:
            self.speak(f"语音识别没有启动：{exc}")
        self.last_status = "ready"
        self.speak("你好，我是你的智能管家，请问有什么可以帮您？")

    def _load_camera(self):
        from camera.depth_camera import load_orbbec_cpp_camera

        install_endpoint_log_filter()
        OrbbecCppCamera = load_orbbec_cpp_camera()
        self.camera = OrbbecCppCamera(align_depth_to_color=True, mirror=False)
        self.camera.start()
        print(f"Orbbec camera info: {self.camera.info()}")

    def _load_object_models(self):
        try:
            from object import locator

            self.object_detector = locator.load_model()
            self.personal_detector = locator.load_active_personal_model() if locator.USE_PERSONAL_MODEL else None
        except Exception as exc:
            self.speak(f"物体检测模型加载失败：{exc}")

    def _load_vl_models(self):
        try:
            from vl import create_vl_assistant
            from vl.object_finder_legacy import create_vl_finder

            model_name = existing_model_path(self.args.vl_model)
            fallback_name = existing_model_path(self.args.vl_fallback_model)
            self.vl_assistant = create_vl_assistant(
                backend=self.args.vl_backend,
                model_name=model_name,
                fallback_model_name=fallback_name,
            )
            self.vl_finder = create_vl_finder(
                backend=self.args.vl_backend,
                model_name=model_name,
                fallback_model_name=fallback_name,
            )
            if hasattr(self.vl_assistant, "load"):
                self.vl_assistant.load()
            if hasattr(self.vl_finder, "load"):
                self.vl_finder.load()
        except Exception as exc:
            self.speak(f"视觉语言模型加载失败：{exc}")

    def _load_face_models(self):
        try:
            from face import recognition

            if not recognition.DEFAULT_MODEL_PATH.exists():
                self.speak("人脸库还没有训练，人脸识别待命失败。")
                return
            vectors, labels, metadata = recognition.load_model(recognition.DEFAULT_MODEL_PATH)
            self.face_db = (vectors, labels, metadata)
            self.face_app = recognition.load_face_app(
                model_name=metadata.get("model_name", recognition.INSIGHTFACE_MODEL_NAME),
                det_size=recognition.DET_SIZE,
            )
        except Exception as exc:
            self.speak(f"人脸识别模型加载失败：{exc}")

    def _load_fall_models(self):
        try:
            import joblib
            import onnxruntime as ort
            from ultralytics import YOLO
            from fall import multimodal as fall_mm

            self.fall_mm = fall_mm
            fall_mm.CONF_THRES = self.args.fall_static_conf
            self.static_session = ort.InferenceSession(
                fall_mm.STATIC_MODEL_PATH,
                providers=["CPUExecutionProvider"],
            )
            self.pose_model = YOLO(fall_mm.POSE_MODEL_PATH)
            self.dynamic_model = joblib.load(fall_mm.DYNAMIC_MODEL_PATH)
        except Exception as exc:
            self.speak(f"跌倒检测模型加载失败：{exc}")

    def _start_voice_thread(self):
        thread = threading.Thread(target=self._voice_loop, name="smartcare-main-voice", daemon=True)
        thread.start()

    def _voice_loop(self):
        if self.args.listen_mode == "push":
            print("语音监听已启动：按 Enter 开始录音，再按 Enter 结束录音；输入 Q 退出。")
        else:
            print("语音监听已启动：后台连续录音识别。窗口按 Q/ESC 退出。")
        while not self.stop_event.is_set():
            if self.args.listen_mode == "push":
                value = input("> ").strip()
                if value.lower() == "q":
                    self.voice_queue.put("退出")
                    break
            try:
                if self.args.listen_mode == "push":
                    text = self.recognizer.listen_until_enter()
                else:
                    text = self.recognizer.listen_once()
            except Exception as exc:
                self.voice_queue.put(f"__error__:{exc}")
                time.sleep(1.0)
                continue
            if text:
                self.voice_queue.put(text)
            elif self.args.listen_mode != "push":
                time.sleep(0.2)

    def run(self):
        self.load_all()
        frame_idx = 0
        try:
            while not self.stop_event.is_set():
                frame_set = self.camera.read(200)
                if frame_set is None:
                    continue
                color_img = frame_set["color"]
                depth_img = frame_set["depth"]
                self.last_frame_shape = color_img.shape

                self._handle_voice_queue(color_img, depth_img)
                if frame_idx % max(1, self.args.object_every) == 0:
                    self._update_objects(color_img, depth_img)
                if frame_idx % max(1, self.args.face_every) == 0:
                    self._update_faces(color_img)
                if frame_idx % max(1, self.args.fall_every) == 0:
                    self._update_fall(color_img)

                view = self._draw_view(color_img.copy())
                cv2.imshow("SmartCare Main", view)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                frame_idx += 1
        finally:
            self.close()

    def close(self):
        self.stop_event.set()
        if self.camera is not None:
            self.camera.stop()
        cv2.destroyAllWindows()
        self.speaker.close()

    def _handle_voice_queue(self, color_img, depth_img):
        while not self.voice_queue.empty():
            text = self.voice_queue.get()
            if text.startswith("__error__:"):
                self.speak(text.replace("__error__:", "语音识别失败：", 1))
                continue

            print(f"[识别文本] {text}")
            if self.danger:
                if is_safety_confirmation(text):
                    self.danger = False
                    self.fall_flag = False
                    self.static_fall_times.clear()
                    self.last_fall_result = "SAFE"
                    self.speak("好的，危险状态已解除。")
                    continue
                self.speak("我检测到可能跌倒。请回答：我没事，或者我还好。")
                continue

            command = parse_command(text)
            print(f"[解析命令] {command}")
            if command.intent == "exit":
                self.speak("好的，我先退出。")
                self.stop_event.set()
            elif command.intent == "help":
                self.speak(HELP_TEXT)
            elif command.intent == "describe_scene":
                self._voice_describe(color_img)
            elif command.intent == "find_object":
                self._voice_find(command.target, color_img, depth_img)
            elif command.intent == "summarize_objects":
                self._voice_summarize_objects()
            elif command.intent == "recognize_face":
                self._voice_recognize_face()
            elif command.intent in {"fall_status", "start_fall_monitor"}:
                self.speak(f"跌倒检测持续运行中，当前状态是{self.last_fall_result}。")
            elif command.intent in {"stop_current", "stop_fall_monitor"}:
                self.speak("主程序中的持续检测不能单独停止。按 Q 可以退出主程序。")
            else:
                self.speak("我还没听懂这个指令。" + HELP_TEXT)

    def _update_objects(self, color_img, depth_img):
        if self.object_detector is None:
            return
        try:
            from object import locator

            model, device, stride, names, pt, imgsz = self.object_detector
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
            if self.personal_detector is not None:
                detections.extend(locator.detect_objects(
                    self.personal_detector["model"],
                    color_img,
                    self.personal_detector["device"],
                    self.personal_detector["pt"],
                    self.personal_detector["stride"],
                    self.personal_detector["names"],
                    self.personal_detector["imgsz"],
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
            self.last_objects = detections
        except Exception as exc:
            self.last_status = f"object error: {exc}"

    def _update_faces(self, color_img):
        if self.face_app is None or self.face_db is None:
            return
        try:
            from face import recognition

            vectors, labels, metadata = self.face_db
            threshold = float(metadata.get("threshold", recognition.DEFAULT_THRESHOLD))
            display_names = metadata.get("display_names", {})
            results = []
            for face in recognition.detect_faces(self.face_app, color_img)[:3]:
                label, distance, similarity = recognition.predict_embedding(face.embedding, vectors, labels, threshold)
                name = "Unknown" if label == "unknown" else display_names.get(label, label)
                results.append({
                    "box": recognition.face_box(face, color_img.shape),
                    "name": name,
                    "similarity": similarity,
                })
            self.last_faces = results
        except Exception as exc:
            self.last_status = f"face error: {exc}"

    def _update_fall(self, color_img):
        if self.static_session is None or self.pose_model is None or self.dynamic_model is None:
            return
        try:
            from fall.features import PerFrameFeature

            fall_mm = self.fall_mm
            if self.pose_feature is None:
                self.pose_feature = PerFrameFeature(color_img.shape[1], color_img.shape[0], fps=25.0, conf_thr=0.2)

            input_tensor, ratio, dwdh = fall_mm.preprocess(color_img)
            static_outputs = self.static_session.run(None, {"images": input_tensor})
            static_results = fall_mm.postprocess(static_outputs[0], ratio, dwdh)
            static_label, static_score = fall_mm.get_static_result(static_results)
            self.last_fall_static = static_label
            self.last_fall_static_score = static_score

            pose_results = self.pose_model(color_img, verbose=False, conf=self.args.yolov8_conf)[0]
            raw_kpts, bbox = fall_mm.get_pose_result(pose_results, color_img)
            feat = self.pose_feature.compute_frame_feat(raw_kpts, bbox)
            self.fall_features.append(feat)

            dynamic_pred = 0
            dynamic_score = 0.0
            if len(self.fall_features) == self.fall_features.maxlen:
                window_input = np.array(self.fall_features).flatten().reshape(1, -1)
                dynamic_score = fall_mm.get_dynamic_score(self.dynamic_model, window_input)
                dynamic_pred = 1 if dynamic_score >= 0.5 else 0
            self.last_fall_dynamic = dynamic_pred
            self.last_fall_dynamic_score = dynamic_score

            now = time.time()
            if dynamic_pred == 1 and not self.fall_flag:
                self.fall_flag = True
                self.fall_flag_started_at = now
                self.static_fall_times.clear()

            if self.fall_flag and static_label == "fall":
                self.static_fall_times.append(now)

            while self.static_fall_times and now - self.static_fall_times[0] > self.args.fall_static_window:
                self.static_fall_times.popleft()

            if self.fall_flag:
                if len(self.static_fall_times) > self.args.fall_static_required_frames:
                    self.last_fall_result = "FALL"
                    if not self.danger:
                        self.danger = True
                        self.last_danger_prompt = 0.0
                elif now - self.fall_flag_started_at <= self.args.fall_static_window:
                    self.last_fall_result = "SUSPECT"
                elif not self.danger:
                    self.fall_flag = False
                    self.static_fall_times.clear()
                    self.last_fall_result = "SAFE"
            elif not self.danger:
                self.last_fall_result = "SAFE"

            if self.danger and time.time() - self.last_danger_prompt >= self.args.danger_prompt_interval:
                self.speak("我检测到可能跌倒。你还好吗？如果没事，请说我没事，或者我还好。")
                self.last_danger_prompt = time.time()
        except Exception as exc:
            self.last_status = f"fall error: {exc}"

    def _voice_describe(self, color_img):
        if self.vl_assistant is None:
            self.speak("视觉语言模型不可用。")
            return
        try:
            from vl.types import VLRequest

            response = self.vl_assistant.run(
                color_img,
                VLRequest(intent="describe_scene", response_mode="answer", max_new_tokens=80),
            )
            self.speak(response.answer or "我没有得到画面描述。")
        except Exception as exc:
            self.speak(f"描述画面失败：{exc}")

    def _voice_find(self, target, color_img, depth_img):
        if not target:
            self.speak("请告诉我想找什么。")
            return
        try:
            from object import locator
            from vl.object_finder_legacy import build_vl_depth_result, parse_user_query

            self.last_vl_candidates = []
            query = parse_user_query(target)
            matched = self._find_detected_object(query)
            if matched is not None:
                sentence = locator.build_object_sentence(matched, color_img.shape)
                self.speak(sentence or f"我找到了{target}。")
                return

            if self.vl_finder is None:
                self.speak("当前检测结果里没找到，视觉语言找物模型也不可用。")
                return

            query = parse_user_query(target)
            candidates = self.vl_finder.find_target_box(color_img, query)
            for candidate in candidates:
                candidate.display_label = query.grounding_prompt or "target"
            self.last_vl_candidates = candidates[:5]
            if not candidates:
                self.speak(f"没有找到{target}。")
                return
            result = build_vl_depth_result(
                query,
                candidates[0],
                depth_img,
                color_img.shape,
                locator.get_stable_depth_from_box,
                locator.describe_direction,
            )
            self.speak(result.description)
        except Exception as exc:
            self.speak(f"找物失败：{exc}")

    def _find_detected_object(self, query):
        if not self.last_objects:
            return None
        target_parts = {
            str(query.raw_text or "").lower(),
            str(query.target_name or "").lower(),
            str(query.object_name or "").lower(),
            str(query.grounding_prompt or "").lower(),
        }
        target_parts = {part for part in target_parts if part}
        best = None
        best_score = -1.0
        from object import locator

        for det in self.last_objects:
            names = {
                str(det.get("class_name", "")).lower(),
                str(locator.get_cn_name(det.get("class_name", ""))).lower(),
            }
            score = 0.0
            for target in target_parts:
                for name in names:
                    if not target or not name:
                        continue
                    if target == name or name in target or target in name:
                        score += 1.0
            if query.color and score > 0:
                # Let detection win by name; color is only a bonus because YOLO boxes are already stable.
                score += 0.2
            if score > best_score:
                best = det
                best_score = score
        return best if best_score > 0 else None

    def _voice_summarize_objects(self):
        if not self.last_objects:
            self.speak("我还没有检测到明显的物品。")
            return
        from object import locator

        sentences = [locator.build_object_sentence(det, self.last_frame_shape) for det in self.last_objects]
        sentences = [sentence for sentence in sentences if sentence]
        self.speak("；".join(sentences[:5]) + "。")

    def _voice_recognize_face(self):
        if not self.last_faces:
            self.speak("我还没有识别到人脸。")
            return
        face = self.last_faces[0]
        if face["name"] == "Unknown":
            self.speak(f"没有识别出这个人，相似度 {face['similarity']:.2f}。")
        else:
            self.speak(f"这是{face['name']}，相似度 {face['similarity']:.2f}。")

    @staticmethod
    def _ascii_label(text, default="target"):
        label = "".join(ch if ord(ch) < 128 and ch.isprintable() else " " for ch in str(text or ""))
        label = " ".join(label.split())
        return label or default

    def _draw_view(self, frame):
        from object import locator

        for det in self.last_objects[:10]:
            locator.draw_detection(frame, det)
        for candidate in self.last_vl_candidates[:5]:
            x1, y1, x2, y2 = candidate.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            label = self._ascii_label(getattr(candidate, "display_label", "") or candidate.label)
            cv2.putText(frame, f"VL {label} {candidate.score:.2f}", (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        for face in self.last_faces:
            x1, y1, x2, y2 = face["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, f"{face['name']} {face['similarity']:.2f}", (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        fall_color = (0, 0, 255) if self.danger else (0, 255, 0)
        cv2.putText(frame, f"Fall: {self.last_fall_result}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, fall_color, 2)
        cv2.putText(
            frame,
            f"fall_static: {self.last_fall_static} {self.last_fall_static_score:.2f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"fall_dynamic: {self.last_fall_dynamic} {self.last_fall_dynamic_score:.2f} flag={int(self.fall_flag)} static_count={len(self.static_fall_times)}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(frame, f"Status: {self.last_status}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        voice_hint = "voice: enter start/stop" if self.args.listen_mode == "push" else "voice: background listening"
        cv2.putText(frame, f"SmartCare Main: Q/ESC quit, {voice_hint}", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return frame


def build_parser():
    parser = argparse.ArgumentParser(description="SmartCare integrated main program.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--record-seconds", type=float, default=5.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--debug-audio", action="store_true")
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--tts-rate", type=int, default=180)
    parser.add_argument("--tts-model-dir", default=str(DEFAULT_TTS_MODEL_DIR))
    parser.add_argument("--tts-sid", type=int, default=2)
    parser.add_argument("--tts-num-threads", type=int, default=1)
    parser.add_argument("--tts-provider", default="cpu")
    parser.add_argument("--vl-backend", choices=("qwen", "florence", "opencv"), default=VL_BACKEND)
    parser.add_argument("--vl-model", default=VL_MODEL_PATH)
    parser.add_argument("--vl-fallback-model", default=VL_FALLBACK_MODEL_PATH)
    parser.add_argument("--object-every", type=int, default=5)
    parser.add_argument("--face-every", type=int, default=10)
    parser.add_argument("--fall-every", type=int, default=1)
    parser.add_argument("--danger-prompt-interval", type=float, default=8.0)
    parser.add_argument("--fall-static-window", type=float, default=2.0)
    parser.add_argument("--fall-static-required-frames", type=int, default=5)
    parser.add_argument("--fall-static-conf", type=float, default=0.60)
    parser.add_argument("--yolov8-conf", type=float, default=0.60)
    parser.add_argument("--listen-mode", choices=("push", "continuous"), default="push")
    parser.add_argument("--push-to-talk", action="store_true", help="Compatibility alias for --listen-mode push.")
    return parser


def main():
    args = build_parser().parse_args()
    if args.push_to_talk:
        args.listen_mode = "push"
    app = SmartCareMainApp(args)
    app.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
