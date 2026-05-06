import json
import importlib.util
import json
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import cv2
import numpy as np


QWEN3_VL_4B_MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
QWEN3_VL_2B_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
FLORENCE_MODEL_NAME = "microsoft/Florence-2-large"
PHRASE_GROUNDING_TASK = "<CAPTION_TO_PHRASE_GROUNDING>"
FLORENCE_CAPTION_TASK = "<MORE_DETAILED_CAPTION>"
MIN_VL_RANK_SCORE = 0.25
MIN_COLOR_MATCH_SCORE = 0.06

COLOR_WORDS = {
    "gray": ("gray", "gray"),
    "black": ("black", "black"),
    "white": ("white", "white"),
    "red": ("red", "red"),
    "blue": ("blue", "blue"),
    "green": ("green", "green"),
    "yellow": ("yellow", "yellow"),
    "pink": ("pink", "pink"),
    "purple": ("purple", "purple"),
    "灰": ("灰色", "gray"),
    "灰色": ("灰色", "gray"),
    "黑": ("黑色", "black"),
    "黑色": ("黑色", "black"),
    "白": ("白色", "white"),
    "白色": ("白色", "white"),
    "红": ("红色", "red"),
    "红色": ("红色", "red"),
    "蓝": ("蓝色", "blue"),
    "蓝色": ("蓝色", "blue"),
    "绿": ("绿色", "green"),
    "绿色": ("绿色", "green"),
    "黄": ("黄色", "yellow"),
    "黄色": ("黄色", "yellow"),
    "粉": ("粉色", "pink"),
    "粉色": ("粉色", "pink"),
    "紫": ("紫色", "purple"),
    "紫色": ("紫色", "purple"),
}

OBJECT_WORDS = {
    "wireless mouse": "wireless mouse",
    "mouse": "mouse",
    "keyboard": "keyboard",
    "cup": "cup",
    "bottle": "bottle",
    "cell phone": "cell phone",
    "phone": "cell phone",
    "book": "book",
    "laptop": "laptop",
    "paper bag": "paper bag",
    "bag": "bag",
    "box": "box",
    "remote": "remote",
    "无线鼠标": "wireless mouse",
    "鼠标": "mouse",
    "机械键盘": "keyboard",
    "键盘": "keyboard",
    "水杯": "cup",
    "杯子": "cup",
    "瓶子": "bottle",
    "手机": "cell phone",
    "书": "book",
    "笔记本电脑": "laptop",
    "电脑": "laptop",
    "纸袋": "paper bag",
    "袋子": "bag",
    "盒子": "box",
    "遥控器": "remote",
}


@dataclass
class ParsedQuery:
    raw_text: str
    target_name: str
    grounding_prompt: str
    color: str = ""
    object_name: str = ""


@dataclass
class VLObjectCandidate:
    box: list[int]
    label: str
    score: float
    source: str = "vl"


@dataclass
class VLDepthResult:
    target_name: str
    box: list[int]
    direction: str
    depth_m: Optional[float]
    description: str
    label: str = ""
    score: float = 0.0
    valid_count: int = 0
    source: str = "vl"


def parse_user_query(text: str) -> ParsedQuery:
    raw_text = text.strip()
    lowered = raw_text.lower()

    color = ""
    color_en = ""
    for key in sorted(COLOR_WORDS, key=len, reverse=True):
        value = COLOR_WORDS[key]
        if key in raw_text or key in lowered:
            color, color_en = value
            break

    object_name = ""
    object_en = ""
    for key in sorted(OBJECT_WORDS, key=len, reverse=True):
        if key in raw_text or key in lowered:
            object_name = key
            object_en = OBJECT_WORDS[key]
            break

    prompt_parts = [part for part in (color_en, object_en) if part]
    grounding_prompt = " ".join(prompt_parts) if prompt_parts else raw_text
    target_name = "".join(part for part in (color, object_name) if part) or raw_text
    return ParsedQuery(
        raw_text=raw_text,
        target_name=target_name,
        grounding_prompt=grounding_prompt,
        color=color_en,
        object_name=object_en,
    )


def clamp_box(box: Iterable[float], image_shape) -> Optional[list[int]]:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def normalize_box(box: Iterable[float], image_shape) -> Optional[list[int]]:
    values = [float(v) for v in box]
    if len(values) != 4:
        return None

    h, w = image_shape[:2]
    max_value = max(values)
    if max_value <= 1.5:
        values = [values[0] * w, values[1] * h, values[2] * w, values[3] * h]
    elif max_value <= 1000 and (values[2] > w or values[3] > h):
        values = [values[0] / 1000 * w, values[1] / 1000 * h, values[2] / 1000 * w, values[3] / 1000 * h]
    return clamp_box(values, image_shape)


def extract_json_objects(text: str) -> list[dict]:
    objects = []
    for match in re.finditer(r"\{.*?\}", text, flags=re.S):
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def parse_qwen_boxes(text: str, image_shape) -> list[VLObjectCandidate]:
    lowered = text.strip().lower()
    if lowered in {"null", "none", "not found", "no target", "no object"}:
        return []

    candidates = []
    for item in extract_json_objects(text):
        found_value = item.get("found")
        if found_value is False or str(found_value).lower() in {"false", "no", "not_found"}:
            return []
        raw_box = (
            item.get("box")
            or item.get("bbox")
            or item.get("bbox_2d")
            or item.get("bounding_box")
        )
        if raw_box is None:
            continue
        box = normalize_box(raw_box, image_shape)
        if box is None:
            continue
        label = str(item.get("label") or item.get("target") or item.get("object") or "qwen target")
        candidates.append(VLObjectCandidate(box=box, label=label, score=0.0, source="qwen"))

    if candidates:
        return candidates

    bracket_boxes = re.findall(r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]", text)
    for raw in bracket_boxes[:5]:
        box = normalize_box([float(v) for v in raw], image_shape)
        if box is not None:
            candidates.append(VLObjectCandidate(box=box, label="qwen target", score=0.4, source="qwen"))
    return candidates


def qwen_says_not_found(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered in {"null", "none", "not found", "no target", "no object"}:
        return True
    for item in extract_json_objects(text):
        found_value = item.get("found")
        if found_value is False or str(found_value).lower() in {"false", "no", "not_found"}:
            return True
    return False


def qwen_prompt(query: ParsedQuery) -> str:
    return (
        "You are helping a robot locate an object in an RGB image. "
        "Only locate the object if it is clearly visible and matches the user's description. "
        "Do not guess. Do not choose a merely similar object. "
        "If the target is not clearly visible, return exactly: {\"found\": false}. "
        "If the target is visible, return exactly one JSON object. "
        "Use pixel coordinates in the original image. "
        "JSON schema: {\"found\":true, \"label\":\"target name\", \"box\":[x1,y1,x2,y2]}. "
        "Do not include extra text. "
        f"User target: {query.raw_text}. English hint: {query.grounding_prompt}."
    )


def qwen_description_prompt() -> str:
    return (
        "\u8bf7\u7528\u4e2d\u6587\u63cf\u8ff0\u8fd9\u5f20\u753b\u9762\uff0c"
        "\u91cd\u70b9\u8bf4\u660e\u4e3b\u8981\u7269\u4f53\u3001"
        "\u989c\u8272\u3001\u4f4d\u7f6e\u5173\u7cfb\u548c\u573a\u666f\u3002"
        "\u56de\u7b54\u63a7\u5236\u5728120\u5b57\u4ee5\u5185\u3002"
    )


class Qwen3VLFinder:
    def __init__(
        self,
        model_name: str = QWEN3_VL_4B_MODEL_NAME,
        fallback_model_name: str = QWEN3_VL_2B_MODEL_NAME,
        device: str = "",
        auto_fallback: bool = True,
    ):
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.device = device
        self.auto_fallback = auto_fallback
        self.active_model_name = model_name
        self.dtype = None
        self.processor = None
        self.model = None

    def load(self):
        if self.model is not None:
            return
        self._load_model(self.model_name)

    def _load_model(self, model_name: str):
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        try:
            print(f"Loading Qwen3-VL model: {model_name} on {self.device}. Please wait...")
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            model_kwargs = {
                "torch_dtype": self.dtype,
                "trust_remote_code": True,
            }
            if importlib.util.find_spec("accelerate") is not None:
                model_kwargs["low_cpu_mem_usage"] = True
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                **model_kwargs,
            ).to(self.device)
            self.model.eval()
            self.active_model_name = model_name
            print(f"Qwen3-VL model ready: {model_name}")
        except Exception as exc:
            self.processor = None
            self.model = None
            if self.auto_fallback and model_name != self.fallback_model_name:
                print(f"Qwen3-VL 4B load failed, fallback to 2B: {exc}")
                self._load_model(self.fallback_model_name)
                return
            raise RuntimeError(
                f"Cannot load {model_name}. Installed transformers={transformers.__version__}. "
                "Qwen3-VL may require a newer transformers version and enough VRAM. "
                "Try: pip install -U transformers accelerate"
            ) from exc

    def _build_inputs(self, pil_img, prompt: str):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            return self.processor(text=[text], images=[pil_img], return_tensors="pt")
        except TypeError:
            return self.processor(text=text, images=pil_img, return_tensors="pt")

    def _generate_text(self, color_img: np.ndarray, prompt: str, max_new_tokens: int = 256) -> str:
        self.load()

        import torch
        from PIL import Image

        rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        inputs = self._build_inputs(pil_img, prompt)
        inputs = {
            key: value.to(self.device, dtype=self.dtype) if torch.is_floating_point(value)
            else value.to(self.device) if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[-1]
        output_ids = generated_ids[:, input_len:]
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    def find_target_box(self, color_img: np.ndarray, query: ParsedQuery, max_boxes: int = 5) -> list[VLObjectCandidate]:
        generated_text = self._generate_text(color_img, qwen_prompt(query), max_new_tokens=128)
        if qwen_says_not_found(generated_text):
            return []
        candidates = parse_qwen_boxes(generated_text, color_img.shape)[:max_boxes]
        if not candidates:
            candidates = generate_opencv_candidates(color_img, query, max_boxes=max_boxes)
        return rank_candidates(color_img, query, candidates)

    def describe_image(self, color_img: np.ndarray, max_new_tokens: int = 128) -> str:
        return self._generate_text(color_img, qwen_description_prompt(), max_new_tokens=max_new_tokens)


class FlorenceGroundingFinder:
    def __init__(self, model_name: str = FLORENCE_MODEL_NAME, device: str = ""):
        self.model_name = model_name
        self.device = device
        self.dtype = None
        self.processor = None
        self.model = None

    def load(self):
        if self.model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"Loading Florence grounding model: {self.model_name} on {self.device}. Please wait...")
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        print(f"Florence grounding model ready: {self.model_name}")

    def find_target_box(self, color_img: np.ndarray, query: ParsedQuery, max_boxes: int = 5) -> list[VLObjectCandidate]:
        self.load()

        import torch
        from PIL import Image

        rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        task_text = PHRASE_GROUNDING_TASK + query.grounding_prompt
        inputs = self.processor(text=task_text, images=pil_img, return_tensors="pt")
        inputs = {
            key: value.to(self.device, dtype=self.dtype) if torch.is_floating_point(value)
            else value.to(self.device) if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
                do_sample=False,
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text,
            task=PHRASE_GROUNDING_TASK,
            image_size=(pil_img.width, pil_img.height),
        )
        payload = parsed.get(PHRASE_GROUNDING_TASK, {})
        bboxes = payload.get("bboxes", [])
        labels = payload.get("labels", [])

        candidates = []
        for i, raw_box in enumerate(bboxes[:max_boxes]):
            box = clamp_box(raw_box, color_img.shape)
            if box is None:
                continue
            label = labels[i] if i < len(labels) else query.grounding_prompt
            candidates.append(VLObjectCandidate(box=box, label=label, score=0.0, source="florence"))
        return rank_candidates(color_img, query, candidates)

    def describe_image(self, color_img: np.ndarray, max_new_tokens: int = 128) -> str:
        self.load()

        import torch
        from PIL import Image

        rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        inputs = self.processor(text=FLORENCE_CAPTION_TASK, images=pil_img, return_tensors="pt")
        inputs = {
            key: value.to(self.device, dtype=self.dtype) if torch.is_floating_point(value)
            else value.to(self.device) if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=3,
                do_sample=False,
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        try:
            parsed = self.processor.post_process_generation(
                generated_text,
                task=FLORENCE_CAPTION_TASK,
                image_size=(pil_img.width, pil_img.height),
            )
            return str(parsed.get(FLORENCE_CAPTION_TASK, generated_text)).strip()
        except Exception:
            return generated_text.strip()


class OpenCVObjectFinder:
    def __init__(self, device: str = ""):
        self.device = device

    def find_target_box(self, color_img: np.ndarray, query: ParsedQuery, max_boxes: int = 5) -> list[VLObjectCandidate]:
        candidates = generate_opencv_candidates(color_img, query, max_boxes=max_boxes)
        return rank_candidates(color_img, query, candidates)

    def describe_image(self, color_img: np.ndarray, max_new_tokens: int = 128) -> str:
        return "Description is not available for the opencv backend."


def create_vl_finder(
    backend: str,
    model_name: str = "",
    fallback_model_name: str = QWEN3_VL_2B_MODEL_NAME,
    device: str = "",
):
    backend = backend.lower().strip()
    if backend in {"opencv", "cv"}:
        return OpenCVObjectFinder(device=device)
    if backend == "qwen":
        return Qwen3VLFinder(
            model_name=model_name or QWEN3_VL_4B_MODEL_NAME,
            fallback_model_name=fallback_model_name,
            device=device,
        )
    if backend == "florence":
        return FlorenceGroundingFinder(model_name=model_name or FLORENCE_MODEL_NAME, device=device)
    raise ValueError(f"Unsupported VL backend: {backend}")


def color_match_score(color_img: np.ndarray, box: list[int], color: str) -> float:
    if not color:
        return 0.0

    x1, y1, x2, y2 = box
    roi = color_img[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    if color == "gray":
        mask = (s < 70) & (v > 35) & (v < 230)
    elif color == "white":
        mask = (s < 60) & (v > 170)
    elif color == "black":
        mask = v < 70
    elif color == "red":
        mask = ((h < 10) | (h > 170)) & (s > 60) & (v > 50)
    elif color == "blue":
        mask = (h > 90) & (h < 135) & (s > 50) & (v > 50)
    elif color == "green":
        mask = (h > 35) & (h < 85) & (s > 50) & (v > 50)
    elif color == "yellow":
        mask = (h > 18) & (h < 38) & (s > 50) & (v > 50)
    elif color == "pink":
        mask = (h > 135) & (h < 175) & (s > 35) & (v > 80)
    elif color == "purple":
        mask = (h > 125) & (h < 160) & (s > 40) & (v > 50)
    else:
        return 0.0

    return float(np.mean(mask))


def generate_opencv_candidates(color_img: np.ndarray, query: ParsedQuery, max_boxes: int = 5) -> list[VLObjectCandidate]:
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    if query.color:
        mask = color_mask(hsv, query.color)
    else:
        mask = np.ones(color_img.shape[:2], dtype=np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(mask, 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    img_area = color_img.shape[0] * color_img.shape[1]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < img_area * 0.002 or area > img_area * 0.45:
            continue
        ratio = w / float(max(1, h))
        if query.object_name and "mouse" in query.object_name and not (1.1 <= ratio <= 3.8):
            continue
        candidates.append(VLObjectCandidate([x, y, x + w, y + h], query.grounding_prompt, 0.0, source="opencv"))

    return candidates[:max_boxes]


def color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    if color == "gray":
        mask = (s < 70) & (v > 35) & (v < 230)
    elif color == "white":
        mask = (s < 60) & (v > 170)
    elif color == "black":
        mask = v < 70
    elif color == "red":
        mask = ((h < 10) | (h > 170)) & (s > 60) & (v > 50)
    elif color == "blue":
        mask = (h > 90) & (h < 135) & (s > 50) & (v > 50)
    elif color == "green":
        mask = (h > 35) & (h < 85) & (s > 50) & (v > 50)
    elif color == "yellow":
        mask = (h > 18) & (h < 38) & (s > 50) & (v > 50)
    elif color == "pink":
        mask = (h > 135) & (h < 175) & (s > 35) & (v > 80)
    elif color == "purple":
        mask = (h > 125) & (h < 160) & (s > 40) & (v > 50)
    else:
        mask = np.ones(hsv.shape[:2], dtype=bool)
    return (mask.astype(np.uint8) * 255)


def rank_candidates(color_img: np.ndarray, query: ParsedQuery, candidates: list[VLObjectCandidate]) -> list[VLObjectCandidate]:
    if not candidates:
        return []

    img_h, img_w = color_img.shape[:2]
    img_area = img_h * img_w
    ranked = []
    for candidate in candidates:
        x1, y1, x2, y2 = candidate.box
        area = max(1, (x2 - x1) * (y2 - y1))
        area_ratio = area / float(max(1, img_area))
        size_score = 1.0 - min(1.0, abs(area_ratio - 0.06) / 0.25)
        label_score = 0.2 if query.object_name and query.object_name in candidate.label.lower() else 0.0
        color_score = color_match_score(color_img, candidate.box, query.color)
        if query.color and color_score < MIN_COLOR_MATCH_SCORE:
            continue
        candidate.score = 0.45 * color_score + 0.35 * size_score + label_score
        if candidate.score >= MIN_VL_RANK_SCORE:
            ranked.append(candidate)

    return sorted(ranked, key=lambda item: item.score, reverse=True)


def build_vl_depth_result(
    query: ParsedQuery,
    candidate: VLObjectCandidate,
    depth_img: np.ndarray,
    color_shape,
    depth_fn: Callable,
    direction_fn: Callable,
) -> VLDepthResult:
    depth_mm, valid_count = depth_fn(candidate.box, depth_img, color_shape)
    direction = direction_fn(candidate.box, color_shape)
    depth_m = depth_mm / 1000.0 if depth_mm is not None else None

    if depth_m is None:
        description = f"{query.target_name}可能在我{direction}，但深度数据不稳定"
    else:
        description = f"{query.target_name}可能在我{direction}，距离我约{depth_m:.2f}米"

    return VLDepthResult(
        target_name=query.target_name,
        box=candidate.box,
        direction=direction,
        depth_m=depth_m,
        description=description,
        label=candidate.label,
        score=candidate.score,
        valid_count=valid_count,
        source=candidate.source,
    )
