import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from config import (
    CAMERA_INDEX,
    VL_BACKEND,
    VL_DESCRIBE_TOKENS,
    VL_FALLBACK_MODEL_PATH,
    VL_INFER_WIDTH,
    VL_INTERVAL_SEC,
    VL_MODEL_PATH,
    existing_model_path,
)
from camera.depth_camera import open_required_depth_camera
from vl import VLRequest, create_vl_assistant


def resize_for_vl(frame, width=VL_INFER_WIDTH):
    if width <= 0 or frame.shape[1] <= width:
        return frame, 1.0, 1.0
    scale = width / float(frame.shape[1])
    height = max(1, int(round(frame.shape[0] * scale)))
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return resized, frame.shape[1] / float(width), frame.shape[0] / float(height)


def scale_boxes(boxes, sx, sy, frame_shape):
    if sx == 1.0 and sy == 1.0:
        return boxes
    h, w = frame_shape[:2]
    for item in boxes:
        x1, y1, x2, y2 = item.box
        item.box = [
            max(0, min(w - 1, int(round(x1 * sx)))),
            max(0, min(h - 1, int(round(y1 * sy)))),
            max(0, min(w - 1, int(round(x2 * sx)))),
            max(0, min(h - 1, int(round(y2 * sy)))),
        ]
    return boxes


def make_assistant(backend=VL_BACKEND):
    print(f"Loading VL backend: {backend}. This may take a while for Qwen/Florence models...")
    return create_vl_assistant(
        backend=backend,
        model_name=existing_model_path(VL_MODEL_PATH),
        fallback_model_name=existing_model_path(VL_FALLBACK_MODEL_PATH),
    )


def read_image(path):
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return frame


def run_once(frame, assistant, intent, text="", response_mode="answer"):
    infer_frame, sx, sy = resize_for_vl(frame)
    request = VLRequest(
        intent=intent,
        user_text=text,
        response_mode=response_mode,
        need_box=response_mode in {"locate", "both"},
        max_new_tokens=VL_DESCRIBE_TOKENS,
    )
    start = time.perf_counter()
    response = assistant.run(infer_frame, request)
    elapsed = (time.perf_counter() - start) * 1000.0
    response.boxes = scale_boxes(response.boxes, sx, sy, frame.shape)
    return response, elapsed


def print_response(response, elapsed_ms):
    print(f"{response.intent} / {response.response_mode} / {elapsed_ms:.0f} ms")
    if response.answer:
        print(response.answer)
    for item in response.boxes:
        print(f"{item.source}\t{item.label}\t{item.box}")


def ascii_label(text, fallback="target"):
    text = str(text or "")
    return text if text.isascii() else fallback


def draw_result(frame, response, status):
    for item in response.boxes:
        x1, y1, x2, y2 = item.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, f"{item.source}: {ascii_label(item.label)[:24]}", (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(frame, status, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def run_image(args, intent, response_mode):
    assistant = make_assistant(args.backend)
    frame = read_image(args.image)
    response, elapsed = run_once(frame, assistant, intent, args.text, response_mode)
    print_response(response, elapsed)

    view = frame.copy()
    draw_result(view, response, "press any key to close")
    cv2.imshow("SmartCare VL", view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_camera(args, intent, response_mode):
    assistant = make_assistant(args.backend)
    cap = open_required_depth_camera()
    if not cap.isOpened():
        raise RuntimeError("Cannot open Orbbec depth camera.")

    last_response = None
    last_elapsed = None
    last_run = 0.0
    pending = None
    status = "waiting"

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                now = time.perf_counter()
                if pending is not None and pending.done():
                    try:
                        last_response, last_elapsed = pending.result()
                        status = f"ready {last_elapsed:.0f} ms"
                        print_response(last_response, last_elapsed)
                    except Exception as exc:
                        status = f"error: {str(exc)[:80]}"
                        print(status)
                    pending = None

                if pending is None and now - last_run >= args.interval:
                    status = "running"
                    pending = executor.submit(run_once, frame.copy(), assistant, intent, args.text, response_mode)
                    last_run = now

                view = frame.copy()
                if last_response is not None:
                    draw_result(view, last_response, status)
                else:
                    cv2.putText(view, status, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("SmartCare VL", view)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r") and pending is None:
                    last_run = 0.0
    finally:
        cap.release()
        cv2.destroyAllWindows()


def add_common_args(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--camera", type=int, nargs="?", const=CAMERA_INDEX)
    parser.add_argument("--backend", choices=("qwen", "florence", "opencv"), default=VL_BACKEND)
    parser.add_argument("--interval", type=float, default=VL_INTERVAL_SEC)


def build_parser():
    parser = argparse.ArgumentParser(description="SmartCare VL.")
    subparsers = parser.add_subparsers(dest="command")

    describe = subparsers.add_parser("describe")
    add_common_args(describe)

    find = subparsers.add_parser("find")
    find.add_argument("text", help="Object description, e.g. 蓝色瓶子")
    add_common_args(find)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    args.text = getattr(args, "text", "")
    intent = "describe_scene" if args.command == "describe" else "find_object"
    response_mode = "answer" if args.command == "describe" else "locate"

    if args.image is not None:
        run_image(args, intent, response_mode)
    else:
        run_camera(args, intent, response_mode)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as exc:
        print(f"Run failed: {exc}")
        sys.exit(1)
