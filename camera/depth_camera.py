import os
import sys
from pathlib import Path

import cv2


SMART_CARE_ROOT = Path(__file__).resolve().parents[1]
CAMERA_ROOT = Path(__file__).resolve().parent
SDK_ROOT = SMART_CARE_ROOT / "third_party" / "OrbbecSDK_C_C++_v1.5.7_win_x64_release" / "SDK"
SDK_DLL_DIR = SDK_ROOT / "lib"


class DepthCameraUnavailable(RuntimeError):
    pass


def _prepare_orbbec_import():
    if hasattr(os, "add_dll_directory") and SDK_DLL_DIR.exists():
        os.add_dll_directory(str(SDK_DLL_DIR))
    if hasattr(os, "add_dll_directory") and CAMERA_ROOT.exists():
        os.add_dll_directory(str(CAMERA_ROOT))
    for path in (CAMERA_ROOT, SMART_CARE_ROOT):
        if str(path) not in sys.path:
            sys.path.append(str(path))


def load_orbbec_cpp_camera():
    _prepare_orbbec_import()
    try:
        from orbbec_cpp_camera import OrbbecCppCamera
    except ImportError as exc:
        raise DepthCameraUnavailable(
            "Orbbec depth camera module is not available. "
            "Build it first with: python camera/setup_orbbec_cpp.py build_ext --inplace"
        ) from exc
    return OrbbecCppCamera


class DepthColorCamera:
    """Small cv2.VideoCapture-like wrapper around the Orbbec color stream."""

    def __init__(self, align_depth_to_color=True, mirror=False, timeout_ms=100):
        OrbbecCppCamera = load_orbbec_cpp_camera()
        self.timeout_ms = int(timeout_ms)
        self.camera = OrbbecCppCamera(align_depth_to_color=align_depth_to_color, mirror=mirror)
        self.started = False
        try:
            self.camera.start()
            self.started = True
            self.info = self.camera.info()
            print(f"Orbbec depth camera opened: {self.info}")
        except Exception as exc:
            raise DepthCameraUnavailable(
                "Cannot open Orbbec depth camera. Please check the USB cable, driver, SDK DLLs, "
                "and that the camera is not occupied by another program."
            ) from exc

    def isOpened(self):
        return self.started

    def read(self):
        if not self.started:
            return False, None
        frame_set = self.camera.read(self.timeout_ms)
        if frame_set is None or "color" not in frame_set:
            return False, None
        frame = frame_set["color"]
        # OrbbecCppCamera.color_frame_to_bgr() already returns OpenCV-ready BGR
        # frames, even when the hardware stream profile reports RGB.
        return True, frame

    def release(self):
        if self.started:
            self.camera.stop()
            self.started = False

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS and isinstance(self.info, dict):
            return float(self.info.get("color_fps") or 30)
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH and isinstance(self.info, dict):
            return float(self.info.get("color_width") or 0)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT and isinstance(self.info, dict):
            return float(self.info.get("color_height") or 0)
        return 0.0

    def set(self, _prop_id, _value):
        return False


def open_required_depth_camera():
    print("Opening Orbbec depth camera...")
    return DepthColorCamera()
