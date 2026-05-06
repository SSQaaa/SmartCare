from pathlib import Path


ROOT = Path(__file__).resolve().parent

# Camera
CAMERA_INDEX = 0

# VL
VL_BACKEND = "qwen"
VL_MODEL_PATH = r"E:\SSQ\models\Qwen3-VL-2B-Instruct"
VL_FALLBACK_MODEL_PATH = r"E:\SSQ\models\Qwen3-VL-2B-Instruct"
VL_INFER_WIDTH = 448
VL_INTERVAL_SEC = 3.0
VL_DESCRIBE_TOKENS = 80


def existing_model_path(path: str) -> str:
    value = Path(path)
    return str(value) if value.exists() else path
