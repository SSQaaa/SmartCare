# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SmartCare is a local vision perception system for home/elderly care scenarios. It integrates Orbbec depth cameras, YOLO object detection, visual language models, face recognition, and fall detection to answer three core questions:
- What objects are in the environment, where are they, and how far from the camera
- Whether registered persons or trained personal objects appear in the frame
- Whether there is a fall risk or a fall has occurred

## Common Commands

All commands assume running from the repository root with the conda environment `torch-learn`:

```powershell
cd E:\SSQ\Sophomore\zqzb\rgzndl\smart_care
```

### Main Application Entry Points

```powershell
# Integrated main program (loads all models, voice control, fall monitoring)
conda run --live-stream --name torch-learn python main.py

# Quick test with lightweight OpenCV VL backend
conda run --live-stream --name torch-learn python main.py --vl-backend opencv

# Push-to-talk voice mode instead of continuous listening
conda run --live-stream --name torch-learn python main.py --push-to-talk
```

### Individual Module Entry Points

```powershell
# Visual Language: describe scene or find objects by text
conda run --live-stream --name torch-learn python main_vl.py describe --camera
conda run --live-stream --name torch-learn python main_vl.py find "蓝色瓶子" --camera --backend qwen

# Object detection with depth positioning
conda run --live-stream --name torch-learn python main_object_locator.py

# Personal object training pipeline
conda run --live-stream --name torch-learn python main_object_trainer.py white_cup --display-name 白色杯子

# Face recognition
conda run --live-stream --name torch-learn python main_face.py enroll --person-name zhangsan --display-name 张三 --samples 20
conda run --live-stream --name torch-learn python main_face.py recognize

# Fall detection (static/dynamic/multimodal)
conda run --live-stream --name torch-learn python main_fall.py multimodal

# Voice control only
conda run --live-stream --name torch-learn python main_voice.py
```

### Rebuilding C++ Camera Module

If the Orbbec C++ camera module is missing or needs rebuilding:

```powershell
conda run --live-stream --name torch-learn python camera/setup_orbbec_cpp.py build_ext --inplace
```

## High-Level Architecture

### Module Organization

The codebase is organized by feature with clear separation between entry points and implementations:

```
smart_care/
├── main_*.py              # Entry points (vl, object_locator, object_trainer, face, fall, voice)
├── config.py              # Shared configuration defaults (VL backend, model paths)
├── camera/                # Orbbec depth camera adapters and C++ pybind setup
├── vl/                    # Visual Language: Qwen/Florence/OpenCV backends
├── object/                # Object detection, personal item training, SQLite memory
├── face/                  # InsightFace/ArcFace recognition and face database
├── fall/                  # Static ONNX, dynamic XGBoost, multimodal fusion
├── voice/                 # sherpa-onnx ASR/TTS, command parser, controller
├── data/                  # COCO Chinese name map, object_memory.db
├── personal_objects/      # Personal object datasets and trained weights
└── models/                # Local sherpa-onnx ASR/TTS model directories
```

### Key Integration Patterns

**Object Detection Pipeline** (`object/locator.py`):
- Dual-model detection: YOLOv5 for COCO classes + optional personal object model
- Depth integration: Uses Orbbec depth frames to estimate object distance
- Spatial relations: Assigns "on table/bed/chair" support relations and pair relations
- Configurable thresholds: `COCO_CONF_THRES=0.60`, `PERSONAL_CONF_THRES=0.60`

**Personal Object Lifecycle** (`object/memory.py`, `object/trainer.py`):
- SQLite database at `data/object_memory.db` tracks registered objects
- Only one object can be "active" at a time; training auto-activates the new object
- Training flow: record video → extract keyframes → manual annotation → tracker expansion → YOLOv5 fine-tuning
- Output: `personal_objects/<name>/runs/finetune/weights/best.pt`

**Fall Detection Fusion** (`fall/multimodal.py`):
- Static branch: ONNX model classifies pose (fall/stand/sit/squat/run) from single frame
- Dynamic branch: YOLOv8 pose → 20-frame feature window → XGBoost classifier
- Fusion logic: Dynamic triggers `SUSPECT` flag → Static confirms within 2-second window → `FALL` state
- Danger state requires explicit voice confirmation ("我没事", "我还好") to clear

**Voice Command Flow** (`voice/commands.py`, `voice/controller.py`):
- Commands parsed from Chinese speech with fuzzy matching for variations
- Key intents: `describe_scene`, `summarize_objects`, `find_object`, `recognize_face`, `start_fall_monitor`
- First-person robot narration for spatial descriptions ("白色杯子在我正前方，距离我约0.33米")

**Visual Language Abstraction** (`vl/assistant.py`, `vl/backend.py`):
- Backend-agnostic interface supporting Qwen3-VL, Florence-2, and OpenCV color/shape matching
- Two operation modes: `describe` (scene description) and `ground` (object localization with bounding boxes)
- Configurable via `config.py`: `VL_BACKEND`, `VL_MODEL_PATH`, `VL_INFER_WIDTH`

### Critical File Paths (Hardcoded)

Several modules have hardcoded absolute paths that must be adjusted for the local environment:

- `config.py`: `VL_MODEL_PATH`, `VL_FALLBACK_MODEL_PATH`
- `object/locator.py`: `ORBBEC_SDK_DIR` (Python SDK path for legacy loader)
- `fall/multimodal.py`: `STATIC_MODEL_PATH`, `POSE_MODEL_PATH`, `DYNAMIC_MODEL_PATH`, `VIDEO_PATH`
- `voice/sherpa_asr.py`: `DEFAULT_MODEL_DIR` (defaults to the SenseVoice model under `models/`)

### Threading and Resource Management

- `main.py` runs voice recognition in a daemon thread (`_voice_loop`) feeding a queue
- Camera frames are read synchronously; heavy models (VL) run with frame skipping controlled by `--object-every`, `--face-every`, `--fall-every`
- Orbbec endpoint log spam is filtered via pipe-based stdout redirection in `install_endpoint_log_filter()`

### Memory and State

- Face database: NPZ file with embeddings, labels, and metadata at `face/face_model.npz`
- Object memory: SQLite with schema for object metadata, activation status, and paths
- Depth history: Per-object median filtering for stable distance readings (`depth_history_map`)
- Fall state: `deque(maxlen=20)` for feature window, danger state with timeout-based re-prompting
