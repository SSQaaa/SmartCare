# SmartCare Project Structure

The project is now organized by feature. Real implementation code should live
inside feature folders, not in old root-level scripts.

```text
smart_care/
├─ main_vl.py                 # VL scene description and object finding
├─ main_object_locator.py     # Object detection + depth locating
├─ main_object_trainer.py     # Personal object training
├─ main_face.py               # Face recognition
├─ main_fall.py               # Fall detection
├─ config.py                  # Small set of common defaults
├─ vl/                        # VL assistant implementation
├─ object/                    # Object locating, memory, trainer
├─ face/                      # Face recognition implementation and data
├─ fall/                      # Static, dynamic, multimodal fall detection
├─ camera/                    # Camera adapters
├─ data/                      # Small test images and maps
├─ personal_objects/          # Personal object datasets and weights
└─ third_party/               # Vendor SDKs
```

## Commands

VL image description:

```powershell
python smart_care/main_vl.py describe --image smart_care/data/test.png
```

VL camera description:

```powershell
python smart_care/main_vl.py describe --camera
```

VL object finding:

```powershell
python smart_care/main_vl.py find "蓝色瓶子" --camera
```

Object locator:

```powershell
python smart_care/main_object_locator.py
```

Train a personal object:

```powershell
python smart_care/main_object_trainer.py baxiantong
python smart_care/main_object_trainer.py baxiantong --video smart_care/personal_objects/baxiantong/baxiantong.mp4
```

Face recognition:

```powershell
python smart_care/main_face.py recognize
```

Fall detection:

```powershell
python smart_care/main_fall.py static
python smart_care/main_fall.py dynamic
python smart_care/main_fall.py multimodal
```

## Where To Edit

- VL code: `smart_care/vl/`
- Object locating and training: `smart_care/object/`
- Face recognition: `smart_care/face/`
- Fall detection: `smart_care/fall/`
- Camera adapters: `smart_care/camera/`
