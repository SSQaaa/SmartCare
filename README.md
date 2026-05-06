# SmartCare 智能照护视觉系统

SmartCare 是一个面向居家/养老照护场景的本地视觉感知项目。项目把 Orbbec 深度相机、YOLO 目标检测、视觉语言模型、人脸识别和跌倒检测组合在一起，用于回答三个核心问题：

- 环境里有什么，它们在哪里，距离摄像头多远；
- 当前画面中是否出现已注册的人或已训练的个人物品；
- 是否存在跌倒风险或已经发生跌倒。

项目入口统一放在 `smart_care/` 根目录，实际实现按功能拆到 `vl/`、`object/`、`face/`、`fall/`、`camera/` 等模块中。

## 运行环境

以下命令默认在仓库根目录运行：

```powershell
cd E:\SSQ\Sophomore\zqzb\rgzndl
```

推荐通过同一个 conda 环境启动：

```powershell
conda run --live-stream --name torch-learn python <入口文件> <参数>
```

常用依赖包括：

- OpenCV、NumPy、PyTorch、Ultralytics/YOLOv5 相关依赖；
- `onnxruntime`，用于静态跌倒模型和 InsightFace 后端；
- `insightface`，用于 ArcFace 人脸识别；
- `pybind11` 和 Orbbec C++ SDK，用于构建 `orbbec_cpp_camera`；
- `transformers` 等视觉语言模型依赖，仅在使用 `qwen` 或 `florence` 后端时需要。

如果 Orbbec C++ 相机模块缺失，可重新构建：

```powershell
conda run --live-stream --name torch-learn python smart_care/camera/setup_orbbec_cpp.py build_ext --inplace
```

## 项目结构

```text
smart_care/
├─ main.py                    # 集成主程序：相机窗口、模型待命、语音总控、持续监测
├─ main_vl.py                 # 视觉语言：图像描述、按文本找物
├─ main_object_locator.py     # YOLO + 深度相机：物体检测、方位和距离估计
├─ main_object_trainer.py     # 个人物品录制、标注、训练和激活
├─ main_face.py               # 人脸采集、训练和识别
├─ main_fall.py               # 跌倒检测入口
├─ config.py                  # 通用默认配置
├─ camera/                    # Orbbec 深度相机封装和 C++ pybind 构建脚本
├─ vl/                        # Qwen/Florence/OpenCV 视觉语言找物后端
├─ object/                    # 目标定位、个人物品训练、SQLite 记忆库
├─ face/                      # InsightFace/ArcFace 人脸识别
├─ fall/                      # 静态、动态、多模态跌倒检测
├─ data/                      # 测试图片、COCO 中文名映射、object_memory.db
├─ personal_objects/          # 个人物品数据集、训练结果和权重
└─ third_party/               # Orbbec SDK 等第三方资源
```

## 功能总览

| 模块 | 入口 | 能力 |
| --- | --- | --- |
| 集成主程序 | `main.py` | 启动后加载核心模型、持续显示相机画面、后台语音监听和跌倒监测 |
| 视觉语言 | `main_vl.py` | 描述图片/相机画面，按文本描述找目标并画框 |
| 物体定位 | `main_object_locator.py` | 检测 COCO 物体和个人物品，估计左右方位、相对关系、深度距离 |
| 个人物品训练 | `main_object_trainer.py` | 录制物品视频、手动关键帧标注、自动跟踪扩展、YOLOv5 微调 |
| 人脸识别 | `main_face.py` | 采集人脸样本、训练人脸库、实时识别 |
| 跌倒检测 | `main_fall.py` | 静态 ONNX 检测、姿态序列动态检测、多模态确认 |
| 语音总控 | `main_voice.py` | 按键说话、离线中文识别、本地语音播报，并调度各视觉功能 |

## 0. 集成主程序

入口：

```text
smart_care/main.py
```

`main.py` 是推荐的总入口。启动后会尽量一次性加载并保持待命：

- Orbbec 彩色和深度相机；
- YOLOv5 COCO 物体检测模型；
- 当前激活的个人物品模型；
- VL 描述/找物模型；
- InsightFace 人脸识别模型和本地人脸库；
- 跌倒检测静态 ONNX 模型、YOLOv8 pose 模型和动态 XGBoost 模型；
- sherpa-onnx SenseVoice 中文语音识别和 sherpa-onnx VITS 本地语音播报。

运行：

```powershell
conda run --live-stream --name torch-learn python smart_care/main.py
```

如果希望先快速测试窗口和轻量功能，可用 OpenCV VL 后端：

```powershell
conda run --live-stream --name torch-learn python smart_care/main.py --vl-backend opencv
```

窗口和语音行为：

- 启动后持续 `imshow` 摄像头窗口；
- 默认在后台连续录音识别语音指令；
- 默认按键说话，终端中按 `Enter` 开始录音，再按 `Enter` 结束录音并识别；如需后台连续录音，可加 `--listen-mode continuous`；
- 窗口中按 `Q` 或 `ESC` 退出；
- 物体、人脸、跌倒状态会持续在后台更新；
- 语音可询问“看看周围”“周围有什么”“帮我找白色杯子”“这是谁”等；
- 找物和物体播报使用机器人第一人称，例如“白色杯子可能在我正前方，距离我约 0.33 米”。

跌倒危险确认：

- 静态或动态跌倒检测任一分支检测到 `fall`，主程序进入危险状态；
- 语音助手会询问“你还好吗”；
- 只有听到 `没事`、`我没关系`、`我还好`、`没问题` 等确认词，才会解除危险状态；
- 危险状态未解除时，其他语音指令会被暂停处理。

常用参数：

- `--vl-backend qwen|florence|opencv`：主程序加载的 VL 后端；
- `--object-every 5`：每隔多少帧跑一次物体检测；
- `--face-every 10`：每隔多少帧跑一次人脸识别；
- `--fall-every 1`：每隔多少帧跑一次跌倒检测，默认每帧检测；
- `--danger-prompt-interval 8`：危险状态下每隔多少秒重复询问；
- `--fall-static-window 2`：动态分支置疑似 flag 后，静态分支累计确认的时间窗口；
- `--fall-static-required-frames 5`：2 秒窗口内静态 fall 帧数必须超过该值才进入 `FALL`；
- `--fall-static-conf 0.6`：静态跌倒检测置信度阈值；
- `--yolov8-conf 0.6`：YOLOv8 pose 动态分支检测置信度阈值；
- `--listen-mode push|continuous`：语音输入方式，默认 `push`，即按 `Enter` 开始录音，再按 `Enter` 结束录音；
- `--no-tts`：只打印播报文本，不调用语音引擎。
- `--tts-engine sherpa|sapi|pyttsx3`：语音播报引擎，默认 `sherpa`；
- `--tts-model-dir smart_care/models/sherpa-onnx-vits-zh-ll`：sherpa VITS 模型目录；

说明：

- `main.py` 会尽量加载所有模型，但某个模型加载失败不会直接终止整个主程序，而是播报/打印对应错误；
- 如果使用 `qwen` 或 `florence`，启动会明显更慢，因为 VL 模型会预加载；
- 跌倒检测模型路径仍来自 `smart_care/fall/multimodal.py` 顶部常量。
- `main.py` 会过滤 Orbbec `Endpoint 0x81 bandwidth` 日志和 InsightFace `estimate deprecated` warning，避免刷屏影响语音/终端交互。
- 当前个人物品模型置信度阈值是 `0.40`；COCO 常见物体检测阈值是 `0.60`。

## 1. 视觉语言描述与找物

入口：

```text
smart_care/main_vl.py
```

支持两类任务：

- `describe`：描述一张图片或相机画面；
- `find`：根据文本描述寻找目标物体，并在窗口中画框。

支持三种后端：

- `qwen`：使用 Qwen3-VL，语义能力强，加载和推理较慢；
- `florence`：使用 Florence grounding；
- `opencv`：基于颜色和轮廓的轻量找物，速度快，但只适合简单颜色/形状目标。

描述图片：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_vl.py describe --image smart_care/data/test.png
```

描述 Orbbec 相机画面，每 3 秒推理一次：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_vl.py describe --camera --interval 3
```

用 Qwen 按文字找物：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_vl.py find "蓝色瓶子" --camera --backend qwen --interval 3
```

用 OpenCV 快速找简单颜色目标：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_vl.py find "blue bottle" --camera --backend opencv --interval 1
```

窗口按键：

- `Q` / `ESC`：退出；
- `R`：相机模式下立即触发下一次推理，前提是当前没有推理任务正在运行。

相关配置在 `smart_care/config.py`：

- `VL_BACKEND`：默认后端；
- `VL_MODEL_PATH` / `VL_FALLBACK_MODEL_PATH`：本地 VL 模型路径；
- `VL_INFER_WIDTH`：推理前缩放宽度；
- `VL_INTERVAL_SEC`：相机模式默认推理间隔。

## 2. 物体检测、关系描述与深度定位

入口：

```text
smart_care/main_object_locator.py
```

默认运行方式：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_object_locator.py
```

这个模块默认使用 Orbbec 深度相机和 YOLOv5：

- 加载 `yolov5/yolov5s.pt` 检测 COCO 常见物体；
- 如果 `object_memory.db` 中存在激活的个人物品模型，会同时加载个人模型；
- 使用深度图估计目标距离摄像头的距离；
- 输出中文方位描述，例如“瓶子在你左前方，距离摄像头约 1.20 米”；
- 对杯子、瓶子、手机、遥控器、书等小物体，会尝试描述“在桌子上面”“在椅子旁边”等关系；
- 可按 `V` 输入 VL 找物目标，让视觉语言模块辅助寻找更具体的目标。

窗口按键：

- `Q` / `ESC`：退出；
- `V`：输入一个 VL 找物描述，例如 `灰色鼠标`。

主要开关在 `smart_care/object/locator.py` 顶部：

- `USE_DEPTH_CAMERA = True`：使用 Orbbec 深度模式；
- `USE_PERSONAL_MODEL = True`：加载当前激活的个人物品模型；
- `ENABLE_VL_OBJECT_FINDER = True`：启用窗口内 VL 找物；
- `VL_BACKEND = "opencv"`：实时定位中的 VL 默认后端，避免重模型卡住摄像头循环。

## 3. 个人物品注册与训练

入口：

```text
smart_care/main_object_trainer.py
```

目标是把某个个人物品训练成一个单类别 YOLOv5 模型，并写入 `smart_care/data/object_memory.db`。训练完成后，该物品会被自动设为 active，`main_object_locator.py` 和 `main.py` 会优先加载它。

训练时需要两个名字：

- `object_name`：终端和数据集使用的英文或拼音 ID，例如 `white_cup`；
- `display_name`：中文显示名和语音别名，例如 `白色杯子`。

完整流程：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_object_trainer.py white_cup --display-name 白色杯子
```

流程会依次执行：

1. 打开 Orbbec 彩色画面录制物品视频；
2. 从视频中抽取关键帧；
3. 手动框选关键帧目标；
4. 用 OpenCV tracker 在相邻关键帧之间扩展标注；
5. 复核所有样本；
6. 生成 YOLO 数据集和增强样本；
7. 调用 `yolov5/train.py` 微调；
8. 保存 `best.pt` 并激活该个人物品。

使用已有视频：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_object_trainer.py white_cup --display-name 白色杯子 --video E:\path\to\object.mp4
```

只使用已有数据集重新训练：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_object_trainer.py white_cup --display-name 白色杯子 --train-only
```

训练参数示例：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_object_trainer.py fangshai --train-only --img 512 --batch 4 --workers 0 --epochs 150
```

常用参数：

- `name`：物品 ID，会被规范化为小写字母/数字/下划线；
- `--display-name`：中文显示名，也是语音和播报使用的物品别名；如果不填写，程序会提示输入；
- `--video`：跳过录制，直接使用已有视频；
- `--train-only`：跳过录制和标注，直接训练已有数据集；
- `--epochs`：训练轮数，默认 `150`；
- `--batch`：batch size，默认 `4`；
- `--img`：训练图片尺寸，默认 `640`；
- `--workers`：DataLoader worker 数，默认 `2`；
- `--patience`：早停耐心值，默认 `10`。

录制窗口按键：

- `S`：开始录制；
- `E`：结束录制；
- `Q`：取消。

样本复核窗口按键：

- 鼠标拖框：修改当前目标框；
- `A` / `D`：上一张 / 下一张；
- `C` / `X`：标为无目标负样本；
- `Space` / `Enter`：确认；
- `Q`：取消。

输出位置：

```text
smart_care/personal_objects/<object_name>/
├─ captures/                  # 录制视频
├─ dataset/                   # YOLO 数据集
│  ├─ images/train, images/val
│  └─ labels/train, labels/val
└─ runs/finetune/weights/
   ├─ best.pt
   └─ last.pt
```

## 4. 人脸识别

入口：

```text
smart_care/main_face.py
```

人脸模块基于 InsightFace `buffalo_s`，采集时保存人脸裁剪图和 ArcFace embedding，训练时生成一个 `npz` 人脸库。

新用户推荐直接使用 `enroll`，采集后自动训练：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_face.py enroll --person-name zhangsan --display-name 张三 --samples 20
```

只采集样本：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_face.py collect --person-name zhangsan --samples 20
```

只训练人脸库：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_face.py train
```

实时识别：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_face.py recognize
```

常用参数：

- `--person-name`：人员 ID，建议使用英文、数字或下划线；
- `--display-name`：显示名；
- `--samples`：采集样本数，默认 `20`；
- `--interval-frames`：每隔多少帧自动保存一张，默认 `8`；
- `--model`：人脸库路径，默认 `smart_care/face/face_model.npz`；
- `--threshold`：识别阈值，越小越严格；
- `--det-size`：InsightFace 检测尺寸，默认 `256`；
- `--process-every`：识别时每隔多少帧推理一次，默认 `5`。

采集规则：

- 画面里必须刚好有一张人脸才会保存；
- 自动保存开启，也可以按 `S` 手动保存；
- `Q` / `ESC` 退出；
- 如果相机刚启动还没有帧，程序会等待多次再报错。

数据位置：

```text
smart_care/face/faces/<person_name>/
├─ metadata.json
└─ samples/
   ├─ <person_name>_0000.jpg
   └─ <person_name>_0000.npy
```

## 5. 跌倒检测

入口：

```text
smart_care/main_fall.py
```

支持三种模式：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_fall.py static
conda run --live-stream --name torch-learn python smart_care/main_fall.py dynamic
conda run --live-stream --name torch-learn python smart_care/main_fall.py multimodal
```

模式说明：

- `static`：使用 YOLOv5 导出的 ONNX 模型，对单帧进行 `fall / stand / sit / squat / run` 分类检测；
- `dynamic`：使用 YOLOv8 pose 提取人体关键点，按 20 帧窗口计算时序特征，再用 XGBoost 判断跌倒；
- `multimodal`：动态模型先触发 `SUSPECT`，静态模型确认后输出 `FALL`，若连续约 2 秒未被静态模型确认则恢复 `SAFE`。

当前跌倒模块的视频路径和模型路径写在源码顶部：

- `smart_care/fall/static.py`
- `smart_care/fall/dynamic.py`
- `smart_care/fall/multimodal.py`

需要更换测试视频或模型时，修改这些文件顶部的 `MODEL_PATH`、`VIDEO_PATH`、`POSE_MODEL_PATH`、`DYNAMIC_MODEL_PATH` 等常量。

输出文件：

- `dynamic` 会写出 `fall_dynamic_result.mp4`；
- `multimodal` 会写出 `fall_multimodal_result.mp4`。

## 6. 语音总控

入口：

```text
smart_care/main_voice.py
```

语音模块是一个独立总控层，不替代原有入口。它通过中文语音指令调度 VL、物体定位、人脸识别和跌倒检测，并把结果用本机语音播报出来。需要所有能力持续待命时，推荐使用 `main.py`。

默认交互方式：

- 按 `Enter` 后开始录音；
- 默认录音 `5` 秒；
- 识别完成后自动执行命令；
- 输入 `Q` 退出。

安装语音依赖：

```powershell
conda run --live-stream --name torch-learn python -m pip install sherpa-onnx sounddevice pyttsx3
```

sherpa-onnx SenseVoice 中文模型目录约定：

```text
smart_care/models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/
```

sherpa-onnx VITS 播报模型目录约定：

```text
smart_care/models/sherpa-onnx-vits-zh-ll/
```

如果模型放在其他位置，用 `--model-dir` 指定：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_voice.py --model-dir E:\SSQ\models\sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09
```

启动语音总控：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_voice.py
```

无麦克风调试，用文字模拟语音识别：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_voice.py --no-tts --text "帮助"
```

常用语音指令：

- `看看周围` / `描述画面`：读取一帧 Orbbec 画面，调用 VL 描述；
- `周围有什么` / `检测物品`：检测常见物体和当前激活个人物品，播报方位和距离；
- `帮我找杯子` / `找一下蓝色瓶子`：按描述找物，能读到深度时播报距离，播报使用机器人第一人称；
- `这是谁` / `识别人脸`：调用已训练人脸库识别画面中的单个人脸；
- `开始跌倒检测`：启动 `main_fall.py multimodal` 子进程；
- `停止跌倒检测` / `停止当前功能`：停止当前长时间功能；
- `跌倒状态`：播报跌倒检测是否正在运行；
- `帮助`：播报可用指令；
- `退出`：关闭语音总控。

常用参数：

- `--record-seconds 5`：连续监听模式下每次录音秒数；按键模式由第二次 `Enter` 结束录音；
- `--sample-rate 16000`：录音采样率；
- `--list-audio-devices`：列出可用录音设备，排查是否录到了错误麦克风；
- `--input-device 设备编号或名称`：指定 `sounddevice` 输入设备；
- `--debug-audio`：每次识别后打印录音时长、RMS 和峰值，排查音量过小或爆音；
- `--no-tts`：只打印，不调用本机语音播报；
- `--tts-rate 180`：播报语速；
- `--tts-engine sherpa|sapi|pyttsx3`：播报引擎，默认 `sherpa`；
- `--tts-model-dir <path>`：sherpa VITS 模型目录；
- `--tts-sid 2`：sherpa VITS 说话人 ID，可尝试不同编号切换音色；
- `--vl-backend qwen|florence|opencv`：语音调度 VL 时使用的后端；
- `--fall-mode static|dynamic|multimodal`：语音启动跌倒检测时使用的模式；
- `--text "帮我找杯子"`：调试命令解析和调度，不使用麦克风。

说明：

- `main_voice.py` 第一版不做持续唤醒词监听，避免误触发；`main.py` 默认也是按键说话，可用 `--listen-mode continuous` 改为后台连续录音；
- 如果缺少 sherpa-onnx 模型或语音依赖，启动时会给出明确提示；
- 跌倒检测当前仍沿用 `fall/` 模块中硬编码的视频和模型路径，语音模块负责启动、停止和运行状态播报；
- 语音播报默认使用 sherpa-onnx VITS，本地离线生成 wav 后播放；如果缺少 sherpa-onnx 或模型文件，可临时用 `--tts-engine sapi` 或 `--tts-engine pyttsx3` 回退到系统语音。

## 7. 数据和记忆文件

重要数据文件：

- `smart_care/data/coco80_cn_names.txt`：COCO 类别中文名映射；
- `smart_care/data/object_memory.db`：个人物品 SQLite 记忆库；
- `smart_care/face/face_model.npz`：人脸识别库；
- `smart_care/personal_objects/`：个人物品数据集、训练日志和权重。

`object_memory.db` 中只会有一个 active 个人物品。每次训练或 `--train-only` 成功后，当前物品会被自动激活。

## 常见问题

### Orbbec 相机打不开

检查：

- USB 连接是否正常；
- 是否被其他程序占用；
- `smart_care/camera/orbbec_cpp_camera.cp311-win_amd64.pyd` 或对应 Python 版本的 `.pyd` 是否存在；
- `smart_care/camera/OrbbecSDK.dll` 是否存在；
- 是否需要重新构建 C++ pybind 模块。

重建命令：

```powershell
conda run --live-stream --name torch-learn python smart_care/camera/setup_orbbec_cpp.py build_ext --inplace
```

### 打开成功但没有画面

Orbbec pipeline 启动成功不代表第一时间就有同步的 color/depth 帧。部分模块会在读不到帧时继续等待；如果仍失败，检查 USB、权限、相机占用情况，以及深度流帧率是否过低。

### 个人物品训练内存不足

优先降低图片尺寸、worker 数或 batch：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_object_trainer.py fangshai --train-only --img 512 --batch 4 --workers 0
```

同时关闭其他占显存/内存的 Python、浏览器或 VL 模型进程。

### VL 推理很慢

`qwen` 和 `florence` 属于重模型，首次加载可能需要较长时间。实时相机模式建议：

```powershell
--interval 3
```

快速调试可使用：

```powershell
--backend opencv
```

### OpenCV 窗口中文乱码

OpenCV 原生 `putText` 对中文支持不好。项目窗口内尽量显示英文，中文结果主要输出在终端。

### 人脸采集没有保存样本

采集逻辑要求“刚好一张人脸”。如果没有保存：

- 确认 `Collect Face Samples` 窗口已经出现；
- 画面中只保留一个人；
- 提高光照，让脸靠近摄像头；
- 如果一直读不到帧，检查 Orbbec 相机连接和占用；
- 如果一直检测不到脸，可尝试调大 `--det-size`。

示例：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_face.py enroll --person-name zhangsan --samples 20 --interval-frames 3
```

## 开发提示

- 新入口尽量放在 `smart_care/main_*.py`；
- 具体实现放到对应功能目录；
- 和 Orbbec 相关的公共逻辑优先复用 `camera/depth_camera.py`；
- 个人物品状态通过 `object/memory.py` 维护，不建议手工改数据库；
- 跌倒模块目前仍偏实验脚本风格，模型路径和视频路径需要按本机环境调整。
