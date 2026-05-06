# SmartCare 使用说明

以下命令默认在项目根目录运行：

```powershell
cd E:\SSQ\Sophomore\zqzb\rgzndl
```

推荐统一使用你的 conda 环境：

```powershell
conda run --live-stream --name torch-learn python <程序路径> <参数>
```

## 1. VL 视觉语言模型

入口文件：

```text
smart_care/main_vl.py
```

功能：

- `describe`：描述画面。
- `find`：根据文本描述寻找物品，并在窗口里画框。

输入来源：

- `--image`：使用一张图片。
- `--camera`：使用 Orbbec 深度相机的彩色画面。

常用参数：

- `--backend qwen`：使用 Qwen3-VL，语义能力强，但速度慢。
- `--backend opencv`：使用 OpenCV 颜色/轮廓逻辑，速度快，但只能做简单找物。
- `--backend florence`：使用 Florence grounding 后端。
- `--interval 3`：摄像头模式下每隔 3 秒推理一次，避免窗口卡死。

示例：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_vl.py describe --image smart_care/data/test.png
```

```powershell
conda run --live-stream --name torch-learn python smart_care/main_vl.py describe --camera --interval 3
```

```powershell
conda run --live-stream --name torch-learn python smart_care/main_vl.py find "蓝色瓶子" --camera --backend qwen --interval 3
```

```powershell
conda run --live-stream --name torch-learn python smart_care/main_vl.py find "blue bottle" --camera --backend opencv --interval 1
```

说明：

- Qwen3-VL 很重，第一次加载可能几十秒到几分钟。
- 视频窗口里的文字尽量使用英文，避免 OpenCV 中文乱码。
- 找物时如果模型明确判断没有目标，会返回未找到，不会再强行给一个伪置信度。

## 2. 物体检测与深度定位

入口文件：

```text
smart_care/main_object_locator.py
```

功能：

- 使用 YOLOv5 检测 COCO 常见物品。
- 使用用户自定义物品模型检测注册物品。
- 使用 Orbbec 深度相机计算目标到摄像头的距离。
- 按 `V` 后可以输入 VL 找物描述。

运行：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_object_locator.py
```

按键：

- `Q` 或 `ESC`：退出。
- `V`：输入一个 VL 找物目标，例如 `蓝色瓶子`。

说明：

- 必须能打开 Orbbec 深度相机。
- 如果没有检测到深度相机，会抛出明确错误。
- 距离默认表示“目标距离摄像头”的距离。

## 3. 用户自定义物品注册与训练

入口文件：

```text
smart_care/main_object_trainer.py
```

功能：

- 录制自定义物品视频。
- 手动标注关键帧。
- 自动跟踪并生成数据集。
- 训练 YOLOv5 自定义物品模型。
- 训练完成后写入 `object_memory.db` 并激活该物品。

最简单的完整流程：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_object_trainer.py fangshai
```

使用已有视频：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_object_trainer.py fangshai --video E:\path\to\video.mp4
```

已经标注好数据集，只重新训练：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_object_trainer.py fangshai --train-only
```

训练参数：

- `--epochs 150`：训练轮数。
- `--batch 4`：批大小，越大越快，但更吃显存和内存。
- `--img 640`：训练图片尺寸，越大越慢，可能更准。
- `--workers 2`：DataLoader worker 数量，越大可能越快，但更吃内存。
- `--patience 10`：早停耐心值。

推荐参数：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_object_trainer.py fangshai --train-only --img 512 --batch 4 --workers 2
```

如果内存报错：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_object_trainer.py fangshai --train-only --img 512 --batch 4 --workers 0
```

如果想尝试更快：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_object_trainer.py fangshai --train-only --img 640 --batch 4 --workers 2
```

数据集划分：

- 训练集和验证集会尽量保持约 `9:1`。
- 使用 `--train-only` 时，也会先对已有数据集进行一次 train/val 重平衡。

标注窗口按键：

- 鼠标拖框：修改目标框。
- `A` / `D`：上一张 / 下一张。
- `C` 或 `X`：标为无目标。
- `Space` 或 `Enter`：确认。
- `Q`：取消。

## 4. 人脸识别

入口文件：

```text
smart_care/main_face.py
```

子命令：

- `collect`：只采集人脸样本。
- `train`：用已有样本训练人脸库。
- `enroll`：采集样本后立刻训练，推荐新用户使用。
- `recognize`：实时人脸识别。

推荐：新用户录入

```powershell
conda run --live-stream --name torch-learn python smart_care/main_face.py enroll --person-name zhangsan --samples 20
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

- `--person-name zhangsan`：人员 ID，建议使用英文、数字或下划线。
- `--display-name 张三`：显示名称，不填则使用 `person-name`。
- `--samples 20`：采集多少张人脸样本。
- `--interval-frames 8`：每隔多少帧保存一次样本。
- `--model smart_care/face/face_model.npz`：人脸库路径。
- `--threshold`：识别阈值，越小越严格。
- `--process-every 3`：每隔几帧识别一次，越大越流畅但响应更慢。

说明：

- 摄像头输入默认使用 Orbbec 深度相机的彩色流。
- 采集时要求画面中尽量只有一个人脸。

## 5. 跌倒检测

入口文件：

```text
smart_care/main_fall.py
```

模式：

- `static`：静态单帧跌倒检测。
- `dynamic`：基于人体姿态序列的动态跌倒检测。
- `multimodal`：动态检测触发疑似状态，再由静态检测确认跌倒。

静态检测：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_fall.py static
```

动态检测：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_fall.py dynamic
```

多模态检测：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_fall.py multimodal
```

多模态逻辑：

- 动态检测到跌倒：进入 `SUSPECT` 疑似跌倒状态。
- 只有在 `SUSPECT` 状态下，静态检测也显示跌倒，才输出 `FALL`。
- 连续 2 秒静态都未检测到跌倒，则清空疑似状态，回到 `SAFE`。

说明：

- 当前 fall 模块的视频路径和模型路径主要写在各自模块顶部。
- 如果要换测试视频，需要修改：
  - `smart_care/fall/static.py`
  - `smart_care/fall/dynamic.py`
  - `smart_care/fall/multimodal.py`

## 6. 常见问题

### Orbbec 深度相机打不开

检查：

- USB 是否连接正常。
- 是否被其他程序占用。
- C++ pybind 模块是否存在于 `smart_care/camera/`。
- 如果需要重新编译：

```powershell
conda run --live-stream --name torch-learn python smart_care/camera/setup_orbbec_cpp.py build_ext --inplace
```

### YOLOv5 训练中途 MemoryError

优先降低 DataLoader 压力：

```powershell
conda run --live-stream --name torch-learn python smart_care/main_object_trainer.py fangshai --train-only --img 512 --batch 4 --workers 0
```

同时关闭浏览器、VL 模型进程、其他 Python 进程。

### VL 推理很慢

Qwen3-VL 属于重模型，低频使用比较合适。摄像头模式建议：

```powershell
--interval 3
```

需要快速调试时用：

```powershell
--backend opencv
```

### OpenCV 窗口中文乱码

OpenCV 原生 `putText` 不支持中文显示。本项目窗口内尽量只显示英文，中文输出主要放在终端。
