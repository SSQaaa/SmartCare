# SmartCare 智能管家

SmartCare 是一个面向居家照护场景的本地智能视觉与语音助手。它运行在连接了 Orbbec 深度相机的电脑或机器人上，持续观察周围环境，并用中文语音和你交互。

它主要解决这些问题：

- 看看周围有什么物品，告诉你大概方位和距离；
- 帮你找杯子、鼠标、瓶子、书等目标物；
- 识别画面里的人是谁；
- 持续检测是否有人跌倒；
- 检测到跌倒风险后主动询问，只有听到“我没事”“我还好”等确认后才解除危险状态；
- 允许你训练自己的个人物品，比如“我的防晒”“我的鼠标”，之后可以用语音找它。

这个项目默认本地运行：视觉模型、语音识别和语音播报都在本机执行，不依赖云端语音服务。

## 快速开始

本文默认你已经进入 `smart_care/` 目录：

```powershell
cd E:\SSQ\Sophomore\zqzb\rgzndl\smart_care
```

```powershell
python main.py
```

启动成功后会出现摄像头窗口。窗口里会显示：

- `Fall: SAFE / SUSPECT / FALL`：当前跌倒状态；
- `fall_static`：静态跌倒检测结果；
- `fall_dynamic`：动态姿态跌倒检测结果；
- 物体框、人脸框、VL 找物框；
- 底部提示：按 `Q` 或 `ESC` 退出。

默认语音交互方式是按键说话：

1. 在终端里按一次 `Enter` 开始录音；
2. 说出指令；
3. 再按一次 `Enter` 结束录音并识别；
4. SmartCare 会执行命令并语音播报结果。

例如你可以说：

```text
看看周围
周围有什么
帮我找杯子
帮我找一下我的鼠标
这是谁
这是什么
跌倒状态
帮助
退出
```

## 日常怎么用

### 描述周围

说：

```text
看看周围
描述画面
这是什么
```

SmartCare 会读取当前相机画面，调用视觉语言模型描述画面内容，然后播报结果。

### 查周围有什么

说：

```text
周围有什么
检测物品
识别物品
```

SmartCare 会用 YOLO 检测常见物体和当前激活的个人物品模型，并结合深度相机估计距离。播报示例：

```text
杯子在我正前方，距离我约0.33米。
```

注意这里的“我”指机器人或相机所在的位置，不是用户的位置。

### 找东西

说：

```text
帮我找杯子
找一下蓝色瓶子
帮我找一下我的鼠标
```

找物逻辑是：

1. 先在当前已经检测到的 YOLO 常见物体和个人物品里找；
2. 如果找不到，再调用 VL 找物；
3. 如果有深度图，会播报方位和距离；
4. VL 找物时会在窗口里用英文框出候选目标，避免 OpenCV 中文乱码。

### 识别人脸

说：

```text
这是谁
识别人脸
```

SmartCare 会识别画面里的人脸。如果本地人脸库没有训练，会提示先录入并训练。

窗口里如果出现 `Unknown 0.07`，意思是人脸模型检测到了一个“像脸”的区域，但没有匹配到本地人脸库。它有时会误框到杯子、图案或背景纹理上，这是人脸误检，不是物体检测结果。

### 跌倒检测

启动 `main.py` 后，跌倒检测会持续运行。窗口会显示：

```text
Fall: SAFE
fall_static: ...
fall_dynamic: ...
```

当前逻辑是：

- 动态检测先触发疑似跌倒 flag；
- 在 2 秒窗口内，静态检测超过 5 帧判断为 `fall`，才进入 `FALL`；
- 一旦进入危险状态，语音助手会询问“你还好吗？”；
- 只有听到“我没事”“我还好”“没关系”“没问题”等确认词后，才解除危险状态。

## 推荐入口

### 一体化主程序

日常使用推荐启动：

```powershell
conda run --live-stream --name torch-learn python main.py
```

它会同时准备这些能力：

- Orbbec 彩色与深度相机；
- 常见物体检测；
- 个人物品检测；
- VL 场景描述和找物；
- 人脸识别；
- 静态和动态跌倒检测；
- sherpa-onnx 语音识别；
- sherpa-onnx VITS 语音播报。

如果只想调试画面和轻量找物，可以用 OpenCV VL 后端：

```powershell
conda run --live-stream --name torch-learn python main.py --vl-backend opencv
```

如果暂时不想播报，只看终端文字：

```powershell
conda run --live-stream --name torch-learn python main.py --no-tts
```

### 只启动语音助手

如果你只想测试语音调度，不打开完整主程序：

```powershell
conda run --live-stream --name torch-learn python main_voice.py
```

测试 TTS 是否能正常发声：

```powershell
conda run --live-stream --name torch-learn python main_voice.py --test-tts
```

不用麦克风，直接用文字模拟一句语音命令：

```powershell
conda run --live-stream --name torch-learn python main_voice.py --no-tts --text "帮助"
```

## 第一次使用前要准备什么

### 1. Orbbec 相机

项目默认使用 Orbbec 深度相机。请确认：

- USB 已连接；
- 没有被其他程序占用；
- `camera/OrbbecSDK.dll` 存在；
- 对应 Python 版本的 `orbbec_cpp_camera*.pyd` 存在。

如果相机模块缺失，可以重新构建：

```powershell
conda run --live-stream --name torch-learn python camera/setup_orbbec_cpp.py build_ext --inplace
```

### 2. 语音识别模型

默认语音识别使用 sherpa-onnx SenseVoice，模型目录为：

```text
models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/
```

目录中至少需要：

```text
model.int8.onnx
tokens.txt
```

### 3. 语音播报模型

默认语音播报使用 sherpa-onnx VITS，模型目录为：

```text
models/sherpa-onnx-vits-zh-ll/
```

目录中至少需要：

```text
model.onnx
lexicon.txt
tokens.txt
```

可以尝试不同音色：

```powershell
conda run --live-stream --name torch-learn python main_voice.py --test-tts --tts-sid 0
conda run --live-stream --name torch-learn python main_voice.py --test-tts --tts-sid 1
conda run --live-stream --name torch-learn python main_voice.py --test-tts --tts-sid 2
```

### 4. 麦克风

如果识别效果差，先列出录音设备：

```powershell
conda run --live-stream --name torch-learn python main_voice.py --list-audio-devices
```

指定正确麦克风，例如设备编号是 `2`：

```powershell
conda run --live-stream --name torch-learn python main.py --input-device 2 --debug-audio
```

`--debug-audio` 会打印录音音量诊断：

- `peak` 很小：可能录错设备、麦克风静音或离得太远；
- `peak` 接近 1：可能爆音，建议降低输入音量。

## 录入人脸

如果你希望 SmartCare 能回答“这是谁”，需要先录入并训练人脸库。

录入并自动训练：

```powershell
conda run --live-stream --name torch-learn python main_face.py enroll --person-name zhangsan --display-name 张三 --samples 20
```

参数含义：

- `--person-name`：内部 ID，建议用英文、拼音、数字或下划线；
- `--display-name`：播报时使用的名字；
- `--samples`：采集样本数量。

采集时规则：

- 画面里最好只有一个人；
- 光线要够；
- 人脸尽量靠近相机；
- 按 `Q` 或 `ESC` 退出；
- 如果刚启动相机暂时没有帧，程序会多等几次。

只采集样本：

```powershell
conda run --live-stream --name torch-learn python main_face.py collect --person-name zhangsan --display-name 张三 --samples 20
```

只训练已有样本：

```powershell
conda run --live-stream --name torch-learn python main_face.py train
```

单独运行实时人脸识别：

```powershell
conda run --live-stream --name torch-learn python main_face.py recognize
```

## 注册个人物品

如果你希望 SmartCare 能找“我的防晒”“我的鼠标”这类个人物品，需要训练个人物品模型。

训练时需要两个名字：

- `object_name`：内部 ID，只用英文、拼音、数字或下划线，例如 `my_mouse`；
- `display_name`：用户听到和说出的名字，例如 `我的鼠标`。

示例：

```powershell
conda run --live-stream --name torch-learn python main_object_trainer.py my_mouse --display-name 我的鼠标
```

流程大致是：

1. 打开相机录制物品视频；
2. 从视频里抽取关键帧；
3. 手动框选目标；
4. 程序扩展标注并生成数据集；
5. 微调 YOLOv5；
6. 训练完成后自动激活该个人物品模型。

如果已经有视频：

```powershell
conda run --live-stream --name torch-learn python main_object_trainer.py my_mouse --display-name 我的鼠标 --video E:\path\to\object.mp4
```

如果数据集已经做好，只重新训练：

```powershell
conda run --live-stream --name torch-learn python main_object_trainer.py my_mouse --display-name 我的鼠标 --train-only
```

训练完成后，你就可以在主程序里说：

```text
帮我找一下我的鼠标
```

当前个人物品检测置信度默认是 `0.40`，常见物体检测置信度默认是 `0.60`。

## 单独使用某个功能

一般推荐使用 `main.py`。如果只想单独测试某个功能，可以用下面入口。

### 场景描述和 VL 找物

描述图片：

```powershell
conda run --live-stream --name torch-learn python main_vl.py describe --image data/test.png
```

用相机描述画面：

```powershell
conda run --live-stream --name torch-learn python main_vl.py describe --camera --interval 3
```

用相机找物：

```powershell
conda run --live-stream --name torch-learn python main_vl.py find "蓝色瓶子" --camera --backend qwen --interval 3
```

### 物体定位

```powershell
conda run --live-stream --name torch-learn python main_object_locator.py
```

窗口中：

- `Q` 或 `ESC` 退出；
- `V` 输入一个 VL 找物目标。

### 跌倒检测

单独运行静态检测：

```powershell
conda run --live-stream --name torch-learn python main_fall.py static
```

单独运行动态检测：

```powershell
conda run --live-stream --name torch-learn python main_fall.py dynamic
```

单独运行多模态检测：

```powershell
conda run --live-stream --name torch-learn python main_fall.py multimodal
```

日常使用不需要单独启动这些，`main.py` 已经持续运行跌倒检测。

## 常用参数

### 主程序参数

```powershell
python main.py --help
```

常用项：

- `--vl-backend qwen|florence|opencv`：选择视觉语言后端；
- `--listen-mode push|continuous`：语音输入方式，默认 `push`；
- `--input-device <编号或名称>`：指定麦克风；
- `--debug-audio`：打印录音音量诊断；
- `--no-tts`：只打印文字，不播报；
- `--tts-sid <编号>`：切换 sherpa VITS 音色；
- `--object-every 5`：每隔多少帧做一次物体检测；
- `--face-every 10`：每隔多少帧做人脸识别；
- `--fall-every 1`：每隔多少帧做一次跌倒检测。

### 语音助手参数

```powershell
python main_voice.py --help
```

常用项：

- `--list-audio-devices`：列出麦克风设备；
- `--input-device <编号或名称>`：指定麦克风；
- `--debug-audio`：诊断录音音量；
- `--test-tts`：测试语音播报；
- `--text "帮助"`：用文字模拟语音命令；
- `--tts-model-dir <路径>`：指定 sherpa VITS 模型目录；
- `--model-dir <路径>`：指定 sherpa SenseVoice 识别模型目录。

## 常见问题

### 启动后没有摄像头画面

先检查：

- Orbbec 是否插好；
- 是否被其他程序占用；
- SDK DLL 和 `.pyd` 文件是否存在；
- USB 带宽是否稳定；
- 程序是否在正确 conda 环境里运行。

如果相机刚打开但暂时没读到帧，稍等几秒；部分 Orbbec 流需要一点启动时间。

### 终端刷出 Endpoint bandwidth 日志

主程序会尽量过滤 Orbbec 的 `Endpoint 0x81 bandwidth` 日志，避免影响交互。如果仍然出现，一般不代表功能错误，只是 SDK 的底层带宽输出。

### 语音识别很差

优先排查麦克风：

```powershell
conda run --live-stream --name torch-learn python main_voice.py --list-audio-devices
```

然后指定设备并开启诊断：

```powershell
conda run --live-stream --name torch-learn python main.py --input-device 2 --debug-audio
```

说话时尽量：

- 离麦克风近一些；
- 环境安静一些；
- 开始录音后再说话；
- 说完后再按第二次 `Enter` 结束录音。

### 语音播报最后一个字被吞掉

项目已经在 sherpa VITS 生成的 wav 末尾补了静音。如果仍然明显，可以在 [tts.py](voice/tts.py) 中把：

```python
tail_silence_ms=350
```

调大到 `500`。

### TTS 没声音

先测试：

```powershell
conda run --live-stream --name torch-learn python main_voice.py --test-tts
```

确认：

- `sherpa-onnx` 已安装；
- `models/sherpa-onnx-vits-zh-ll/` 下模型文件完整；
- Windows 输出设备和音量正常；

### 窗口里中文乱码

OpenCV 原生文字绘制不适合中文。项目窗口内尽量显示英文标签，中文结果主要在终端和语音播报中输出。

### 人脸框出现 Unknown

`Unknown` 表示检测到了人脸或疑似人脸，但没有匹配到本地人脸库。它可能是真人未录入，也可能是杯子、图案、背景纹理造成的误检。

### 找物时没有找到

可以尝试：

- 换一种说法，例如“找杯子”改成“找白色杯子”；
- 把目标放到相机视野内；
- 确认光照足够；
- 如果是个人物品，确认已经训练并激活；
- 对于复杂目标，使用 `--vl-backend qwen` 比 `opencv` 更强，但加载更慢。

## 数据保存在哪里

常见数据位置：

```text
face/faces/                 人脸采集样本
face/face_model.npz         人脸识别库
data/object_memory.db       个人物品记忆库
personal_objects/           个人物品数据集和训练权重
models/                     本地语音模型
```

一般用户不需要手动修改这些文件。

## 使用建议

- 日常使用优先启动 `main.py`；
- 第一次使用先测试相机、TTS 和麦克风；
- 人脸和个人物品需要先录入训练，之后语音命令才会更好用；
- 跌倒检测是辅助提醒，不应替代真实医疗或安全设备；
- 如果模型加载很慢，属于正常现象，尤其是 VL 模型首次启动时。
