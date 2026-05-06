# SmartCare Local Models

Place local model files that should not be committed here.

Expected voice ASR layout:

```text
smart_care/models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/
```

The sherpa-onnx SenseVoice directory should contain `model.int8.onnx` and
`tokens.txt`. You can also start the voice controller with `--model-dir <path>`
if the model lives elsewhere.

Expected voice TTS layout:

```text
smart_care/models/sherpa-onnx-vits-zh-ll/
```

The sherpa-onnx VITS directory should contain `model.onnx`, `lexicon.txt`,
`tokens.txt`, and the optional normalization FST files such as `phone.fst`,
`date.fst`, and `number.fst`.
