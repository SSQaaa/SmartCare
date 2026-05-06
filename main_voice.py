import argparse
import sys

from config import VL_BACKEND, VL_FALLBACK_MODEL_PATH, VL_MODEL_PATH
from voice.sherpa_asr import DEFAULT_MODEL_DIR, SherpaOnnxRecognizer, VoiceDependencyError
from voice.controller import VoiceController
from voice.tts import DEFAULT_TTS_MODEL_DIR


def build_parser():
    parser = argparse.ArgumentParser(description="SmartCare voice controller.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="sherpa-onnx SenseVoice model directory.")
    parser.add_argument("--record-seconds", type=float, default=5.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--input-device", default=None, help="sounddevice input device index or name.")
    parser.add_argument("--list-audio-devices", action="store_true", help="List available audio devices and exit.")
    parser.add_argument("--debug-audio", action="store_true", help="Print microphone volume diagnostics after each recording.")
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--tts-rate", type=int, default=180)
    parser.add_argument("--tts-model-dir", default=str(DEFAULT_TTS_MODEL_DIR), help="sherpa-onnx VITS model directory.")
    parser.add_argument("--tts-sid", type=int, default=2, help="sherpa VITS speaker id.")
    parser.add_argument("--tts-num-threads", type=int, default=1)
    parser.add_argument("--tts-provider", default="cpu")
    parser.add_argument("--vl-backend", choices=("qwen", "florence", "opencv"), default=VL_BACKEND)
    parser.add_argument("--vl-model", default=VL_MODEL_PATH)
    parser.add_argument("--vl-fallback-model", default=VL_FALLBACK_MODEL_PATH)
    parser.add_argument("--fall-mode", choices=("static", "dynamic", "multimodal"), default="multimodal")
    parser.add_argument("--text", default="", help="Debug mode: execute this text command without microphone ASR.")
    parser.add_argument("--test-tts", action="store_true", help="Speak a short test sentence and exit.")
    return parser


def main():
    args = build_parser().parse_args()
    if args.list_audio_devices:
        try:
            print(SherpaOnnxRecognizer.list_audio_devices())
        except VoiceDependencyError as exc:
            print(exc)
        return
    controller = VoiceController(
        model_dir=args.model_dir,
        record_seconds=args.record_seconds,
        sample_rate=args.sample_rate,
        input_device=args.input_device,
        debug_audio=args.debug_audio,
        tts_enabled=not args.no_tts,
        tts_rate=args.tts_rate,
        tts_model_dir=args.tts_model_dir,
        tts_sid=args.tts_sid,
        tts_num_threads=args.tts_num_threads,
        tts_provider=args.tts_provider,
        vl_backend=args.vl_backend,
        vl_model=args.vl_model,
        vl_fallback_model=args.vl_fallback_model,
        fall_mode=args.fall_mode,
    )
    try:
        if args.test_tts:
            controller.speak("语音播报测试。你好，我是你的智能管家。")
            return
        if args.text:
            controller.run_text_once(args.text)
        else:
            controller.run()
    finally:
        controller.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
