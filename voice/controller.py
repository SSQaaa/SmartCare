from .sherpa_asr import DEFAULT_MODEL_DIR, SherpaOnnxRecognizer, VoiceDependencyError
from .commands import HELP_TEXT, parse_command
from .tts import DEFAULT_TTS_MODEL_DIR, SpeechSpeaker
from .vision_actions import ActionResult, VisionActions


class VoiceController:
    def __init__(
        self,
        model_dir=DEFAULT_MODEL_DIR,
        record_seconds=5.0,
        sample_rate=16000,
        tts_enabled=True,
        tts_rate=180,
        tts_engine="sapi",
        tts_model_dir=DEFAULT_TTS_MODEL_DIR,
        tts_sid=2,
        tts_num_threads=1,
        tts_provider="cpu",
        input_device=None,
        debug_audio=False,
        vl_backend="qwen",
        vl_model="",
        vl_fallback_model="",
        fall_mode="multimodal",
    ):
        self.recognizer = SherpaOnnxRecognizer(
            model_dir=model_dir,
            sample_rate=sample_rate,
            record_seconds=record_seconds,
            input_device=input_device,
            debug_audio=debug_audio,
        )
        self.speaker = SpeechSpeaker(
            enabled=tts_enabled,
            rate=tts_rate,
            engine=tts_engine,
            model_dir=tts_model_dir,
            sid=tts_sid,
            num_threads=tts_num_threads,
            provider=tts_provider,
        )
        self.actions = VisionActions(
            vl_backend=vl_backend,
            vl_model=vl_model,
            vl_fallback_model=vl_fallback_model,
            fall_mode=fall_mode,
        )
        self.running = True

    def close(self):
        self.actions.close()
        self.speaker.close()

    def speak(self, text):
        self.speaker.speak(text)

    def dispatch(self, text):
        command = parse_command(text)
        print(f"[识别文本] {text}")
        print(f"[解析命令] {command}")

        try:
            result = self._execute(command)
        except Exception as exc:
            result = ActionResult(False, f"执行失败：{exc}", repr(exc))

        self.speak(result.spoken_text)
        if result.debug_text:
            print(f"[调试] {result.debug_text}")
        return result

    def _execute(self, command):
        if command.intent == "exit":
            self.running = False
            return ActionResult(True, "好的，语音总控已退出。")
        if command.intent == "help":
            return ActionResult(True, HELP_TEXT)
        if command.intent == "stop_current":
            return self.actions.stop_current()
        if command.intent == "describe_scene":
            return self.actions.describe_scene()
        if command.intent == "find_object":
            return self.actions.find_object(command.target)
        if command.intent == "summarize_objects":
            return self.actions.summarize_objects()
        if command.intent == "recognize_face":
            return self.actions.recognize_face()
        if command.intent == "start_fall_monitor":
            return self.actions.start_fall_monitor()
        if command.intent == "stop_fall_monitor":
            return self.actions.stop_fall_monitor()
        if command.intent == "fall_status":
            return self.actions.fall_status()
        return ActionResult(False, "我还没听懂这个指令。" + HELP_TEXT)

    def run(self):
        print("SmartCare 语音总控")
        print("按 Enter 开始录音，再按 Enter 结束录音；输入 Q 退出。")
        try:
            self.recognizer.load()
        except VoiceDependencyError as exc:
            self.speak(str(exc))
            return
        self.speak("你好，我是你的智能管家，请问有什么可以帮您？")
        while self.running:
            value = input("> ").strip()
            if value.lower() == "q":
                self.dispatch("退出")
                break
            try:
                text = self.recognizer.listen_until_enter()
            except VoiceDependencyError as exc:
                self.speak(str(exc))
                break
            if not text:
                self.speak("我没有听清楚，请再说一次。")
                continue
            self.dispatch(text)

    def run_text_once(self, text):
        return self.dispatch(text)
