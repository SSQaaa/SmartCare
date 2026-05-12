import queue
import re
import time

import numpy as np
from pathlib import Path

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09"
DEFAULT_HOTWORDS_FILE = Path(__file__).resolve().parent / "hotwords.txt"

class VoiceDependencyError(RuntimeError):
    pass

class SherpaOnnxRecognizer:
    def __init__(
        self,
        model_dir=DEFAULT_MODEL_DIR,
        sample_rate=16000,
        record_seconds=5.0,
        input_device=None,
        debug_audio=False,
        trim_silence=True,
        hotwords_file=DEFAULT_HOTWORDS_FILE,
        vad_enabled=False,
        vad_min_rms=0.010,
        vad_start_ms=160,
        vad_end_silence_ms=650,
        vad_preroll_ms=300,
    ):
        self.model_dir = Path(model_dir)
        self.sample_rate = int(sample_rate)
        self.record_seconds = float(record_seconds)
        self.input_device = self._normalize_device(input_device)
        self.debug_audio = bool(debug_audio)
        self.trim_silence = bool(trim_silence)
        self.hotwords_file = Path(hotwords_file) if hotwords_file else None
        self.vad_enabled = bool(vad_enabled)
        self.vad_min_rms = float(vad_min_rms)
        self.vad_start_ms = int(vad_start_ms)
        self.vad_end_silence_ms = int(vad_end_silence_ms)
        self.vad_preroll_ms = int(vad_preroll_ms)
        self._recognizer = None
        self._hotword_rules = []

    @staticmethod
    def _normalize_device(device):
        if device is None or device == "":
            return None
        if isinstance(device, str) and device.strip().isdigit():
            return int(device.strip())
        return device

    @staticmethod
    def list_audio_devices():
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise VoiceDependencyError(
                "缺少录音依赖。请先安装：python -m pip install sherpa-onnx sounddevice"
            ) from exc
        return sd.query_devices()

    def load(self):
        if not self.model_dir.exists():
            raise VoiceDependencyError(
                f"没有找到 sherpa-onnx 中文模型。请放到 {self.model_dir}，或用 --model-dir 指定路径。"
            )
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise VoiceDependencyError(
                "缺少 sherpa-onnx。请先安装：python -m pip install sherpa-onnx sounddevice"
            ) from exc
            
        if self._recognizer is None:
            model_path = self.model_dir / "model.int8.onnx"
            tokens_path = self.model_dir / "tokens.txt"
            if not model_path.exists() or not tokens_path.exists():
                raise VoiceDependencyError(
                    "sherpa-onnx 模型目录不完整，需要包含 model.int8.onnx 和 tokens.txt。"
                )
            # 新版 SenseVoice 正确初始化方式
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(model_path),
                tokens=str(tokens_path),
                num_threads=1,
                language="zh",
                use_itn=True,
            )
        return self._recognizer

    def load_hotwords(self):
        if self._hotword_rules or self.hotwords_file is None or not self.hotwords_file.exists():
            return self._hotword_rules

        rules = []
        with open(self.hotwords_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    canonical, aliases = line.split("=", 1)
                    alias_list = [part.strip() for part in aliases.replace("，", ",").split(",")]
                else:
                    canonical, alias_list = line, []
                canonical = canonical.strip()
                words = [canonical] + [alias for alias in alias_list if alias]
                for word in words:
                    compact_word = self._compact_text(word)
                    if canonical and compact_word:
                        rules.append((canonical, compact_word))
        self._hotword_rules = rules
        return self._hotword_rules

    @staticmethod
    def _compact_text(text):
        punctuation = "，。！？；：、,.!?;:（）()[]【】“”\"' "
        compact = "".join(str(text or "").strip().lower().split())
        return "".join(ch for ch in compact if ch not in punctuation)

    def _apply_hotwords(self, text):
        text = str(text or "").strip()
        compact = self._compact_text(text)
        if not compact:
            return text
        for canonical, alias in self.load_hotwords():
            if compact == alias:
                return canonical
        for canonical, alias in self.load_hotwords():
            if alias in compact and canonical not in text:
                return compact.replace(alias, self._compact_text(canonical))
        return text

    def _preprocess_audio(self, audio):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)

        if self.trim_silence and audio.size:
            abs_audio = np.abs(audio)
            peak = float(abs_audio.max(initial=0.0))
            if peak > 0.0:
                threshold = max(0.006, peak * 0.08)
                active = np.flatnonzero(abs_audio > threshold)
                if active.size:
                    padding = int(0.20 * self.sample_rate)
                    start = max(0, int(active[0]) - padding)
                    end = min(audio.size, int(active[-1]) + padding)
                    audio = audio[start:end]

        return audio

    def _print_audio_stats(self, audio):
        if not self.debug_audio:
            return
        if audio.size == 0:
            print("[录音诊断] 没有采集到音频。")
            return
        rms = float(np.sqrt(np.mean(np.square(audio))))
        peak = float(np.max(np.abs(audio)))
        duration = audio.size / float(self.sample_rate)
        device = "系统默认输入" if self.input_device is None else self.input_device
        print(f"[录音诊断] device={device}, duration={duration:.2f}s, rms={rms:.4f}, peak={peak:.4f}")
        if peak < 0.02 or rms < 0.003:
            print("[录音诊断] 音量很小，可能录错设备、麦克风被静音，或离麦克风太远。")
        elif peak > 0.98:
            print("[录音诊断] 音频接近爆音，建议降低麦克风输入音量。")

    def _decode_audio(self, audio):
        recognizer = self.load()
        audio = self._preprocess_audio(audio)
        self._print_audio_stats(audio)

        if audio.size == 0:
            return ""

        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate=self.sample_rate, waveform=audio)
        recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        text = re.sub(r"<\|[^|]+?\|>", "", text).strip()
        return self._apply_hotwords(text)

    def _open_input_stream(self, audio_queue, block_size):
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise VoiceDependencyError(
                "缺少录音依赖。请先安装：python -m pip install sherpa-onnx sounddevice"
            ) from exc

        def callback(indata, _frames, _time, status):
            if status:
                print(status)
            audio_queue.put(indata.copy())

        return sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=block_size,
            dtype="float32",
            channels=1,
            device=self.input_device,
            callback=callback,
        )

    def _is_voice_block(self, block, noise_rms):
        rms = float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0
        threshold = max(self.vad_min_rms, noise_rms * 3.0)
        return rms >= threshold, rms

    def _listen_once_vad(self, record_seconds):
        self.load()
        audio_queue = queue.Queue()
        block_size = max(400, int(self.sample_rate * 0.10))
        max_blocks = max(1, int(record_seconds * self.sample_rate / block_size))
        block_ms = 100.0
        start_blocks = max(1, int(np.ceil(self.vad_start_ms / block_ms)))
        silence_blocks = max(1, int(np.ceil(self.vad_end_silence_ms / block_ms)))
        preroll_blocks = max(1, int(np.ceil(self.vad_preroll_ms / block_ms)))
        preroll = []
        chunks = []
        speech_blocks = 0
        trailing_silence = 0
        started = False
        noise_rms = self.vad_min_rms * 0.5

        print(f"开始 VAD 监听，请说话。最长等待 {record_seconds:.1f} 秒...")
        start_time = time.time()
        with self._open_input_stream(audio_queue, block_size):
            for i in range(max_blocks):
                try:
                    data = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    if time.time() - start_time >= record_seconds:
                        break
                    continue
                block = np.asarray(data[:, 0], dtype=np.float32)
                is_voice, rms = self._is_voice_block(block, noise_rms)

                if not started:
                    if not is_voice:
                        noise_rms = 0.95 * noise_rms + 0.05 * rms
                        preroll.append(block)
                        preroll = preroll[-preroll_blocks:]
                        speech_blocks = 0
                        continue
                    speech_blocks += 1
                    preroll.append(block)
                    preroll = preroll[-preroll_blocks:]
                    if speech_blocks >= start_blocks:
                        started = True
                        chunks.extend(preroll)
                        trailing_silence = 0
                    continue

                chunks.append(block)
                if is_voice:
                    trailing_silence = 0
                else:
                    trailing_silence += 1
                    if trailing_silence >= silence_blocks:
                        break

        if not chunks:
            return ""
        audio = np.concatenate(chunks).astype(np.float32)
        return self._decode_audio(audio)

    def listen_once(self, record_seconds=None, use_vad=False):
        self.load()
        record_seconds = self.record_seconds if record_seconds is None else float(record_seconds)
        use_vad = self.vad_enabled if use_vad is None else bool(use_vad)
        if use_vad:
            return self._listen_once_vad(record_seconds)

        audio_queue = queue.Queue()
        block_size = 8000
        blocks = max(1, int(record_seconds * self.sample_rate / block_size))
        print(f"开始录音，请说话。{record_seconds:.1f} 秒后自动识别...")

        audio = np.zeros((blocks * block_size,), dtype=np.float32)
        idx = 0
        with self._open_input_stream(audio_queue, block_size):
            for _ in range(blocks):
                data = audio_queue.get()
                audio[idx:idx+len(data)] = data[:, 0]
                idx += len(data)

        return self._decode_audio(audio[:idx])

    def listen_until_enter(self):
        self.load()
        audio_queue = queue.Queue()
        block_size = 4000
        chunks = []
        print("开始录音，请说话。按 Enter 结束录音并识别...")
        with self._open_input_stream(audio_queue, block_size):
            input()
            while not audio_queue.empty():
                chunks.append(audio_queue.get())

        if not chunks:
            return ""
        audio = np.concatenate([chunk[:, 0] for chunk in chunks]).astype(np.float32)
        return self._decode_audio(audio)

        return re.sub(r"<\|[^|]+?\|>", "", text).strip()
