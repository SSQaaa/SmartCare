import queue
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np


DEFAULT_TTS_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "sherpa-onnx-vits-zh-ll"


class SpeechSpeaker:
    def __init__(
        self,
        enabled=True,
        rate=180,
        async_mode=False,
        model_dir=DEFAULT_TTS_MODEL_DIR,
        sid=2,
        num_threads=1,
        provider="cpu",
        tail_silence_ms=350,
    ):
        self.enabled = bool(enabled)
        self.rate = int(rate)
        self.async_mode = bool(async_mode)
        self.model_dir = Path(model_dir)
        self.sid = int(sid)
        self.num_threads = int(num_threads)
        self.provider = str(provider or "cpu")
        self.tail_silence_ms = max(0, int(tail_silence_ms))
        self._queue = queue.Queue()
        self._thread = None
        self._sherpa_tts = None
        self._closed = False
        self._lock = threading.Lock()

        if self.enabled:
            if self.async_mode:
                self._start_worker()
            else:
                self._init_sherpa_tts()

    def _init_sherpa_tts(self):
        if self._sherpa_tts is not None:
            return True
        try:
            import sherpa_onnx
        except ImportError:
            print("缺少 sherpa-onnx，无法使用 sherpa VITS 播报。可安装：python -m pip install sherpa-onnx")
            return False

        model_path = self.model_dir / "model.onnx"
        lexicon_path = self.model_dir / "lexicon.txt"
        tokens_path = self.model_dir / "tokens.txt"
        missing = [path.name for path in (model_path, lexicon_path, tokens_path) if not path.exists()]
        if missing:
            print(f"sherpa VITS 模型目录不完整：{self.model_dir}，缺少 {', '.join(missing)}")
            return False

        rule_fsts = [
            str(path)
            for path in (self.model_dir / "phone.fst", self.model_dir / "date.fst", self.model_dir / "number.fst")
            if path.exists()
        ]

        try:
            tts_config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=str(model_path),
                        lexicon=str(lexicon_path),
                        tokens=str(tokens_path),
                    ),
                    provider=self.provider,
                    num_threads=self.num_threads,
                ),
                rule_fsts=",".join(rule_fsts),
                max_num_sentences=1,
            )
            if not tts_config.validate():
                print("sherpa VITS 配置校验失败，请检查模型目录和 sherpa-onnx 版本。")
                return False
            self._sherpa_tts = sherpa_onnx.OfflineTts(tts_config)
            return True
        except Exception as exc:
            print(f"sherpa VITS 初始化失败：{exc}")
            self._sherpa_tts = None
            return False

    def _start_worker(self):
        self._thread = threading.Thread(target=self._run, name="smartcare-tts", daemon=True)
        self._thread.start()

    def _run(self):
        self._init_sherpa_tts()
        while not self._closed:
            item = self._queue.get()
            if item is None:
                break
            if isinstance(item, tuple) and item and item[0] == "recording":
                _, text, recording_path, cache_path = item
                self._speak_recording_now(text, recording_path, cache_path)
            else:
                self._speak_now(item)

    def _write_wav(self, path, samples, sample_rate):
        samples = np.asarray(samples, dtype=np.float32)
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
        samples = np.clip(samples, -1.0, 1.0)
        tail_samples = int(int(sample_rate) * self.tail_silence_ms / 1000.0)
        if tail_samples > 0:
            samples = np.concatenate([samples, np.zeros(tail_samples, dtype=np.float32)])

        pcm = (samples * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(int(sample_rate))
            handle.writeframes(pcm.tobytes())

    def _play_wav(self, path):
        try:
            import sounddevice as sd

            with wave.open(str(path), "rb") as handle:
                sample_rate = handle.getframerate()
                frames = handle.readframes(handle.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            sd.play(audio, sample_rate)
            sd.wait()
            sd.stop()
            return True
        except Exception:
            pass

        try:
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME)
            return True
        except Exception as exc:
            print(f"播放 wav 失败：{exc}")
            return False

    def _generate_wav(self, text, path):
        if not self._init_sherpa_tts():
            return False
        try:
            speed = max(0.5, min(2.0, self.rate / 180.0))
            audio = self._sherpa_tts.generate(text, sid=self.sid, speed=speed)
            if len(audio.samples) == 0:
                print("sherpa VITS 没有生成音频。")
                return False
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._write_wav(path, audio.samples, audio.sample_rate)
            return True
        except Exception as exc:
            print(f"sherpa VITS 生成缓存音频失败：{exc}")
            return False

    def _speak_now(self, text):
        with self._lock:
            temp_path = Path(tempfile.gettempdir()) / "smartcare_sherpa_tts.wav"
            if self._generate_wav(text, temp_path):
                self._play_wav(temp_path)

    def speak(self, text):
        text = str(text or "").strip()
        if not text:
            return
        print(f"[语音播报] {text}")
        if not self.enabled:
            return
        if self.async_mode and self._thread is not None:
            self._queue.put(text)
        else:
            self._speak_now(text)

    def _speak_recording_now(self, text, recording_path=None, cache_path=None):
        for path in (recording_path, cache_path):
            if path and Path(path).exists() and self._play_wav(path):
                return

        if cache_path:
            with self._lock:
                if self._generate_wav(text, cache_path) and self._play_wav(cache_path):
                    return

        self._speak_now(text)

    def speak_recording(self, text, recording_path=None, cache_path=None):
        text = str(text or "").strip()
        if not text:
            return
        print(f"[语音播报] {text}")
        if not self.enabled:
            return
        if self.async_mode and self._thread is not None:
            self._queue.put(("recording", text, recording_path, cache_path))
        else:
            self._speak_recording_now(text, recording_path, cache_path)

    def close(self):
        self._closed = True
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=2.0)
