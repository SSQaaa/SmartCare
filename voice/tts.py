import queue
import subprocess
import sys
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
        engine="sherpa",
        model_dir=DEFAULT_TTS_MODEL_DIR,
        sid=2,
        num_threads=1,
        provider="cpu",
        tail_silence_ms=350,
    ):
        self.enabled = bool(enabled)
        self.rate = int(rate)
        self.async_mode = bool(async_mode)
        self.engine = str(engine or "sherpa").lower()
        self.model_dir = Path(model_dir)
        self.sid = int(sid)
        self.num_threads = int(num_threads)
        self.provider = str(provider or "cpu")
        self.tail_silence_ms = max(0, int(tail_silence_ms))
        self._queue = queue.Queue()
        self._thread = None
        self._engine = None
        self._sherpa_tts = None
        self._closed = False
        self._lock = threading.Lock()

        if self.enabled:
            if self.async_mode:
                self._start_worker()
            elif self.engine == "sherpa":
                self._init_sherpa_tts()
            elif self.engine == "pyttsx3":
                self._init_engine()

    def _init_engine(self):
        if self._engine is not None:
            return
        try:
            import pyttsx3
        except ImportError:
            if self.engine == "pyttsx3":
                print("缺少 pyttsx3，无法使用 pyttsx3 播报。可安装：python -m pip install pyttsx3")
            return
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate)
        except Exception as exc:
            print(f"pyttsx3 初始化失败，将使用系统语音兜底：{exc}")
            self._engine = None

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
        if self.engine == "sherpa":
            self._init_sherpa_tts()
        elif self.engine == "pyttsx3":
            self._init_engine()
        while not self._closed:
            text = self._queue.get()
            if text is None:
                break
            self._speak_now(text)

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

        if sys.platform == "win32":
            try:
                import winsound

                winsound.PlaySound(str(path), winsound.SND_FILENAME)
                return True
            except Exception as exc:
                print(f"播放 wav 失败：{exc}")
        return False

    def _speak_with_sherpa(self, text):
        if not self._init_sherpa_tts():
            return False
        try:
            speed = max(0.5, min(2.0, self.rate / 180.0))
            audio = self._sherpa_tts.generate(text, sid=self.sid, speed=speed)
            if len(audio.samples) == 0:
                print("sherpa VITS 没有生成音频。")
                return False
            temp_path = Path(tempfile.gettempdir()) / "smartcare_sherpa_tts.wav"
            self._write_wav(temp_path, audio.samples, audio.sample_rate)
            return self._play_wav(temp_path)
        except Exception as exc:
            print(f"sherpa VITS 播报失败：{exc}")
            return False

    def _speak_with_system_speech(self, text):
        if sys.platform != "win32":
            return False
        voice_rate = max(-10, min(10, int((self.rate - 180) / 20)))
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
                handle.write(text)
                temp_path = Path(handle.name)
            safe_path = str(temp_path).replace("'", "''")
            script = (
                f"$text = Get-Content -LiteralPath '{safe_path}' -Raw -Encoding UTF8; "
                "try { "
                "$v = New-Object -ComObject SAPI.SpVoice; "
                f"$v.Rate = {voice_rate}; "
                "$v.Volume = 100; "
                "[void]$v.Speak($text); "
                "} catch { "
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.Rate = {voice_rate}; "
                "$s.Volume = 100; "
                "$s.Speak($text); "
                "}"
            )
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if completed.returncode == 0:
                return True
            if completed.stderr:
                print(f"Windows 系统语音返回错误：{completed.stderr.strip()}")
            return False
        except Exception as exc:
            print(f"Windows 系统语音兜底失败：{exc}")
            return False
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def _speak_now(self, text):
        with self._lock:
            if self.engine == "sherpa" and self._speak_with_sherpa(text):
                return
            if self._engine is not None:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                    return
                except Exception as exc:
                    print(f"pyttsx3 播报失败，将使用系统语音兜底：{exc}")
            if self._speak_with_system_speech(text):
                return
            if self.engine != "pyttsx3":
                self._init_engine()
                if self._engine is not None:
                    try:
                        self._engine.say(text)
                        self._engine.runAndWait()
                        return
                    except Exception as exc:
                        print(f"pyttsx3 兜底播报失败：{exc}")
            print("语音引擎不可用，只能显示文字。安装：python -m pip install pyttsx3，并确认 Windows 音量和输出设备。")

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

    def close(self):
        self._closed = True
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=2.0)
