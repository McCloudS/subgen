"""
Tests for the whisper.cpp backend: backend selection, CLI arguments, device logging,
language detection and VAD. whisper-cli is mocked unless WHISPER_CPP_TEST_CLI and
WHISPER_CPP_TEST_MODEL point at a real binary and model (optional integration test).
"""
import io
import json
import logging
import os
import wave
from unittest.mock import MagicMock

import pytest

import subgen


def _fake_run_writing(json_payload, stderr="", returncode=0):
    """subprocess.run stand-in that writes `json_payload` to the -of prefix + .json."""
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        prefix = cmd[cmd.index('-of') + 1]
        with open(prefix + '.json', 'w', encoding='utf-8') as f:
            json.dump(json_payload, f)
        return MagicMock(returncode=returncode, stderr=stderr, stdout="")

    run.calls = calls
    return run


@pytest.fixture
def wcp(monkeypatch, tmp_path):
    model = tmp_path / "ggml-small.bin"
    model.write_bytes(b"x")
    monkeypatch.setattr(subgen, "transcribe_backend", "whispercpp")
    monkeypatch.setattr(subgen, "_ensure_whispercpp_model", lambda: str(model))
    monkeypatch.setattr(subgen, "whisper_cpp_device", "")
    monkeypatch.setattr(subgen, "vad_filter", False)
    monkeypatch.setattr(subgen, "_whispercpp_last_device", None)
    return tmp_path


class TestBackendResolution:
    def test_auto_prefers_whispercpp_when_cli_present(self, monkeypatch):
        monkeypatch.setattr(subgen.shutil, "which", lambda p: "/usr/bin/whisper-cli")
        assert subgen.resolve_transcribe_backend("auto", "whisper-cli") == "whispercpp"

    def test_auto_falls_back_to_faster_whisper(self, monkeypatch):
        monkeypatch.setattr(subgen.shutil, "which", lambda p: None)
        assert subgen.resolve_transcribe_backend("auto", "whisper-cli") == "faster-whisper"

    @pytest.mark.parametrize("value", ["whispercpp", "whisper.cpp", "WHISPER-CPP"])
    def test_whispercpp_aliases(self, value):
        assert subgen.resolve_transcribe_backend(value, "x") == "whispercpp"

    def test_explicit_faster_whisper_wins_even_with_cli(self, monkeypatch):
        monkeypatch.setattr(subgen.shutil, "which", lambda p: "/usr/bin/whisper-cli")
        assert subgen.resolve_transcribe_backend("faster-whisper", "whisper-cli") == "faster-whisper"

    def test_unknown_value_uses_auto(self, monkeypatch):
        monkeypatch.setattr(subgen.shutil, "which", lambda p: None)
        assert subgen.resolve_transcribe_backend("bogus", "whisper-cli") == "faster-whisper"


class TestDeviceArgs:
    def test_default_lets_whispercpp_use_gpu(self, wcp, monkeypatch):
        # TRANSCRIBE_DEVICE=cpu is common in existing compose files; it must not disable whisper.cpp's GPU
        monkeypatch.setattr(subgen, "transcribe_device", "cpu")
        assert subgen._whispercpp_device_args() == []

    def test_cpu_opt_out(self, wcp, monkeypatch):
        monkeypatch.setattr(subgen, "whisper_cpp_device", "cpu")
        assert subgen._whispercpp_device_args() == ["-ng"]

    def test_device_index(self, wcp, monkeypatch):
        monkeypatch.setattr(subgen, "whisper_cpp_device", "1")
        assert subgen._whispercpp_device_args() == ["-dev", "1"]

    def test_invalid_device_ignored(self, wcp, monkeypatch):
        # a non-integer reaches std::stoi in whisper-cli and aborts every job
        monkeypatch.setattr(subgen, "whisper_cpp_device", "nvidia")
        assert subgen._whispercpp_device_args() == []


class TestTranscribeCommand:
    def test_unknown_language_uses_auto_not_english(self, wcp, monkeypatch):
        run = _fake_run_writing({"result": {"language": "fr"}, "transcription": []})
        monkeypatch.setattr(subgen.subprocess, "run", run)
        result = subgen._transcribe_whispercpp(b"RIFF", "transcribe", "", "x.mkv")
        cmd = run.calls[0]
        assert cmd[cmd.index("-l") + 1] == "auto"
        assert result.language == "fr"

    def test_known_language_and_translate(self, wcp, monkeypatch):
        run = _fake_run_writing({"result": {"language": "de"}, "transcription": []})
        monkeypatch.setattr(subgen.subprocess, "run", run)
        subgen._transcribe_whispercpp(b"RIFF", "translate", "de", "x.mkv")
        cmd = run.calls[0]
        assert cmd[cmd.index("-l") + 1] == "de"
        assert "--translate" in cmd

    def test_no_previous_text_context(self, wcp, monkeypatch):
        """-mc 0 (like condition_on_previous_text=False) prevents repetition loops."""
        run = _fake_run_writing({"result": {"language": "en"}, "transcription": []})
        monkeypatch.setattr(subgen.subprocess, "run", run)
        subgen._transcribe_whispercpp(b"RIFF", "transcribe", "en", "x.mkv")
        cmd = run.calls[0]
        assert cmd[cmd.index("-mc") + 1] == "0"

    def test_vad_uses_real_flags(self, wcp, monkeypatch):
        vad = wcp / "ggml-silero-v5.1.2.bin"
        vad.write_bytes(b"x")
        monkeypatch.setattr(subgen, "vad_filter", True)
        monkeypatch.setattr(subgen, "vad_threshold", 0.6)
        monkeypatch.setattr(subgen, "whisper_cpp_vad_model", str(vad))
        run = _fake_run_writing({"result": {"language": "en"}, "transcription": []})
        monkeypatch.setattr(subgen.subprocess, "run", run)
        subgen._transcribe_whispercpp(b"RIFF", "transcribe", "en", "x.mkv")
        cmd = run.calls[0]
        assert "--vad-thold" not in cmd  # not a whisper-cli flag; it made whisper-cli exit without output
        assert cmd[cmd.index("--vad"):cmd.index("--vad") + 5] == ["--vad", "-vm", str(vad), "-vt", "0.6"]

    def test_vad_skipped_when_model_unavailable(self, wcp, monkeypatch):
        monkeypatch.setattr(subgen, "vad_filter", True)
        monkeypatch.setattr(subgen, "_ensure_whispercpp_vad_model", lambda: "")
        run = _fake_run_writing({"result": {"language": "en"}, "transcription": []})
        monkeypatch.setattr(subgen.subprocess, "run", run)
        subgen._transcribe_whispercpp(b"RIFF", "transcribe", "en", "x.mkv")
        assert "--vad" not in run.calls[0]

    def test_nonzero_exit_raises_with_stderr_tail(self, wcp, monkeypatch):
        run = _fake_run_writing({}, stderr="lots of logs\nfatal: boom", returncode=3)
        monkeypatch.setattr(subgen.subprocess, "run", run)
        with pytest.raises(RuntimeError, match="boom"):
            subgen._transcribe_whispercpp(b"RIFF", "transcribe", "en", "x.mkv")


class TestDeviceLogging:
    def test_logs_gpu_once(self, wcp, caplog):
        stderr = "whisper_backend_init_gpu: using Vulkan0 backend\n"
        with caplog.at_level(logging.INFO):
            subgen._log_whispercpp_device(stderr)
            subgen._log_whispercpp_device(stderr)
        assert [r.getMessage() for r in caplog.records].count("whisper.cpp is using GPU device: Vulkan0") == 1

    def test_warns_on_silent_cpu_fallback(self, wcp, caplog):
        with caplog.at_level(logging.WARNING):
            subgen._log_whispercpp_device("whisper_backend_init_gpu: no GPU found\n")
        assert any("no usable GPU" in r.getMessage() for r in caplog.records)

    def test_cpu_image_does_not_warn(self, wcp, monkeypatch, caplog):
        monkeypatch.setenv("GGML_BACKEND", "cpu")
        with caplog.at_level(logging.WARNING):
            subgen._log_whispercpp_device("whisper_backend_init_gpu: no GPU found\n")
        assert not any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_failed_gpu_init_reported_as_cpu(self, wcp, caplog):
        stderr = ("whisper_backend_init_gpu: using Vulkan0 backend\n"
                  "whisper_backend_init_gpu: failed to initialize Vulkan0 backend\n")
        with caplog.at_level(logging.WARNING):
            subgen._log_whispercpp_device(stderr)
        assert any("no usable GPU" in r.getMessage() for r in caplog.records)

    def test_no_warning_when_cpu_requested(self, wcp, monkeypatch, caplog):
        monkeypatch.setattr(subgen, "whisper_cpp_device", "cpu")
        with caplog.at_level(logging.WARNING):
            subgen._log_whispercpp_device("whisper_backend_init_gpu: no GPU found\n")
        assert not any(r.levelno >= logging.WARNING for r in caplog.records)


class TestLanguageDetection:
    def test_reads_language_from_json(self, wcp, monkeypatch):
        run = _fake_run_writing({"result": {"language": "ja"}, "transcription": []})
        monkeypatch.setattr(subgen.subprocess, "run", run)
        assert subgen.detect_language_whispercpp(b"RIFF") == "ja"
        assert "-dl" in run.calls[0] and "-oj" in run.calls[0]

    def test_missing_language_raises(self, wcp, monkeypatch):
        monkeypatch.setattr(subgen.subprocess, "run", _fake_run_writing({"result": {}}))
        with pytest.raises(RuntimeError):
            subgen.detect_language_whispercpp(b"RIFF")

    def test_timeout_never_below_previous_3600(self, wcp, monkeypatch):
        seen = {}
        def run(cmd, **kw):
            seen.update(kw)
            with open(cmd[cmd.index('-of') + 1] + '.json', 'w') as f:
                json.dump({"result": {"language": "en"}, "transcription": []}, f)
            return MagicMock(returncode=0, stderr="")
        monkeypatch.setattr(subgen, "asr_timeout", 900)
        monkeypatch.setattr(subgen.subprocess, "run", run)
        subgen._transcribe_whispercpp(b"RIFF", "transcribe", "en", "x.mkv")
        assert seen["timeout"] == 3600

    @pytest.mark.parametrize("name", ["ggml-base.en.bin", "ggml-small.en-q5_1.bin"])
    def test_english_only_model_short_circuits(self, wcp, monkeypatch, name):
        monkeypatch.setattr(subgen, "_ensure_whispercpp_model", lambda: "/m/" + name)
        run = MagicMock()
        monkeypatch.setattr(subgen.subprocess, "run", run)
        assert subgen.detect_language_whispercpp(b"RIFF") == "en"
        run.assert_not_called()

    @pytest.mark.parametrize("name", ["ggml-large-v3-turbo.bin", "ggml-model.bin", "ggml-distil-large-v3.fp32.bin"])
    def test_multilingual_models_run_detection(self, wcp, monkeypatch, name):
        monkeypatch.setattr(subgen, "_ensure_whispercpp_model", lambda: "/m/" + name)
        monkeypatch.setattr(subgen.subprocess, "run", _fake_run_writing({"result": {"language": "it"}}))
        assert subgen.detect_language_whispercpp(b"RIFF") == "it"

    def test_file_detection_queues_transcription(self, wcp, monkeypatch):
        """Previously whisper.cpp returned early here and the file was never transcribed."""
        monkeypatch.setattr(subgen, "start_model", MagicMock())
        monkeypatch.setattr(subgen, "delete_model", MagicMock())
        monkeypatch.setattr(subgen, "extract_audio_segment_to_memory", MagicMock(return_value=b"WAV"))
        monkeypatch.setattr(subgen, "detect_language_whispercpp", MagicMock(return_value="es"))
        result = subgen.detect_language_task("/media/show.mkv")
        task = result if isinstance(result, dict) else None
        assert task is not None and task["force_language"] == subgen.LanguageCode.from_string("es")

    def test_pcm16_to_wav_roundtrip(self):
        pcm = b"\x01\x00\xff\x7f" * 8
        with wave.open(io.BytesIO(subgen.pcm16_to_wav_bytes(pcm))) as wf:
            assert (wf.getnchannels(), wf.getsampwidth(), wf.getframerate()) == (1, 2, 16000)
            assert wf.readframes(wf.getnframes()) == pcm


@pytest.mark.skipif(not (os.getenv("WHISPER_CPP_TEST_CLI") and os.getenv("WHISPER_CPP_TEST_MODEL")),
                    reason="set WHISPER_CPP_TEST_CLI and WHISPER_CPP_TEST_MODEL to run against a real whisper-cli")
class TestRealWhisperCli:
    @pytest.fixture
    def real(self, monkeypatch):
        monkeypatch.setattr(subgen, "whisper_cli_path", os.environ["WHISPER_CPP_TEST_CLI"])
        monkeypatch.setattr(subgen, "_ensure_whispercpp_model", lambda: os.environ["WHISPER_CPP_TEST_MODEL"])
        monkeypatch.setattr(subgen, "whisper_cpp_device", "")
        monkeypatch.setattr(subgen, "_whispercpp_last_device", None)
        sample = os.getenv("WHISPER_CPP_TEST_AUDIO")
        if not sample:
            pytest.skip("set WHISPER_CPP_TEST_AUDIO to a speech WAV")
        with open(sample, "rb") as f:
            return f.read()

    def test_detect_language(self, real):
        assert subgen.detect_language_whispercpp(real) == "en"

    def test_transcribe_auto_language_with_vad(self, real, monkeypatch):
        vad = os.getenv("WHISPER_CPP_TEST_VAD_MODEL")
        monkeypatch.setattr(subgen, "vad_filter", bool(vad))
        monkeypatch.setattr(subgen, "whisper_cpp_vad_model", vad or "")
        result = subgen._transcribe_whispercpp(real, "transcribe", "", "jfk.wav")
        assert result.language == "en"
        assert result.segments and "country" in " ".join(s["text"] for s in result.segments).lower()
