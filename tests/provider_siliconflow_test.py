# ruff: noqa: E402

import asyncio
import io
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

# Match the plugin entrypoint's import order before importing the provider.
import astrbot_plugin_tts_emotion_router.core  # noqa: F401

from astrbot_plugin_tts_emotion_router.tts.provider_siliconflow import (
    COSYVOICE_TAIL_PADDING_MS,
    SiliconFlowTTS,
)


def _wav_bytes(duration_ms=120, sample_rate=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x01\x00" * round(sample_rate * duration_ms / 1000))
    return buffer.getvalue()


class _FakeResponse:
    def __init__(self, content, delay=0):
        self.status = 200
        self.headers = {"Content-Type": "audio/wav"}
        self._content = content
        self._delay = delay

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def read(self):
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._content

    async def json(self):
        return {}

    async def text(self):
        return ""


class _FakeSession:
    def __init__(self, content, delay=0):
        self.closed = False
        self.content = content
        self.delay = delay
        self.posts = []

    def post(self, url, *, headers, json):
        self.posts.append({"url": url, "headers": headers, "json": json})
        return _FakeResponse(self.content, self.delay)

    async def close(self):
        self.closed = True


class SiliconFlowTTSTests(unittest.IsolatedAsyncioTestCase):
    def _client(self, session, model="FunAudioLLM/CosyVoice2-0.5B"):
        client = SiliconFlowTTS(
            "https://api.siliconflow.cn/v1",
            "test-key",
            model,
            fmt="wav",
            speed=1.0,
            gain=0,
            sample_rate=16000,
            max_retries=0,
        )
        client._session = session
        return client

    async def test_cosyvoice_adds_terminal_punctuation_and_tail_silence(self):
        session = _FakeSession(_wav_bytes())
        client = self._client(session)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = await client.synth(
                "妈妈说：“开心啊”",
                "voice-id",
                Path(temp_dir),
            )

            self.assertIsNotNone(output)
            self.assertEqual(session.posts[0]["json"]["input"], "妈妈说：“开心啊。”")
            self.assertIs(session.posts[0]["json"]["stream"], False)
            with wave.open(str(output), "rb") as wav_file:
                expected_frames = round(
                    wav_file.getframerate() * (120 + COSYVOICE_TAIL_PADDING_MS) / 1000
                )
                self.assertEqual(wav_file.getnframes(), expected_frames)
                tail_frames = round(
                    wav_file.getframerate() * COSYVOICE_TAIL_PADDING_MS / 1000
                )
                wav_file.setpos(wav_file.getnframes() - tail_frames)
                self.assertEqual(
                    wav_file.readframes(tail_frames),
                    b"\x00\x00" * tail_frames,
                )

    async def test_same_key_concurrent_requests_share_one_generation(self):
        session = _FakeSession(_wav_bytes(), delay=0.05)
        client = self._client(session)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            first, second = await asyncio.gather(
                client.synth("并发缓存测试", "voice-id", output_dir),
                client.synth("并发缓存测试", "voice-id", output_dir),
            )

            self.assertEqual(first, second)
            self.assertEqual(len(session.posts), 1)
            self.assertEqual(list(output_dir.glob(".*.raw.*")), [])
            self.assertEqual(list(output_dir.glob(".*.padded.*")), [])

    async def test_nonzero_invalid_cache_is_replaced(self):
        session = _FakeSession(_wav_bytes())
        client = self._client(session)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            first = await client.synth("坏缓存测试", "voice-id", output_dir)
            self.assertIsNotNone(first)
            Path(first).write_bytes(b"not-a-valid-wave-file" * 10)

            second = await client.synth("坏缓存测试", "voice-id", output_dir)

            self.assertEqual(first, second)
            self.assertEqual(len(session.posts), 2)
            with wave.open(str(second), "rb") as wav_file:
                self.assertGreater(wav_file.getnframes(), 0)

    async def test_non_cosyvoice_keeps_text_and_does_not_pad(self):
        session = _FakeSession(_wav_bytes())
        client = self._client(session, model="gpt-tts-pro")

        with tempfile.TemporaryDirectory() as temp_dir:
            output = await client.synth("  保持原文  ", "voice-id", Path(temp_dir))

            self.assertIsNotNone(output)
            self.assertEqual(session.posts[0]["json"]["input"], "  保持原文  ")
            with wave.open(str(output), "rb") as wav_file:
                self.assertEqual(wav_file.getnframes(), round(16000 * 0.12))

    async def test_padding_failure_keeps_valid_original_audio(self):
        session = _FakeSession(_wav_bytes())
        client = self._client(session)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "astrbot_plugin_tts_emotion_router.tts.provider_siliconflow.append_audio_silence",
                side_effect=RuntimeError("ffmpeg unavailable"),
            ):
                output = await client.synth(
                    "补静音失败回退测试",
                    "voice-id",
                    Path(temp_dir),
                )

            self.assertIsNotNone(output)
            with wave.open(str(output), "rb") as wav_file:
                self.assertEqual(wav_file.getnframes(), round(16000 * 0.12))


if __name__ == "__main__":
    unittest.main()
