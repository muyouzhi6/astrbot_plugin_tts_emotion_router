import asyncio
import io
import json
import wave
import pytest

from astrbot_plugin_tts_emotion_router.core.config import ConfigManager
from astrbot_plugin_tts_emotion_router.tts import cache
from astrbot_plugin_tts_emotion_router.tts.errors import failure
from astrbot_plugin_tts_emotion_router.tts.provider_breeze import BreezeTTS


def _wav_bytes(duration_ms=120, sample_rate=24000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(b"\x01\x00" * round(sample_rate * duration_ms / 1000))
    return buffer.getvalue()


class _FakeResponse:
    def __init__(self, content=b"", status=200, payload=None, headers=None):
        self.status = status
        self.headers = headers if headers is not None else {"Content-Type": "audio/wav"}
        self._content = content
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def read(self):
        return self._content

    async def json(self, content_type=None):
        return self._payload if self._payload is not None else json.loads(self._content)

    async def text(self):
        return self._content.decode("utf-8", errors="replace")


class _FakeSession:
    closed = False

    def __init__(self, response):
        self.response = response
        self.posts = []

    def post(self, url, *, params, headers, json):
        self.posts.append({"url": url, "params": params, "headers": headers, "json": json})
        return self.response

    async def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _fast_decoder(monkeypatch):
    monkeypatch.setattr(cache, "check_decoder", lambda: "ffmpeg")
    monkeypatch.setattr(cache, "decode_duration", lambda *args, **kwargs: 1.0)


def _client(**kwargs):
    options = dict(
        api_url="https://api.breeze.blue/v1",
        api_key="brz_test",
        model="breeze-tts-2",
        voice_id="voc_default",
        fmt="wav",
        speed=1.0,
        sample_rate=24000,
        max_retries=0,
        timeout=30,
    )
    options.update(kwargs)
    return BreezeTTS(**options)


def test_request_contract_and_emotion_instruction(tmp_path):
    session = _FakeSession(_FakeResponse(_wav_bytes()))
    client = _client(guidance_scale=2.0, instructions="像朋友聊天一样自然。")
    client._session = session

    output = asyncio.run(
        client.synth("今天真开心", "voc_abc", tmp_path, speed=1.15, emotion="happy")
    )

    assert output is not None and output.exists()
    request = session.posts[0]
    assert request["url"] == "https://api.breeze.blue/v1/text-to-speech/voc_abc"
    assert request["params"] == {"output_format": "wav"}
    assert request["headers"]["xi-api-key"] == "brz_test"
    assert request["json"]["model_id"] == "breeze-tts-2"
    assert request["json"]["voice_settings"]["speed"] == pytest.approx(1.15)
    assert request["json"]["voice_settings"]["guidance_scale"] == 2.0
    assert "像朋友聊天一样自然。" in request["json"]["instructions"]
    assert "开心" in request["json"]["instructions"]


def test_emotion_instruction_can_be_disabled(tmp_path):
    session = _FakeSession(_FakeResponse(_wav_bytes()))
    client = _client(emotion_instructions_enable=False)
    client._session = session

    assert asyncio.run(client.synth("你好", "voc_abc", tmp_path, emotion="sad"))

    assert "instructions" not in session.posts[0]["json"]


def test_auth_error_is_reported(tmp_path):
    payload = {"ok": False, "code": "AUTH_REQUIRED", "detail": "bad key", "error": "bad key"}
    session = _FakeSession(_FakeResponse(json.dumps(payload).encode(), status=401, headers={}))
    client = _client()
    client._session = session

    async def run():
        assert await client.synth("你好", "voc_abc", tmp_path) is None
        assert failure.get().code == "auth"

    asyncio.run(run())


def test_breeze_config_defaults():
    config = ConfigManager({"tts_engine": {"provider": "breeze"}})
    cfg = config.get_api_config()

    assert config.get_tts_provider() == "breeze"
    assert cfg["url"] == "https://api.breeze.blue/v1"
    assert cfg["model"] == "breeze-tts-2"
    assert cfg["format"] == "mp3"
    assert cfg["sample_rate"] == 24000
    assert cfg["emotion_instructions_enable"] is True
    assert cfg["default_voice"] == ""


def test_breeze_config_validation_rejects_out_of_range():
    from astrbot_plugin_tts_emotion_router.core.diagnostics import configuration_errors

    config = ConfigManager(
        {
            "tts_engine": {
                "provider": "breeze",
                "breeze": {
                    "key": "k",
                    "model": "breeze-tts-2",
                    "default_voice": "voc_abc",
                    "volume": 3,
                    "guidance_scale": 0.5,
                    "language_code": "zh-CN",
                },
            }
        }
    )
    assert config.validation_errors()
    errors = configuration_errors(config.get_api_config())
    assert any("volume" in item for item in errors)
    assert any("guidance_scale" in item for item in errors)
    assert any("language_code" in item for item in errors)
