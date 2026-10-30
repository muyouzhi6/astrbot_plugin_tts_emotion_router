"""Load the complete plugin against an isolated host API, without AST extraction."""
import asyncio
import importlib
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from astrbot_plugin_tts_emotion_router.core import diagnostics
from astrbot_plugin_tts_emotion_router.tts.cache import audio_cache
from astrbot_plugin_tts_emotion_router.tts.errors import failure


def test_version_consistency():
    from astrbot_plugin_tts_emotion_router.core.constants import PLUGIN_VERSION
    root = Path(__file__).resolve().parents[1]
    assert f'version: {PLUGIN_VERSION}' in (root / 'metadata.yaml').read_text(encoding='utf-8')
    assert f'version-{PLUGIN_VERSION}-' in (root / 'README.md').read_text(encoding='utf-8')
    assert (root / 'CHANGELOG.md').read_text(encoding='utf-8').split('## ')[1].startswith(PLUGIN_VERSION + ' -')
    assert '@register(PLUGIN_ID, PLUGIN_NAME, PLUGIN_DESC, PLUGIN_VERSION)' in (root / 'main.py').read_text(encoding='utf-8')


@pytest.fixture
def plugin_module(monkeypatch, tmp_path):
    registrations = []
    def decorator(*args, **kwargs):
        def register(obj):
            registrations.append(obj.__name__)
            return obj
        return register
    class Star:
        def __init__(self, context):
            self.context = context
    class Component:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    filters = SimpleNamespace(PermissionType=SimpleNamespace(ADMIN=1), **{
        name: decorator for name in ('command', 'permission_type', 'on_llm_request',
        'on_llm_response', 'on_decorating_result', 'after_message_sent', 'llm_tool')})
    modules = {
        'astrbot.core.star.filter.command': dict(GreedyStr=str),
        'astrbot.api.event': dict(AstrMessageEvent=object, filter=filters),
        'astrbot.core.message.components': dict(Record=Component, Plain=Component),
        'astrbot.api.star': dict(Context=object, Star=Star, register=decorator),
        'astrbot.core.config.astrbot_config': dict(AstrBotConfig=type('AstrBotConfig', (dict,), {})),
        'astrbot.api.provider': dict(LLMResponse=object),
        'astrbot.core.message.message_event_result': dict(ResultContentType=SimpleNamespace(LLM_RESULT=1)),
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    name = 'astrbot_plugin_tts_emotion_router.main'
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    monkeypatch.setattr(module, 'TEMP_DIR', tmp_path)
    assert 'TTSEmotionRouter' in registrations and 'tts_say' in registrations
    yield module
    sys.modules.pop(name, None)


@pytest.mark.parametrize('segmented', [False, True])
@pytest.mark.parametrize('voice_mode', ['emotion', 'temporary', 'missing'])
def test_loaded_plugin_manual_route(plugin_module, tmp_path, segmented, voice_mode):
    module = plugin_module
    client = SimpleNamespace(synth=AsyncMock(return_value=tmp_path / 'audio.wav'), close=AsyncMock())
    original = module.TTSEmotionRouter._create_tts_client
    module.TTSEmotionRouter._create_tts_client = lambda self: client
    try:
        plugin = module.TTSEmotionRouter(SimpleNamespace(), {
            'tts_engine': {'provider': 'mimo', 'mimo': {'key': 'test', 'voice': '', 'model': 'test'}},
            'segmented_tts': {'enable': segmented, 'split_pattern': '['},
            'emotion_route': {'enable': True, 'voice_map': {'happy': 'emotion-voice'} if voice_mode == 'emotion' else {}},
        })
    finally:
        module.TTSEmotionRouter._create_tts_client = original
    plugin.tts_processor.validate_audio_file = AsyncMock(return_value=True)
    event = SimpleNamespace(unified_msg_origin='test-session')
    text = '用冰糖的音色说你好' if voice_mode == 'temporary' else '今天真开心'
    async def run():
        try:
            result = await plugin._build_manual_tts_chain(event, text)
            assert not plugin.config.validation_errors()
            assert plugin.config.validation_warnings()
            if voice_mode == 'missing':
                assert not result[0] and '未设置音色' in result[2]
                client.synth.assert_not_called()
            else:
                assert result[0], result[2]
                assert client.synth.call_args.args[1] == ('冰糖' if voice_mode == 'temporary' else 'emotion-voice')
        finally:
            await plugin.terminate()
    asyncio.run(run())


@pytest.mark.parametrize('fault', ['missing', 'execution', 'timeout'])
@pytest.mark.parametrize('existing', [False, True])
def test_decoder_failure_never_rebills(monkeypatch, tmp_path, fault, existing):
    class Client:
        format = 'wav'
        def __init__(self):
            self._calls = 0
        @audio_cache
        async def synth(self, text, voice, out_dir):
            self._calls += 1
            path = out_dir / 'sample.wav'
            path.write_bytes(b'preserved audio')
            return path
    import astrbot_plugin_tts_emotion_router.tts.cache as cache
    async def run():
        client = Client()
        real_decode = cache.decode_duration
        monkeypatch.setattr(cache, 'decode_duration', lambda *a: 1)
        if existing:
            await client.synth('text', 'voice', tmp_path)
        monkeypatch.setattr(cache, 'decode_duration', real_decode)
        if fault == 'missing':
            monkeypatch.setattr(diagnostics.shutil, 'which', lambda _: None)
        else:
            def fail(args, **kwargs):
                if fault == 'execution':
                    raise PermissionError('private detail')
                if '-version' in args:
                    return SimpleNamespace(returncode=0)
                raise subprocess.TimeoutExpired(args, 30)
            monkeypatch.setattr(diagnostics.subprocess, 'run', fail)
        for _ in range(2):
            assert await client.synth('text', 'voice', tmp_path) is None
            assert failure.get().code == 'decoder'
            assert 'private detail' not in failure.get().message
        expected = 1 if existing or fault == 'timeout' else 0
        assert client._calls == expected
        files = list(tmp_path.glob('v3-*'))
        assert len(files) == expected
        if files:
            assert files[0].read_bytes() == b'preserved audio'
        monkeypatch.undo()
        monkeypatch.setattr(cache, 'decode_duration', lambda *a: 1)
        assert await client.synth('text', 'voice', tmp_path)
        assert client._calls == 1
    asyncio.run(run())


def test_breeze_provider_factory(plugin_module):
    module = plugin_module
    plugin = module.TTSEmotionRouter(
        SimpleNamespace(),
        {
            "tts_engine": {
                "provider": "breeze",
                "breeze": {
                    "key": "test-key",
                    "default_voice": "voc_test",
                    "guidance_scale": 2.0,
                },
            }
        },
    )
    try:
        client = plugin.tts_client
        assert type(client).__name__ == "BreezeTTS"
        assert client.api_url == "https://api.breeze.blue/v1"
        assert client.voice_id == "voc_test"
        assert client.guidance_scale == 2.0
    finally:
        asyncio.run(plugin.terminate())
