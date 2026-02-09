# -*- coding: utf-8 -*-
"""TTS Emotion Router - AstrBot Compatibility (Simplified)"""

from __future__ import annotations

import logging
import importlib
from typing import Any

logger = logging.getLogger(__name__)


def import_astr_message_event() -> Any:
    try:
        from astrbot.api.event import AstrMessageEvent
        return AstrMessageEvent
    except Exception:
        from astrbot.core.platform import AstrMessageEvent
        return AstrMessageEvent


def import_filter() -> Any:
    try:
        from astrbot.api.event import filter as _filter
        return _filter
    except Exception:
        pass

    try:
        _filter = importlib.import_module("astrbot.api.event.filter")
        return _filter
    except Exception:
        pass

    import astrbot.core.star.register as _reg

    class _FilterCompat:
        def command(self, *a, **k):
            return _reg.register_command(*a, **k)
        def on_llm_request(self, *a, **k):
            return _reg.register_on_llm_request(*a, **k)
        def on_llm_response(self, *a, **k):
            return _reg.register_on_llm_response(*a, **k)
        def on_decorating_result(self, *a, **k):
            return _reg.register_on_decorating_result(*a, **k)
        def after_message_sent(self, *a, **k):
            return _reg.register_after_message_sent(*a, **k)
        def on_after_message_sent(self, *a, **k):
            return _reg.register_after_message_sent(*a, **k)

    return _FilterCompat()


def import_message_components() -> tuple:
    try:
        from astrbot.core.message.components import Record, Plain
        return Record, Plain
    except Exception:
        from astrbot.api.message_components import Record, Plain
        return Record, Plain


def import_context_and_star() -> tuple:
    from astrbot.api.star import Context, Star, register
    return Context, Star, register


def import_astrbot_config() -> Any:
    from astrbot.core.config.astrbot_config import AstrBotConfig
    return AstrBotConfig


def import_llm_response() -> Any:
    from astrbot.api.provider import LLMResponse
    return LLMResponse


def import_result_content_type() -> Any:
    from astrbot.core.message.message_event_result import ResultContentType
    return ResultContentType


def initialize_compat() -> None:
    """No-op. Kept for backward compatibility with main.py imports."""
    pass
