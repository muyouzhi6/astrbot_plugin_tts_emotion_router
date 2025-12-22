# -*- coding: utf-8 -*-
"""
TTS Emotion Router - Core Module

核心模块，包含常量定义、兼容性处理、配置管理、会话状态和情绪标记处理。
"""

from .constants import (
    CONFIG_FILE,
    TEMP_DIR,
    PLUGIN_DIR,
    EMOTIONS,
    INVISIBLE_CHARS,
    EMOTION_KEYWORDS,
    EMOTION_SYNONYMS,
    EMOTION_PREFERENCE_MAP,
    AUDIO_CLEANUP_TTL_SECONDS,
    AUDIO_MIN_VALID_SIZE,
    AUDIO_VALID_EXTENSIONS,
    DEFAULT_API_MODEL,
    DEFAULT_API_FORMAT,
    DEFAULT_API_SPEED,
    DEFAULT_API_GAIN,
    DEFAULT_PROB,
    DEFAULT_TEXT_LIMIT,
    DEFAULT_COOLDOWN,
    DEFAULT_EMO_MARKER_TAG,
)
from .session import SessionState, SessionManager
from .config import ConfigManager
from .marker import EmotionMarkerProcessor
from .tts_processor import TTSProcessor, TTSResultBuilder, TTSConditionChecker
from .compat import (
    initialize_compat,
    import_astr_message_event,
    import_filter,
    import_message_components,
    import_context_and_star,
    import_astrbot_config,
    import_llm_response,
    import_result_content_type,
)

__all__ = [
    # 常量
    "CONFIG_FILE",
    "TEMP_DIR",
    "PLUGIN_DIR",
    "EMOTIONS",
    "INVISIBLE_CHARS",
    "EMOTION_KEYWORDS",
    "EMOTION_SYNONYMS",
    "EMOTION_PREFERENCE_MAP",
    "AUDIO_CLEANUP_TTL_SECONDS",
    "AUDIO_MIN_VALID_SIZE",
    "AUDIO_VALID_EXTENSIONS",
    "DEFAULT_API_MODEL",
    "DEFAULT_API_FORMAT",
    "DEFAULT_API_SPEED",
    "DEFAULT_API_GAIN",
    "DEFAULT_PROB",
    "DEFAULT_TEXT_LIMIT",
    "DEFAULT_COOLDOWN",
    "DEFAULT_EMO_MARKER_TAG",
    # 类
    "SessionState",
    "SessionManager",
    "ConfigManager",
    "EmotionMarkerProcessor",
    # 兼容性函数
    "initialize_compat",
    "import_astr_message_event",
    "import_filter",
    "import_message_components",
    "import_context_and_star",
    "import_astrbot_config",
    "import_llm_response",
    "import_result_content_type",
    # TTS 处理器
    "TTSProcessor",
    "TTSResultBuilder",
    "TTSConditionChecker",
]