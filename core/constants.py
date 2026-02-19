# -*- coding: utf-8 -*-
"""Shared constants for TTS Emotion Router."""

from pathlib import Path
from typing import Dict, List, Pattern, Set, Tuple
import re

# Plugin metadata
PLUGIN_ID = "astrbot_plugin_tts_emotion_router"
PLUGIN_NAME = "TTS 情绪路由"
PLUGIN_DESC = "支持情绪路由、多服务商与会话策略的 TTS 插件。"
PLUGIN_VERSION = "3.0.2"
PLUGIN_AUTHOR = "muyouzhi6"

# Paths
PLUGIN_DIR = Path(__file__).parent.parent
CONFIG_FILE = PLUGIN_DIR / "config.json"
TEMP_DIR = PLUGIN_DIR / "temp"

# Emotion constants
EMOTIONS: Tuple[str, ...] = ("happy", "sad", "angry", "neutral")

INVISIBLE_CHARS: List[str] = [
    "\ufeff",
    "\u200b",
    "\u200c",
    "\u200d",
    "\u200e",
    "\u200f",
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
]

EMOTION_KEYWORDS: Dict[str, Pattern] = {
    "happy": re.compile(r"(happy|great|awesome|excited|lol|nice)", re.I),
    "sad": re.compile(r"(sad|sorry|upset|depressed|cry)", re.I),
    "angry": re.compile(r"(angry|mad|furious|annoyed|rage)", re.I),
}

EMOTION_SYNONYMS: Dict[str, Set[str]] = {
    "happy": {"happy", "joy", "joyful", "cheerful", "excited", "positive"},
    "sad": {"sad", "sorrow", "depressed", "down", "unhappy", "upset"},
    "angry": {"angry", "mad", "furious", "annoyed", "irritated", "rage"},
    "neutral": {"neutral", "calm", "normal", "objective", "ok", "fine"},
}

EMOTION_PREFERENCE_MAP: Dict[str, str] = {
    "sad": "neutral",
    "angry": "neutral",
    "happy": "happy",
    "neutral": "neutral",
}

# Audio constants
AUDIO_CLEANUP_TTL_SECONDS: int = 2 * 3600
AUDIO_MIN_VALID_SIZE: int = 100
AUDIO_VALID_EXTENSIONS: List[str] = [".mp3", ".wav", ".opus", ".pcm"]

# Runtime cleanup limits
SESSION_CLEANUP_INTERVAL_SECONDS: int = 1800
SESSION_MAX_IDLE_SECONDS: int = 86400
SESSION_MAX_COUNT: int = 3000
INFLIGHT_SIG_TTL_SECONDS: int = 180
INFLIGHT_SIG_MAX_COUNT: int = 2000

# Defaults: provider/api
DEFAULT_API_MODEL: str = "gpt-tts-pro"
DEFAULT_TTS_PROVIDER: str = "siliconflow"
DEFAULT_SILICONFLOW_URL: str = "https://api.siliconflow.cn/v1"
DEFAULT_MINIMAX_URL: str = "https://api.minimaxi.com/v1/t2a_v2"
DEFAULT_MINIMAX_MODEL: str = "speech-2.8-hd"
DEFAULT_MINIMAX_VOICE_ID: str = "male-qn-qingse"
DEFAULT_MINIMAX_VOL: float = 1.0
DEFAULT_MINIMAX_PITCH: int = 0
DEFAULT_MINIMAX_BITRATE: int = 128000
DEFAULT_MINIMAX_CHANNEL: int = 1

DEFAULT_FEATURE_MODE: str = "blacklist"
DEFAULT_API_FORMAT: str = "mp3"
DEFAULT_API_SPEED: float = 1.0
DEFAULT_API_TIMEOUT: int = 30
DEFAULT_API_MAX_RETRIES: int = 2
DEFAULT_API_GAIN: float = 0.0
DEFAULT_SAMPLE_RATE_MP3_WAV: int = 44100
DEFAULT_SAMPLE_RATE_OTHER: int = 48000

# Defaults: feature switches
DEFAULT_PROB: float = 0.8
DEFAULT_VOICE_OUTPUT_ENABLE: bool = True
DEFAULT_TEXT_VOICE_ENABLE: bool = False
DEFAULT_SEGMENTED_OUTPUT_ENABLE: bool = False
DEFAULT_PROBABILITY_OUTPUT_ENABLE: bool = True

# Defaults: runtime checks
DEFAULT_TEXT_LIMIT: int = 200
DEFAULT_TEXT_MIN_LIMIT: int = 5
DEFAULT_COOLDOWN: int = 0
DEFAULT_EMO_MARKER_TAG: str = "EMO"
DEFAULT_SEGMENTED_MIN_SEGMENT_LENGTH: int = 5

DEFAULT_EMOTION_KEYWORDS_LIST: Dict[str, List[str]] = {
    "happy": ["happy", "great", "awesome", "lol"],
    "sad": ["sad", "sorry", "upset", "cry"],
    "angry": ["angry", "mad", "annoyed", "rage"],
}

# Limits
MIN_PROB: float = 0.0
MAX_PROB: float = 1.0

# Misc
DEFAULT_TEST_TEXT: str = "这是一条 TTS 测试语音。"
HISTORY_WRITE_DELAY: float = 0.8
