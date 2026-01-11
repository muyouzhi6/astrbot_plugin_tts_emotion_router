# -*- coding: utf-8 -*-
"""
TTS Emotion Router - Constants

常量定义模块，包含路径、情绪类型、关键词映射等常量。
"""

from pathlib import Path
from typing import Dict, Set, List, Tuple, Pattern
import re

# ==================== 插件元数据 ====================

PLUGIN_ID = "astrbot_plugin_tts_emotion_router"
PLUGIN_NAME = "木有知"
PLUGIN_DESC = "按情绪路由到不同音色的TTS插件"
PLUGIN_VERSION = "0.5.0"
PLUGIN_AUTHOR = "Soulter"  # Based on context usually, but keeping it generic or from main.py if available. Wait, main.py says "木有知" is the name? No, register(id, name, desc, version).
# main.py: @register("astrbot_plugin_tts_emotion_router", "木有知", "按情绪路由到不同音色的TTS插件", "0.5.0")
# The second argument is usually the author or plugin name. I will assume "木有知" is the name as per code.

# ==================== 路径常量 ====================

PLUGIN_DIR = Path(__file__).parent.parent
"""插件根目录"""

CONFIG_FILE = PLUGIN_DIR / "config.json"
"""旧版本地配置文件路径（作为迁移来源）"""

TEMP_DIR = PLUGIN_DIR / "temp"
"""临时文件目录"""

VENDORED_ROOT = PLUGIN_DIR / "AstrBot"
"""插件自带的 AstrBot 根目录"""

VENDORED_ASTRBOT = VENDORED_ROOT / "astrbot"
"""插件自带的 astrbot 模块目录"""


# ==================== 情绪常量 ====================

EMOTIONS: Tuple[str, ...] = ("happy", "sad", "angry", "neutral")
"""支持的情绪类型"""


# ==================== 不可见字符 ====================

INVISIBLE_CHARS: List[str] = [
    "\ufeff",  # BOM
    "\u200b",  # Zero Width Space
    "\u200c",  # Zero Width Non-Joiner
    "\u200d",  # Zero Width Joiner
    "\u200e",  # Left-to-Right Mark
    "\u200f",  # Right-to-Left Mark
    "\u202a",  # Left-to-Right Embedding
    "\u202b",  # Right-to-Left Embedding
    "\u202c",  # Pop Directional Formatting
    "\u202d",  # Left-to-Right Override
    "\u202e",  # Right-to-Left Override
]
"""需要移除的不可见字符列表"""


# ==================== 情绪关键词 ====================

EMOTION_KEYWORDS: Dict[str, Pattern] = {
    "happy": re.compile(
        r"(开心|快乐|高兴|喜悦|愉快|兴奋|喜欢|令人开心|挺好|不错|开心|happy|joy|delight|excited|great|awesome|lol)",
        re.I,
    ),
    "sad": re.compile(
        r"(伤心|难过|沮丧|低落|悲伤|哭|流泪|难受|失望|委屈|心碎|sad|depress|upset|unhappy|blue|tear)",
        re.I,
    ),
    "angry": re.compile(
        r"(生气|愤怒|火大|恼火|气愤|气死|怒|怒了|生气了|angry|furious|mad|rage|annoyed|irritat)",
        re.I,
    ),
}
"""情绪关键词正则映射（用于启发式分类）"""


# ==================== 情绪同义词映射 ====================

EMOTION_SYNONYMS: Dict[str, Set[str]] = {
    "happy": {
        "happy", "joy", "joyful", "cheerful", "delighted", "excited",
        "smile", "positive", "开心", "快乐", "高兴", "喜悦", "兴奋", "愉快",
    },
    "sad": {
        "sad", "sorrow", "sorrowful", "depressed", "down", "unhappy",
        "cry", "crying", "tearful", "blue", "upset", "伤心", "难过",
        "沮丧", "低落", "悲伤", "流泪",
    },
    "angry": {
        "angry", "mad", "furious", "annoyed", "irritated", "rage",
        "rageful", "wrath", "生气", "愤怒", "恼火", "气愤",
    },
    "neutral": {
        "neutral", "calm", "plain", "normal", "objective", "ok", "fine",
        "meh", "average", "confused", "uncertain", "unsure", "平静",
        "冷静", "一般", "中立", "客观", "困惑", "迷茫",
    },
}
"""情绪同义词映射（用于标签归一化）"""


# ==================== 情绪偏好映射 ====================

EMOTION_PREFERENCE_MAP: Dict[str, str] = {
    "sad": "angry",
    "angry": "angry",
    "happy": "happy",
    "neutral": "happy",
}
"""情绪偏好映射（当目标情绪无对应音色时的回退策略）"""


# ==================== 配置字段白名单 ====================

CONFIG_MIGRATE_KEYS: List[str] = [
    "global_enable",
    "enabled_umos",        # 白名单 UMO 列表（替代原 enabled_sessions）
    "disabled_umos",       # 黑名单 UMO 列表（替代原 disabled_sessions）
    "text_voice_umos",     # 文字+语音同显 UMO 列表（新增）
    "prob",
    "text_limit",
    "cooldown",
    "allow_mixed",
    "api",
    "voice_map",
    "emotion",
    "speed_map",
]
"""配置迁移时需要拷贝的字段白名单（UMO = 统一消息来源）"""


# ==================== 音频相关常量 ====================

AUDIO_CLEANUP_TTL_SECONDS: int = 2 * 3600
"""临时音频文件清理时间（2小时）"""

AUDIO_MIN_VALID_SIZE: int = 100
"""音频文件最小有效大小（字节）"""

AUDIO_VALID_EXTENSIONS: List[str] = [".mp3", ".wav", ".opus", ".pcm"]
"""支持的音频文件扩展名"""


# ==================== 默认配置值 ====================

DEFAULT_API_MODEL: str = "gpt-tts-pro"
"""默认 TTS API 模型"""

DEFAULT_API_FORMAT: str = "mp3"
"""默认音频格式"""

DEFAULT_API_SPEED: float = 1.0
"""默认语速"""

DEFAULT_API_GAIN: float = 5.0
"""默认音量增益（dB）"""

DEFAULT_SAMPLE_RATE_MP3_WAV: int = 44100
"""MP3/WAV 默认采样率"""

DEFAULT_SAMPLE_RATE_OTHER: int = 48000
"""其他格式默认采样率"""

DEFAULT_PROB: float = 0.8
"""默认 TTS 触发概率"""

DEFAULT_TEXT_LIMIT: int = 80
"""默认文本长度上限"""

DEFAULT_TEXT_MIN_LIMIT: int = 5
"""默认文本长度下限（低于此值不触发TTS）"""

DEFAULT_COOLDOWN: int = 5
"""默认冷却时间（秒）"""

DEFAULT_EMO_MARKER_TAG: str = "EMO"
"""默认情绪标记标签"""

DEFAULT_EMOTION_KEYWORDS_LIST: Dict[str, List[str]] = {
    "happy": ["开心", "高兴", "喜欢", "太棒了", "哈哈", "lol", ":)", "😀"],
    "sad": ["难过", "伤心", "失望", "糟糕", "无语", "唉", "sad", ":(", "😢"],
    "angry": ["气死", "愤怒", "生气", "nm", "tmd", "淦", "怒", "怒了", "😡"],
}
"""默认情绪关键词列表（用于配置界面显示）"""


# ==================== 命令限制常量 ====================

MIN_PROB: float = 0.0
MAX_PROB: float = 1.0
"""概率范围"""

MIN_GAIN: float = -10.0
MAX_GAIN: float = 10.0
"""增益范围 (dB)"""

DEFAULT_TEST_TEXT: str = "你好，这是一个TTS测试"
"""默认测试文本"""


# ==================== 其他常量 ====================

HISTORY_WRITE_DELAY: float = 0.8
"""历史记录写入延迟（秒）"""