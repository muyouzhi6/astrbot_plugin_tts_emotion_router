# -*- coding: utf-8 -*-
"""Shared constants for TTS Emotion Router."""

from pathlib import Path
from typing import Dict, List, Pattern, Set, Tuple
import re

# Plugin metadata
PLUGIN_ID = "astrbot_plugin_tts_emotion_router"
PLUGIN_NAME = "TTS 情绪路由"
PLUGIN_DESC = "支持情绪路由、多服务商与会话策略的 TTS 插件。"
PLUGIN_VERSION = "3.2.0"
PLUGIN_AUTHOR = "muyouzhi6"

# Paths
PLUGIN_DIR = Path(__file__).parent.parent
CONFIG_FILE = PLUGIN_DIR / "config.json"
TEMP_DIR = PLUGIN_DIR / "temp"

# Emotion constants - 18 emotions are inferred from dialogue context.
# 基础情绪 (9种): 开心/悲伤/愤怒/恐惧/惊讶/兴奋/委屈/平静/冷漠
BASIC_EMOTIONS = ("happy", "sad", "angry", "fear", "surprise", "excited", "wronged", "neutral", "cold")
# 复合情绪 (9种): 怅然/欣慰/无奈/愧疚/释然/嫉妒/厌倦/忐忑/动情
COMPLEX_EMOTIONS = ("melancholy", "gratified", "helpless", "guilty", "relieved", "jealous", "weary", "anxious", "affectionate")
# 特殊功能标签：唱歌仍走同一条路由，但不算“语境情绪”。
SPECIAL_TAGS = ("sing",)

EMOTIONS: Tuple[str, ...] = BASIC_EMOTIONS + COMPLEX_EMOTIONS + SPECIAL_TAGS
CONTEXT_EMOTIONS: Tuple[str, ...] = BASIC_EMOTIONS + COMPLEX_EMOTIONS

# These are not context emotions. They are independent MiMo style dimensions
# that may be configured permanently or overridden temporarily in a user turn.
MIMO_STYLE_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "overall_tone": ("温柔", "高冷", "活泼", "严肃", "慵懒", "俏皮", "深沉", "干练", "凌厉"),
    "timbre_positioning": ("磁性", "醇厚", "清亮", "空灵", "稚嫩", "苍老", "甜美", "沙哑", "醇雅"),
    "persona_accent": ("夹子音", "御姐音", "正太音", "大叔音", "台湾腔"),
    "dialect": ("东北话", "四川话", "河南话", "粤语"),
    "role_play": ("孙悟空", "林黛玉"),
}
MIMO_STYLE_LABEL_TO_CATEGORY: Dict[str, str] = {
    label: category
    for category, labels in MIMO_STYLE_CATEGORIES.items()
    for label in labels
}
MIMO_STYLE_HINTS: Dict[str, str] = {
    category: "/".join(labels)
    for category, labels in MIMO_STYLE_CATEGORIES.items()
}

# MiMo performance tags are explicit stage directions written by the user/LLM.
# They are not inferred or auto-inserted; the sanitizer only keeps whitelisted
# parenthesized tags in TTS text when the feature is enabled for MiMo.
MIMO_PERFORMANCE_TAG_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "rhythm": ("吸气", "深呼吸", "叹气", "长叹一口气", "喘息", "屏息"),
    "emotion_state": ("紧张", "害怕", "激动", "疲惫", "委屈", "撒娇", "心虚", "震惊", "不耐烦"),
    "voice_feature": ("颤抖", "声音颤抖", "变调", "破音", "鼻音", "气声", "沙哑"),
    "cry_laugh": ("笑", "轻笑", "大笑", "冷笑", "抽泣", "呜咽", "哽咽", "嚎啕大哭"),
}
MIMO_PERFORMANCE_TAGS: Tuple[str, ...] = tuple(
    tag
    for tags in MIMO_PERFORMANCE_TAG_CATEGORIES.values()
    for tag in tags
)
MIMO_PERFORMANCE_TAG_HINT: str = " / ".join(MIMO_PERFORMANCE_TAGS)

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
    # \u57fa\u7840\u60c5\u7eea
    "happy": re.compile(r"(happy|great|awesome|excited|lol|nice|\u5f00\u5fc3|\u9ad8\u5174|\u54c8\u54c8|\u563b\u563b|\u592a\u597d\u4e86|\u68d2|\u597d\u8036)", re.I),
    "sad": re.compile(r"(sad|sorry|upset|depressed|cry|\u60b2\u4f24|\u96be\u8fc7|\u4f24\u5fc3|\u545c\u545c|\u5509|\u53ef\u60dc|\u5fc3\u75bc)", re.I),
    "angry": re.compile(r"(angry|mad|furious|annoyed|rage|\u6124\u6012|\u751f\u6c14|\u6c14\u6b7b|\u53ef\u6076|\u70e6\u6b7b\u4e86|\u8ba8\u538c)", re.I),
    "neutral": re.compile(r"(neutral|calm|normal|\u5e73\u9759|\u51b7\u9759|\u6b63\u5e38|\u5ba2\u89c2)", re.I),
    "surprise": re.compile(r"(surprise|wow|omg|\u60ca\u8bb6|\u54c7|\u5929\u54ea|\u4e0d\u4f1a\u5427|\u771f\u7684\u5417|\u6211\u53bb|\u9707\u60ca)", re.I),
    "fear": re.compile(r"(fear|scared|afraid|terrified|\u5bb3\u6015|\u6050\u60e7|\u5413\u6b7b|\u597d\u53ef\u6015|\u745f\u745f\u53d1\u6296)", re.I),
    "excited": re.compile(r"(excited|thrilled|\u5174\u594b|\u6fc0\u52a8|\u592a\u68d2\u4e86|\u597d\u6fc0\u52a8|\u51b2\u51b2\u51b2)", re.I),
    "wronged": re.compile(r"(wronged|mistaken|\u59d4\u5c48|\u51a4\u6789|\u597d\u59d4\u5c48|\u5624\u5624\u5624)", re.I),
    "cold": re.compile(r"(cold|indifferent|\u51b7\u6f20|\u51b7\u6de1|\u65e0\u6240\u8c13|\u968f\u4fbf)", re.I),
    # \u590d\u5408\u60c5\u7eea
    "melancholy": re.compile(r"(melancholy|wistful|\u6005\u7136|\u82e5\u6709\u6240\u601d|\u611f\u6168|\u60c6\u6005)", re.I),
    "gratified": re.compile(r"(gratified|pleased|\u6b23\u6170|\u611f\u5230\u5b89\u6170|\u503c\u5f97\u4e86)", re.I),
    "helpless": re.compile(r"(helpless|\u65e0\u5948|\u65e0\u53ef\u5948\u4f55|\u6ca1\u529e\u6cd5|\u53ea\u80fd\u8fd9\u6837)", re.I),
    "guilty": re.compile(r"(guilty|sorry|\u6127\u759a|\u62b1\u6b49|\u5bf9\u4e0d\u8d77|\u4e0d\u597d\u610f\u601d)", re.I),
    "relieved": re.compile(r"(relieved|\u91ca\u7136|\u653e\u5fc3\u4e86|\u677e\u4e86\u53e3\u6c14|\u7ec8\u4e8e)", re.I),
    "jealous": re.compile(r"(jealous|envious|\u5ac9\u5992|\u7fa1\u6155|\u597d\u7fa1\u6155|\u9178\u4e86)", re.I),
    "weary": re.compile(r"(weary|tired|\u538c\u5026|\u75b2\u60eb|\u7d2f\u4e86|\u591f\u4e86)", re.I),
    "anxious": re.compile(r"(anxious|nervous|\u5fd0\u5fd1|\u7d27\u5f20|\u4e0d\u5b89|\u62c5\u5fc3)", re.I),
    "affectionate": re.compile(r"(affectionate|loving|\u52a8\u60c5|\u6df1\u60c5|\u611f\u52a8|\u7231)", re.I),
    # \u6574\u4f53\u8bed\u8c03
    "gentle": re.compile(r"(gentle|tender|\u6e29\u67d4|\u8f7b\u58f0|\u67d4\u548c|\u6162\u6162|\u8f7b\u67d4)", re.I),
    "cool": re.compile(r"(cool|aloof|\u9ad8\u51b7|\u51b7\u6de1|\u50b2\u5a07|\u4e0d\u5c51)", re.I),
    "lively": re.compile(r"(lively|cheerful|\u6d3b\u6cfc|\u5f00\u6717|\u5143\u6c14|\u6d3b\u529b)", re.I),
    "serious": re.compile(r"(serious|solemn|\u4e25\u8083|\u8ba4\u771f|\u6b63\u7ecf|\u91cd\u8981|\u90d1\u91cd)", re.I),
    "lazy": re.compile(r"(lazy|idle|\u6175\u61d2|\u61d2\u6563|\u56f0|\u4e0d\u60f3\u52a8)", re.I),
    "playful": re.compile(r"(playful|mischievous|\u4fcf\u76ae|\u6d3b\u6cfc|\u8c03\u76ae|\u563f\u563f|\u76ae\u4e00\u4e0b)", re.I),
    "deep": re.compile(r"(deep|profound|\u6df1\u6c89|\u6c89\u7a33|\u4f4e\u6c89|\u6ca7\u6851)", re.I),
    "capable": re.compile(r"(capable|efficient|\u5e72\u7ec3|\u7cbe\u660e|\u5229\u843d|\u679c\u65ad)", re.I),
    "sharp": re.compile(r"(sharp|keen|\u51cc\u5389|\u9510\u5229|\u7280\u5229|\u5c16\u9510)", re.I),
    # \u97f3\u8272\u5b9a\u4f4d
    "magnetic": re.compile(r"(magnetic|charming|\u78c1\u6027|\u8ff7\u4eba|\u6709\u9b45\u529b)", re.I),
    "mellow": re.compile(r"(mellow|rich|\u9187\u539a|\u6d53\u90c1|\u6d51\u539a)", re.I),
    "clear": re.compile(r"(clear|bright|\u6e05\u4eae|\u6e05\u6670|\u660e\u4eae|\u6e05\u6f88)", re.I),
    "ethereal": re.compile(r"(ethereal|airy|\u7a7a\u7075|\u98d8\u6e3a|\u4ed9\u6c14|\u68a6\u5e7b)", re.I),
    "young": re.compile(r"(young|childish|\u7a1a\u5ae9|\u5e74\u8f7b|\u7ae5\u58f0|\u53ef\u7231)", re.I),
    "old": re.compile(r"(old|elderly|\u82cd\u8001|\u5e74\u8fc8|\u6ca7\u6851|\u8001\u6210)", re.I),
    "sweet": re.compile(r"(sweet|\u751c\u7f8e|\u751c\u871c|\u751c|\u597d\u751c)", re.I),
    "hoarse": re.compile(r"(hoarse|raspy|\u6c99\u54d1|\u5636\u54d1|\u54d1)", re.I),
    "elegant": re.compile(r"(elegant|refined|\u9187\u96c5|\u4f18\u96c5|\u9ad8\u96c5|\u7aef\u5e84)", re.I),
    # \u7279\u6b8a\u6807\u7b7e
    "sing": re.compile(r"(sing|singing|\u5531\u6b4c|\u5531|\u6b4c)", re.I),
}

EMOTION_SYNONYMS: Dict[str, Set[str]] = {
    # \u57fa\u7840\u60c5\u7eea
    "happy": {"happy", "joy", "joyful", "cheerful", "excited", "positive", "\u5f00\u5fc3", "\u9ad8\u5174", "\u597d\u8036"},
    "sad": {"sad", "sorrow", "depressed", "down", "unhappy", "upset", "\u60b2\u4f24", "\u96be\u8fc7", "\u5fc3\u75bc"},
    "angry": {"angry", "mad", "furious", "annoyed", "irritated", "rage", "\u6124\u6012", "\u751f\u6c14", "\u8ba8\u538c"},
    "neutral": {"neutral", "calm", "normal", "objective", "ok", "fine", "\u5e73\u9759", "\u51b7\u9759", "\u6b63\u5e38"},
    "surprise": {"surprise", "amazed", "astonished", "wow", "omg", "\u60ca\u8bb6", "\u54c7", "\u6211\u53bb"},
    "fear": {"fear", "scared", "afraid", "terrified", "\u5bb3\u6015", "\u6050\u60e7", "\u745f\u745f\u53d1\u6296"},
    "excited": {"excited", "thrilled", "\u5174\u594b", "\u6fc0\u52a8", "\u597d\u6fc0\u52a8", "\u51b2\u51b2\u51b2"},
    "wronged": {"wronged", "mistaken", "\u59d4\u5c48", "\u51a4\u6789", "\u597d\u59d4\u5c48", "\u5624\u5624\u5624"},
    "cold": {"cold", "indifferent", "\u51b7\u6f20", "\u51b7\u6de1", "\u65e0\u6240\u8c13", "\u968f\u4fbf"},
    # \u590d\u5408\u60c5\u7eea
    "melancholy": {"melancholy", "wistful", "\u6005\u7136", "\u82e5\u6709\u6240\u601d", "\u611f\u6168", "\u60c6\u6005"},
    "gratified": {"gratified", "pleased", "\u6b23\u6170", "\u611f\u5230\u5b89\u6170", "\u503c\u5f97\u4e86"},
    "helpless": {"helpless", "\u65e0\u5948", "\u65e0\u53ef\u5948\u4f55", "\u6ca1\u529e\u6cd5", "\u53ea\u80fd\u8fd9\u6837"},
    "guilty": {"guilty", "sorry", "\u6127\u759a", "\u62b1\u6b49", "\u5bf9\u4e0d\u8d77", "\u4e0d\u597d\u610f\u601d"},
    "relieved": {"relieved", "\u91ca\u7136", "\u653e\u5fc3\u4e86", "\u677e\u4e86\u53e3\u6c14", "\u7ec8\u4e8e"},
    "jealous": {"jealous", "envious", "\u5ac9\u5992", "\u7fa1\u6155", "\u597d\u7fa1\u6155", "\u9178\u4e86"},
    "weary": {"weary", "tired", "\u538c\u5026", "\u75b2\u60eb", "\u7d2f\u4e86", "\u591f\u4e86"},
    "anxious": {"anxious", "nervous", "\u5fd0\u5fd1", "\u7d27\u5f20", "\u4e0d\u5b89", "\u62c5\u5fc3"},
    "affectionate": {"affectionate", "loving", "\u52a8\u60c5", "\u6df1\u60c5", "\u611f\u52a8", "\u7231"},
    # \u6574\u4f53\u8bed\u8c03
    "gentle": {"gentle", "tender", "\u6e29\u67d4", "\u8f7b\u58f0", "\u67d4\u548c", "\u8f7b\u67d4"},
    "cool": {"cool", "aloof", "\u9ad8\u51b7", "\u51b7\u6de1", "\u50b2\u5a07", "\u4e0d\u5c51"},
    "lively": {"lively", "cheerful", "\u6d3b\u6cfc", "\u5f00\u6717", "\u5143\u6c14", "\u6d3b\u529b"},
    "serious": {"serious", "solemn", "\u4e25\u8083", "\u8ba4\u771f", "\u6b63\u7ecf", "\u90d1\u91cd"},
    "lazy": {"lazy", "idle", "\u6175\u61d2", "\u61d2\u6563", "\u56f0", "\u4e0d\u60f3\u52a8"},
    "playful": {"playful", "mischievous", "\u4fcf\u76ae", "\u8c03\u76ae", "\u563f\u563f", "\u76ae\u4e00\u4e0b"},
    "deep": {"deep", "profound", "\u6df1\u6c89", "\u6c89\u7a33", "\u4f4e\u6c89", "\u6ca7\u6851"},
    "capable": {"capable", "efficient", "\u5e72\u7ec3", "\u7cbe\u660e", "\u5229\u843d", "\u679c\u65ad"},
    "sharp": {"sharp", "keen", "\u51cc\u5389", "\u9510\u5229", "\u7280\u5229", "\u5c16\u9510"},
    # \u97f3\u8272\u5b9a\u4f4d
    "magnetic": {"magnetic", "charming", "\u78c1\u6027", "\u8ff7\u4eba", "\u6709\u9b45\u529b"},
    "mellow": {"mellow", "rich", "\u9187\u539a", "\u6d53\u90c1", "\u6d51\u539a"},
    "clear": {"clear", "bright", "\u6e05\u4eae", "\u6e05\u6670", "\u660e\u4eae", "\u6e05\u6f88"},
    "ethereal": {"ethereal", "airy", "\u7a7a\u7075", "\u98d8\u6e3a", "\u4ed9\u6c14", "\u68a6\u5e7b"},
    "young": {"young", "childish", "\u7a1a\u5ae9", "\u5e74\u8f7b", "\u7ae5\u58f0", "\u53ef\u7231"},
    "old": {"old", "elderly", "\u82cd\u8001", "\u5e74\u8fc8", "\u6ca7\u6851", "\u8001\u6210"},
    "sweet": {"sweet", "\u751c\u7f8e", "\u751c\u871c", "\u751c", "\u597d\u751c"},
    "hoarse": {"hoarse", "raspy", "\u6c99\u54d1", "\u5636\u54d1", "\u54d1"},
    "elegant": {"elegant", "refined", "\u9187\u96c5", "\u4f18\u96c5", "\u9ad8\u96c5", "\u7aef\u5e84"},
    # \u7279\u6b8a\u6807\u7b7e
    "sing": {"sing", "singing", "\u5531\u6b4c", "\u5531", "\u6b4c"},
}

EMOTION_PREFERENCE_MAP: Dict[str, str] = {
    # \u57fa\u7840\u60c5\u7eea - \u4fdd\u6301\u539f\u6837
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "neutral": "neutral",
    "surprise": "surprise",
    "fear": "fear",
    "excited": "excited",
    "wronged": "wronged",
    "cold": "cold",
    # \u590d\u5408\u60c5\u7eea - \u4fdd\u6301\u539f\u6837
    "melancholy": "melancholy",
    "gratified": "gratified",
    "helpless": "helpless",
    "guilty": "guilty",
    "relieved": "relieved",
    "jealous": "jealous",
    "weary": "weary",
    "anxious": "anxious",
    "affectionate": "affectionate",
    # \u6574\u4f53\u8bed\u8c03 - \u4fdd\u6301\u539f\u6837
    "gentle": "gentle",
    "cool": "cool",
    "lively": "lively",
    "serious": "serious",
    "lazy": "lazy",
    "playful": "playful",
    "deep": "deep",
    "capable": "capable",
    "sharp": "sharp",
    # \u97f3\u8272\u5b9a\u4f4d - \u4fdd\u6301\u539f\u6837
    "magnetic": "magnetic",
    "mellow": "mellow",
    "clear": "clear",
    "ethereal": "ethereal",
    "young": "young",
    "old": "old",
    "sweet": "sweet",
    "hoarse": "hoarse",
    "elegant": "elegant",
    # \u7279\u6b8a\u6807\u7b7e
    "sing": "sing",
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
DEFAULT_MINIMAX_OUTPUT_FORMAT: str = "hex"
DEFAULT_MINIMAX_LANGUAGE_BOOST: str = ""
DEFAULT_MINIMAX_PROXY: str = ""

MINIMAX_EXPRESSIVE_MODELS: Tuple[str, ...] = ("speech-2.8-hd", "speech-2.8-turbo")
MINIMAX_EXPRESSIVE_TAGS: Tuple[str, ...] = (
    "laughs",
    "chuckle",
    "coughs",
    "clear-throat",
    "groans",
    "breath",
    "pant",
    "inhale",
    "exhale",
    "gasps",
    "sniffs",
    "sighs",
    "snorts",
    "burps",
    "lip-smacking",
    "humming",
    "hissing",
    "emm",
    "sneezes",
)

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
