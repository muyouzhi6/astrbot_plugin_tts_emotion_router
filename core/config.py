# -*- coding: utf-8 -*-
"""TTS Emotion Router - Configuration Manager (v3.1 Simplified)"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .constants import (
    CONFIG_FILE,
    DEFAULT_API_FORMAT,
    DEFAULT_API_GAIN,
    DEFAULT_API_MAX_RETRIES,
    DEFAULT_API_MODEL,
    DEFAULT_API_SPEED,
    DEFAULT_API_TIMEOUT,
    DEFAULT_COOLDOWN,
    DEFAULT_EMO_MARKER_TAG,
    DEFAULT_EMOTION_KEYWORDS_LIST,
    DEFAULT_FEATURE_MODE,
    DEFAULT_MINIMAX_BITRATE,
    DEFAULT_MINIMAX_CHANNEL,
    DEFAULT_MINIMAX_MODEL,
    DEFAULT_MINIMAX_PITCH,
    DEFAULT_MINIMAX_URL,
    DEFAULT_MINIMAX_VOICE_ID,
    DEFAULT_MINIMAX_VOL,
    DEFAULT_PROB,
    DEFAULT_PROBABILITY_OUTPUT_ENABLE,
    DEFAULT_SAMPLE_RATE_MP3_WAV,
    DEFAULT_SAMPLE_RATE_OTHER,
    DEFAULT_SEGMENTED_MIN_SEGMENT_LENGTH,
    DEFAULT_SEGMENTED_OUTPUT_ENABLE,
    DEFAULT_SILICONFLOW_URL,
    DEFAULT_TEXT_LIMIT,
    DEFAULT_TEXT_MIN_LIMIT,
    DEFAULT_TEXT_VOICE_ENABLE,
    DEFAULT_TTS_PROVIDER,
    DEFAULT_VOICE_OUTPUT_ENABLE,
)

logger = logging.getLogger(__name__)

FEATURE_VOICE_OUTPUT = "voice_output"
FEATURE_TEXT_VOICE = "text_voice_output"
FEATURE_SEGMENTED = "segmented_output"
FEATURE_PROBABILITY = "probability_output"

VALID_FEATURES = {
    FEATURE_VOICE_OUTPUT,
    FEATURE_TEXT_VOICE,
    FEATURE_SEGMENTED,
    FEATURE_PROBABILITY,
}


def _normalize_mode(mode: Any) -> str:
    value = str(mode or DEFAULT_FEATURE_MODE).strip().lower()
    return "whitelist" if value == "whitelist" else "blacklist"


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


class ConfigManager:
    """Configuration manager for TTS Emotion Router.

    Supports AstrBotConfig (WebUI) and local JSON fallback.
    All mutation methods are async-only to avoid sync/async duplication.
    """

    def __init__(self, config: Optional[Any] = None):
        self._is_astrbot_config = False
        self._config: Union[Any, Dict[str, Any]] = {}
        self._save_lock = asyncio.Lock()

        try:
            from astrbot.core.config.astrbot_config import AstrBotConfig
            if isinstance(config, AstrBotConfig):
                self._is_astrbot_config = True
                self._config = config
            else:
                self._config = config or {}
        except ImportError:
            self._config = config or {}

        self._ensure_defaults()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    async def save_async(self) -> None:
        async with self._save_lock:
            if self._is_astrbot_config:
                if hasattr(self._config, "save_config"):
                    await asyncio.to_thread(self._config.save_config)
                return
            try:
                def _write():
                    CONFIG_FILE.write_text(
                        json.dumps(self._config, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                await asyncio.to_thread(_write)
            except Exception as e:
                logger.error("Save config failed: %s", e)

    # ------------------------------------------------------------------
    # Basic dict-like APIs
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self._config.get(key, default)
        except Exception:
            return default

    async def set_and_save(self, key: str, value: Any) -> None:
        self._config[key] = value
        await self.save_async()

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        try:
            return key in self._config
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_defaults(enable: bool) -> Dict[str, Any]:
        return {
            "enable": enable,
            "mode": DEFAULT_FEATURE_MODE,
            "enabled_umos": [],
            "disabled_umos": [],
        }

    def _ensure_defaults(self) -> None:
        # UMO guide
        raw_umo_guide = str(self._config.get("umo_guide", "") or "").strip()
        if not raw_umo_guide or "/sid" not in raw_umo_guide:
            self._config["umo_guide"] = "在聊天中发送 /sid 获取当前会话 UMO。"

        # Feature policies
        fp = self.get("feature_policies", {}) or {}
        for feat, default_enable in [
            (FEATURE_VOICE_OUTPUT, DEFAULT_VOICE_OUTPUT_ENABLE),
            (FEATURE_TEXT_VOICE, DEFAULT_TEXT_VOICE_ENABLE),
            (FEATURE_SEGMENTED, DEFAULT_SEGMENTED_OUTPUT_ENABLE),
            (FEATURE_PROBABILITY, DEFAULT_PROBABILITY_OUTPUT_ENABLE),
        ]:
            if feat not in fp:
                fp[feat] = self._feature_defaults(default_enable)
        self._config["feature_policies"] = fp

        # Probability
        probability = self.get("probability", {}) or {}
        if "prob" not in probability:
            probability["prob"] = DEFAULT_PROB
        self._config["probability"] = probability

        # Scalar defaults
        defaults = {
            "text_limit": DEFAULT_TEXT_LIMIT,
            "text_min_limit": DEFAULT_TEXT_MIN_LIMIT,
            "cooldown": DEFAULT_COOLDOWN,
            "allow_mixed": False,
            "show_references": True,
        }
        for k, v in defaults.items():
            if k not in self._config:
                self._config[k] = v

        # TTS engine
        engine = self.get("tts_engine", {}) or {}
        if "provider" not in engine:
            engine["provider"] = DEFAULT_TTS_PROVIDER
        if "timeout" not in engine:
            engine["timeout"] = DEFAULT_API_TIMEOUT
        if "max_retries" not in engine:
            engine["max_retries"] = DEFAULT_API_MAX_RETRIES

        sf = engine.get("siliconflow", {}) or {}
        sf_defaults = {
            "url": DEFAULT_SILICONFLOW_URL,
            "key": "",
            "model": DEFAULT_API_MODEL,
            "format": DEFAULT_API_FORMAT,
            "speed": DEFAULT_API_SPEED,
            "gain": DEFAULT_API_GAIN,
            "sample_rate": DEFAULT_SAMPLE_RATE_MP3_WAV,
            "default_voice": "",
        }
        for k, v in sf_defaults.items():
            if k not in sf:
                sf[k] = v
        engine["siliconflow"] = sf

        mm = engine.get("minimax", {}) or {}
        mm_defaults = {
            "url": DEFAULT_MINIMAX_URL,
            "key": "",
            "model": DEFAULT_MINIMAX_MODEL,
            "voice_id": DEFAULT_MINIMAX_VOICE_ID,
            "speed": DEFAULT_API_SPEED,
            "vol": DEFAULT_MINIMAX_VOL,
            "pitch": DEFAULT_MINIMAX_PITCH,
            "emotion": "neutral",
            "audio_format": DEFAULT_API_FORMAT,
            "sample_rate": 32000,
            "bitrate": DEFAULT_MINIMAX_BITRATE,
            "channel": DEFAULT_MINIMAX_CHANNEL,
            "subtitle_enable": False,
            "pronunciation_dict": {},
        }
        for k, v in mm_defaults.items():
            if k not in mm:
                mm[k] = v
        engine["minimax"] = mm
        self._config["tts_engine"] = engine

        # Emotion route
        route = self.get("emotion_route", {}) or {}
        route_defaults = {
            "enable": True,
            "voice_map": {},
            "speed_map": {},
            "keywords": copy.deepcopy(DEFAULT_EMOTION_KEYWORDS_LIST),
        }
        for k, v in route_defaults.items():
            if k not in route:
                route[k] = v
        marker = route.get("marker", {}) or {}
        if "enable" not in marker:
            marker["enable"] = True
        if "tag" not in marker:
            marker["tag"] = DEFAULT_EMO_MARKER_TAG
        route["marker"] = marker
        self._config["emotion_route"] = route

        # Segmented TTS
        seg = self.get("segmented_tts", {}) or {}
        seg_defaults = {
            "enable": False,
            "interval_mode": "fixed",
            "fixed_interval": 1.5,
            "adaptive_buffer": 0.5,
            "max_segments": 10,
            "min_segment_chars": 50,
            "split_pattern": r"[。？！!?\n…]+",
            "min_segment_length": DEFAULT_SEGMENTED_MIN_SEGMENT_LENGTH,
        }
        for k, v in seg_defaults.items():
            if k not in seg:
                seg[k] = v
        # Fix corrupted split_pattern (encoding issues)
        raw_sp = str(seg.get("split_pattern", ""))
        if any(c in raw_sp for c in ("銆", "鈥", "閵")):
            seg["split_pattern"] = r"[。？！!?\n…]+"
        self._config["segmented_tts"] = seg

    # ------------------------------------------------------------------
    # Feature policy APIs
    # ------------------------------------------------------------------

    def get_feature_policy(self, feature: str) -> Dict[str, Any]:
        if feature not in VALID_FEATURES:
            return self._feature_defaults(False)
        policies = self.get("feature_policies", {}) or {}
        policy = copy.deepcopy(policies.get(feature, {}))
        defaults = self._feature_defaults(False)
        defaults.update(policy)
        defaults["mode"] = _normalize_mode(defaults.get("mode"))
        defaults["enabled_umos"] = list(defaults.get("enabled_umos", []) or [])
        defaults["disabled_umos"] = list(defaults.get("disabled_umos", []) or [])
        defaults["enable"] = bool(defaults.get("enable", False))
        return defaults

    async def set_feature_policy_async(self, feature: str, policy: Dict[str, Any]) -> None:
        if feature not in VALID_FEATURES:
            return
        merged = self.get_feature_policy(feature)
        merged.update(policy or {})
        merged["mode"] = _normalize_mode(merged.get("mode"))
        merged["enabled_umos"] = list(merged.get("enabled_umos", []) or [])
        merged["disabled_umos"] = list(merged.get("disabled_umos", []) or [])
        merged["enable"] = bool(merged.get("enable", False))

        policies = self.get("feature_policies", {}) or {}
        policies[feature] = merged
        await self.set_and_save("feature_policies", policies)

    def _is_feature_enabled_for_umo(self, feature: str, umo: str) -> bool:
        policy = self.get_feature_policy(feature)
        if not policy["enable"]:
            return False
        if _normalize_mode(policy["mode"]) == "whitelist":
            return umo in policy["enabled_umos"]
        return umo not in policy["disabled_umos"]

    def is_voice_output_enabled_for_umo(self, umo: str) -> bool:
        return self._is_feature_enabled_for_umo(FEATURE_VOICE_OUTPUT, umo)

    def is_text_voice_output_enabled_for_umo(self, umo: str) -> bool:
        return self._is_feature_enabled_for_umo(FEATURE_TEXT_VOICE, umo)

    def is_segmented_output_enabled_for_umo(self, umo: str) -> bool:
        return self._is_feature_enabled_for_umo(FEATURE_SEGMENTED, umo)

    def is_probability_output_enabled_for_umo(self, umo: str) -> bool:
        return self._is_feature_enabled_for_umo(FEATURE_PROBABILITY, umo)

    # ------------------------------------------------------------------
    # UMO list mutation (unified for all features)
    # ------------------------------------------------------------------

    async def add_umo_to_feature(self, feature: str, umo: str, list_name: str = "enabled_umos") -> None:
        policy = self.get_feature_policy(feature)
        if umo not in policy[list_name]:
            policy[list_name].append(umo)
            await self.set_feature_policy_async(feature, policy)

    async def remove_umo_from_feature(self, feature: str, umo: str, list_name: str = "enabled_umos") -> None:
        policy = self.get_feature_policy(feature)
        if umo in policy[list_name]:
            policy[list_name].remove(umo)
            await self.set_feature_policy_async(feature, policy)

    # Convenience shortcuts for voice_output
    async def add_to_enabled_umos_async(self, umo: str) -> None:
        await self.add_umo_to_feature(FEATURE_VOICE_OUTPUT, umo, "enabled_umos")

    async def remove_from_enabled_umos_async(self, umo: str) -> None:
        await self.remove_umo_from_feature(FEATURE_VOICE_OUTPUT, umo, "enabled_umos")

    async def add_to_disabled_umos_async(self, umo: str) -> None:
        await self.add_umo_to_feature(FEATURE_VOICE_OUTPUT, umo, "disabled_umos")

    async def remove_from_disabled_umos_async(self, umo: str) -> None:
        await self.remove_umo_from_feature(FEATURE_VOICE_OUTPUT, umo, "disabled_umos")

    # ------------------------------------------------------------------
    # TTS engine APIs
    # ------------------------------------------------------------------

    def get_tts_provider(self) -> str:
        engine = self.get("tts_engine", {}) or {}
        provider = str(engine.get("provider", DEFAULT_TTS_PROVIDER)).strip().lower()
        return provider if provider in {"siliconflow", "minimax"} else DEFAULT_TTS_PROVIDER

    def is_emotion_route_enabled(self) -> bool:
        return bool((self.get("emotion_route", {}) or {}).get("enable", True))

    def get_default_voice(self) -> str:
        route = self.get("emotion_route", {}) or {}
        if self.is_emotion_route_enabled():
            voice_map = route.get("voice_map", {}) or {}
            if voice_map.get("neutral"):
                return str(voice_map["neutral"])
        api_cfg = self.get_api_config()
        return str(api_cfg.get("default_voice", "") or "")

    def get_api_config(self) -> Dict[str, Any]:
        engine = self.get("tts_engine", {}) or {}
        provider = self.get_tts_provider()
        timeout = _safe_int(engine.get("timeout"), DEFAULT_API_TIMEOUT)
        max_retries = _safe_int(engine.get("max_retries"), DEFAULT_API_MAX_RETRIES)

        if provider == "minimax":
            mm = engine.get("minimax", {}) or {}
            audio_format = str(mm.get("audio_format", DEFAULT_API_FORMAT)).lower()
            return {
                "provider": "minimax",
                "url": str(mm.get("url", DEFAULT_MINIMAX_URL)),
                "key": str(mm.get("key", "")),
                "model": str(mm.get("model", DEFAULT_MINIMAX_MODEL)),
                "voice_id": str(mm.get("voice_id", DEFAULT_MINIMAX_VOICE_ID)),
                "speed": _safe_float(mm.get("speed"), DEFAULT_API_SPEED),
                "vol": _safe_float(mm.get("vol"), DEFAULT_MINIMAX_VOL),
                "pitch": _safe_int(mm.get("pitch"), DEFAULT_MINIMAX_PITCH),
                "emotion": str(mm.get("emotion", "neutral")),
                "format": audio_format,
                "sample_rate": _safe_int(mm.get("sample_rate"), 32000),
                "bitrate": _safe_int(mm.get("bitrate"), DEFAULT_MINIMAX_BITRATE),
                "channel": _safe_int(mm.get("channel"), DEFAULT_MINIMAX_CHANNEL),
                "subtitle_enable": bool(mm.get("subtitle_enable", False)),
                "pronunciation_dict": copy.deepcopy(mm.get("pronunciation_dict", {}) or {}),
                "timeout": timeout,
                "max_retries": max_retries,
                "default_voice": str(mm.get("voice_id", DEFAULT_MINIMAX_VOICE_ID)),
                "gain": 0.0,
            }

        sf = engine.get("siliconflow", {}) or {}
        fmt = str(sf.get("format", DEFAULT_API_FORMAT)).lower()
        sr_default = DEFAULT_SAMPLE_RATE_MP3_WAV if fmt in ("mp3", "wav") else DEFAULT_SAMPLE_RATE_OTHER
        return {
            "provider": "siliconflow",
            "url": str(sf.get("url", DEFAULT_SILICONFLOW_URL)).rstrip("/"),
            "key": str(sf.get("key", "")),
            "model": str(sf.get("model", DEFAULT_API_MODEL)),
            "format": fmt,
            "speed": _safe_float(sf.get("speed"), DEFAULT_API_SPEED),
            "gain": _safe_float(sf.get("gain"), DEFAULT_API_GAIN),
            "sample_rate": _safe_int(sf.get("sample_rate"), sr_default),
            "timeout": timeout,
            "max_retries": max_retries,
            "default_voice": str(sf.get("default_voice", "")),
            "voice_id": str(sf.get("default_voice", "")),
            "vol": DEFAULT_MINIMAX_VOL,
            "pitch": DEFAULT_MINIMAX_PITCH,
            "emotion": "neutral",
            "bitrate": DEFAULT_MINIMAX_BITRATE,
            "channel": DEFAULT_MINIMAX_CHANNEL,
            "subtitle_enable": False,
        }

    # ------------------------------------------------------------------
    # Emotion route + marker
    # ------------------------------------------------------------------

    def get_voice_map(self) -> Dict[str, str]:
        route = self.get("emotion_route", {}) or {}
        return dict(route.get("voice_map", {}) or {})

    def get_speed_map(self) -> Dict[str, float]:
        route = self.get("emotion_route", {}) or {}
        return dict(route.get("speed_map", {}) or {})

    def get_marker_config(self) -> Dict[str, Any]:
        route = self.get("emotion_route", {}) or {}
        return dict(route.get("marker", {}) or {})

    def is_marker_enabled(self) -> bool:
        return bool(self.get_marker_config().get("enable", True))

    def get_marker_tag(self) -> str:
        return str(self.get_marker_config().get("tag", DEFAULT_EMO_MARKER_TAG))

    def get_emotion_keywords(self) -> Dict[str, List[str]]:
        route = self.get("emotion_route", {}) or {}
        return dict(route.get("keywords", {}) or {})

    # ------------------------------------------------------------------
    # Scalar getters
    # ------------------------------------------------------------------

    def get_global_enable(self) -> bool:
        policy = self.get_feature_policy(FEATURE_VOICE_OUTPUT)
        return bool(policy["enable"] and policy["mode"] == "blacklist")

    def get_enabled_umos(self) -> List[str]:
        return self.get_feature_policy(FEATURE_VOICE_OUTPUT)["enabled_umos"]

    def get_disabled_umos(self) -> List[str]:
        return self.get_feature_policy(FEATURE_VOICE_OUTPUT)["disabled_umos"]

    def get_prob(self) -> float:
        probability = self.get("probability", {}) or {}
        return _safe_float(probability.get("prob"), DEFAULT_PROB)

    def get_text_limit(self) -> int:
        return _safe_int(self.get("text_limit", DEFAULT_TEXT_LIMIT), DEFAULT_TEXT_LIMIT)

    def get_text_min_limit(self) -> int:
        return _safe_int(self.get("text_min_limit", DEFAULT_TEXT_MIN_LIMIT), DEFAULT_TEXT_MIN_LIMIT)

    def get_cooldown(self) -> int:
        return _safe_int(self.get("cooldown", DEFAULT_COOLDOWN), DEFAULT_COOLDOWN)

    def get_allow_mixed(self) -> bool:
        return bool(self.get("allow_mixed", False))

    def get_show_references(self) -> bool:
        return bool(self.get("show_references", True))

    # ------------------------------------------------------------------
    # Async setters (no sync duplicates)
    # ------------------------------------------------------------------

    async def set_voice_output_enable_async(self, enable: bool) -> None:
        await self.set_feature_policy_async(FEATURE_VOICE_OUTPUT, {"enable": bool(enable)})

    async def set_prob_async(self, prob: float) -> None:
        probability = self.get("probability", {}) or {}
        probability["prob"] = float(prob)
        await self.set_and_save("probability", probability)

    # ------------------------------------------------------------------
    # Segmented TTS
    # ------------------------------------------------------------------

    def get_segmented_tts_config(self) -> Dict[str, Any]:
        return self.get("segmented_tts", {}) or {}

    def is_segmented_tts_enabled(self) -> bool:
        return bool(self.get_segmented_tts_config().get("enable", False))

    def get_segmented_tts_interval_mode(self) -> str:
        mode = str(self.get_segmented_tts_config().get("interval_mode", "fixed"))
        return mode if mode in ("fixed", "adaptive") else "fixed"

    def get_segmented_tts_fixed_interval(self) -> float:
        return _safe_float(self.get_segmented_tts_config().get("fixed_interval"), 1.5)

    def get_segmented_tts_adaptive_buffer(self) -> float:
        return _safe_float(self.get_segmented_tts_config().get("adaptive_buffer"), 0.5)

    def get_segmented_tts_max_segments(self) -> int:
        return _safe_int(self.get_segmented_tts_config().get("max_segments"), 10)

    def get_segmented_tts_min_segment_chars(self) -> int:
        return _safe_int(self.get_segmented_tts_config().get("min_segment_chars"), 50)

    def get_segmented_tts_split_pattern(self) -> str:
        return str(self.get_segmented_tts_config().get("split_pattern", r"[。？！!?\n…]+"))

    def get_segmented_tts_min_segment_length(self) -> int:
        return _safe_int(
            self.get_segmented_tts_config().get("min_segment_length"),
            DEFAULT_SEGMENTED_MIN_SEGMENT_LENGTH,
        )
