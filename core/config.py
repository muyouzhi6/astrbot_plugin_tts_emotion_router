# -*- coding: utf-8 -*-
"""
TTS Emotion Router - Configuration

配置管理模块，处理配置的加载、保存和迁移。
"""

from __future__ import annotations

import json
import logging
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .constants import (
    CONFIG_FILE,
    CONFIG_MIGRATE_KEYS,
    DEFAULT_API_MODEL,
    DEFAULT_API_FORMAT,
    DEFAULT_API_SPEED,
    DEFAULT_API_GAIN,
    DEFAULT_SAMPLE_RATE_MP3_WAV,
    DEFAULT_SAMPLE_RATE_OTHER,
    DEFAULT_PROB,
    DEFAULT_TEXT_LIMIT,
    DEFAULT_COOLDOWN,
    DEFAULT_EMO_MARKER_TAG,
    DEFAULT_EMOTION_KEYWORDS_LIST,
)

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    配置管理器。
    
    支持两种配置模式：
    1. AstrBotConfig 模式：使用面板生成的插件配置
    2. 本地 JSON 模式：直接读写插件目录下的 config.json
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        初始化配置管理器。
        
        Args:
            config: 可以是 AstrBotConfig 实例、dict 或 None
        """
        self._is_astrbot_config = False
        self._config: Union[Any, Dict[str, Any]] = {}
        
        # 检测配置类型
        try:
            from astrbot.core.config.astrbot_config import AstrBotConfig
            if isinstance(config, AstrBotConfig):
                self._is_astrbot_config = True
                self._config = config
                self._try_migrate_from_local()
            else:
                self._config = self._load_local_config(config or {})
        except ImportError:
            self._config = self._load_local_config(config or {})
            
        self._ensure_defaults()
    
    def _ensure_defaults(self) -> None:
        """确保配置中包含必要的默认结构，以便 UI 正确生成。"""
        # 确保 emotion.keywords 存在
        if "emotion" not in self._config:
            self._config["emotion"] = {}
        
        emo_cfg = self._config["emotion"]
        if "keywords" not in emo_cfg:
            emo_cfg["keywords"] = DEFAULT_EMOTION_KEYWORDS_LIST
            # 如果是 AstrBotConfig，可能需要触发保存或更新？
            # 这里直接修改 dict 引用，通常 AstrBotConfig 会代理 __getitem__/__setitem__
            
    def _try_migrate_from_local(self) -> None:
        """尝试从本地 config.json 迁移配置到 AstrBotConfig。"""
        try:
            if getattr(self._config, "first_deploy", False) and CONFIG_FILE.exists():
                disk = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                # 仅拷贝已知字段，避免脏键
                for k in CONFIG_MIGRATE_KEYS:
                    if k in disk:
                        self._config[k] = disk[k]
                # 调用 AstrBotConfig 的 save_config 方法
                if hasattr(self._config, "save_config"):
                    self._config.save_config()  # type: ignore
                logger.info("migrated config from local file")
        except Exception as e:
            logger.warning(f"config migration failed: {e}")
    
    def _load_local_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        加载本地配置文件。
        
        Args:
            cfg: 传入的配置字典
            
        Returns:
            合并后的配置字典
        """
        try:
            if CONFIG_FILE.exists():
                disk = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            else:
                disk = {}
        except Exception as e:
            logger.error(f"failed to load local config: {e}")
            disk = {}
        
        merged = {**disk, **(cfg or {})}
        
        # 写回磁盘
        try:
            CONFIG_FILE.write_text(
                json.dumps(merged, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"failed to write local config: {e}")
        
        return merged
    
    def save(self) -> None:
        """保存配置到持久化存储（同步版本，不建议在事件循环中使用）。"""
        if self._is_astrbot_config:
            if hasattr(self._config, "save_config"):
                self._config.save_config()  # type: ignore
        else:
            try:
                CONFIG_FILE.write_text(
                    json.dumps(self._config, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
            except Exception as e:
                logger.error(f"save config failed: {e}")

    async def save_async(self) -> None:
        """异步保存配置到持久化存储。"""
        if self._is_astrbot_config:
            # AstrBotConfig 目前没有异步 save 接口，只能调用同步的
            # 如果 save_config 内部有阻塞操作，这里应该 wrap 一下
            if hasattr(self._config, "save_config"):
                await asyncio.to_thread(self._config.save_config) # type: ignore
        else:
            try:
                def _write():
                    CONFIG_FILE.write_text(
                        json.dumps(self._config, ensure_ascii=False, indent=2),
                        encoding="utf-8"
                    )
                await asyncio.to_thread(_write)
            except Exception as e:
                logger.error(f"save config failed: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值。
        
        Args:
            key: 配置键名
            default: 默认值
            
        Returns:
            配置值
        """
        try:
            if self._is_astrbot_config:
                return self._config.get(key, default)
            return self._config.get(key, default)
        except Exception as e:
            logger.error(f"get config error: key={key}, error={e}")
            return default
    
    def set(self, key: str, value: Any, save: bool = False) -> None:
        """
        设置配置值（同步保存，已废弃，请使用 set_async）。
        """
        self._config[key] = value
        if save:
            self.save()

    async def set_async(self, key: str, value: Any, save: bool = False) -> None:
        """
        异步设置配置值。
        
        Args:
            key: 配置键名
            value: 配置值
            save: 是否立即保存
        """
        self._config[key] = value
        if save:
            await self.save_async()
    
    def __getitem__(self, key: str) -> Any:
        """支持 config[key] 语法。"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """支持 config[key] = value 语法。"""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """支持 key in config 语法。"""
        try:
            if self._is_astrbot_config:
                return key in self._config
            return key in self._config
        except Exception as e:
            logger.error(f"config contains check error: key={key}, error={e}")
            return False
    
    @property
    def raw(self) -> Union[Any, Dict[str, Any]]:
        """获取原始配置对象。"""
        return self._config
    
    # ==================== 便捷属性 ====================
    
    def get_api_config(self) -> Dict[str, Any]:
        """获取 API 配置。"""
        api = self.get("api", {}) or {}
        return {
            "url": api.get("url", ""),
            "key": api.get("key", ""),
            "model": api.get("model", DEFAULT_API_MODEL),
            "format": api.get("format", DEFAULT_API_FORMAT),
            "speed": float(api.get("speed", DEFAULT_API_SPEED)),
            "gain": float(api.get("gain", DEFAULT_API_GAIN)),
            "sample_rate": int(api.get(
                "sample_rate",
                DEFAULT_SAMPLE_RATE_MP3_WAV
                if api.get("format", DEFAULT_API_FORMAT) in ("mp3", "wav")
                else DEFAULT_SAMPLE_RATE_OTHER
            )),
        }
    
    def get_voice_map(self) -> Dict[str, str]:
        """获取情绪-音色映射。"""
        return self.get("voice_map", {}) or {}
    
    def get_speed_map(self) -> Dict[str, float]:
        """获取情绪-语速映射。"""
        return self.get("speed_map", {}) or {}
    
    def get_global_enable(self) -> bool:
        """获取全局开关状态。"""
        return bool(self.get("global_enable", True))
    
    def get_enabled_sessions(self) -> List[str]:
        """获取白名单会话列表（已废弃，请使用 get_enabled_umos）。"""
        # 向后兼容：优先读取新的 enabled_umos，如果不存在则尝试读取旧的 enabled_sessions
        umos = self.get("enabled_umos", None)
        if umos is not None:
            return list(umos)
        return list(self.get("enabled_sessions", []))
    
    def get_disabled_sessions(self) -> List[str]:
        """获取黑名单会话列表（已废弃，请使用 get_disabled_umos）。"""
        # 向后兼容：优先读取新的 disabled_umos，如果不存在则尝试读取旧的 disabled_sessions
        umos = self.get("disabled_umos", None)
        if umos is not None:
            return list(umos)
        return list(self.get("disabled_sessions", []))
    
    def get_enabled_umos(self) -> List[str]:
        """
        获取白名单 UMO（统一消息来源）列表。
        
        UMO 格式示例：
        - 私聊: "GroupId_UserId"（如 "0_123456"）
        - 群聊: "GroupId_UserId"（如 "789_123456"）
        
        可通过 /sid 命令获取当前会话的 UMO 值。
        
        Returns:
            List[str]: 白名单 UMO 列表
        """
        # 向后兼容：优先读取新的 enabled_umos，如果不存在则尝试读取旧的 enabled_sessions
        umos = self.get("enabled_umos", None)
        if umos is not None:
            return list(umos)
        return list(self.get("enabled_sessions", []))
    
    def get_disabled_umos(self) -> List[str]:
        """
        获取黑名单 UMO（统一消息来源）列表。
        
        UMO 格式示例：
        - 私聊: "GroupId_UserId"（如 "0_123456"）
        - 群聊: "GroupId_UserId"（如 "789_123456"）
        
        可通过 /sid 命令获取当前会话的 UMO 值。
        
        Returns:
            List[str]: 黑名单 UMO 列表
        """
        # 向后兼容：优先读取新的 disabled_umos，如果不存在则尝试读取旧的 disabled_sessions
        umos = self.get("disabled_umos", None)
        if umos is not None:
            return list(umos)
        return list(self.get("disabled_sessions", []))
    
    def get_text_voice_umos(self) -> List[str]:
        """
        获取文字+语音同显 UMO 列表。
        
        在此列表中的 UMO 会同时发送文字和语音消息。
        
        Returns:
            List[str]: 文字+语音同显 UMO 列表
        """
        return list(self.get("text_voice_umos", []))
    
    def get_prob(self) -> float:
        """获取 TTS 触发概率。"""
        return float(self.get("prob", DEFAULT_PROB))
    
    def get_text_limit(self) -> int:
        """获取文本长度限制。"""
        return int(self.get("text_limit", DEFAULT_TEXT_LIMIT))
    
    def get_cooldown(self) -> int:
        """获取冷却时间。"""
        return int(self.get("cooldown", DEFAULT_COOLDOWN))
    
    def get_allow_mixed(self) -> bool:
        """获取是否允许混合输出。"""
        return bool(self.get("allow_mixed", False))
    
    def get_show_references(self) -> bool:
        """获取是否显示参考文献。"""
        return bool(self.get("show_references", True))
    
    def get_emotion_config(self) -> Dict[str, Any]:
        """获取情绪配置。"""
        return self.get("emotion", {}) or {}
    
    def get_marker_config(self) -> Dict[str, Any]:
        """获取情绪标记配置。"""
        emo_cfg = self.get_emotion_config()
        if isinstance(emo_cfg, dict):
            return emo_cfg.get("marker", {}) or {}
        return {}
    
    def is_marker_enabled(self) -> bool:
        """检查情绪标记是否启用。"""
        return bool(self.get_marker_config().get("enable", True))
    
    def get_marker_tag(self) -> str:
        """获取情绪标记标签。"""
        return str(self.get_marker_config().get("tag", DEFAULT_EMO_MARKER_TAG))
    
    def get_emotion_keywords(self) -> Dict[str, List[str]]:
        """
        获取情绪关键词配置。
        
        Returns:
            Dict[str, List[str]]: 情绪关键词字典，如 {"happy": ["开心", ...], ...}
        """
        emo_cfg = self.get_emotion_config()
        if isinstance(emo_cfg, dict):
            return emo_cfg.get("keywords", {}) or {}
        return {}

    # ==================== 会话管理（UMO 版本） ====================
    
    def is_session_enabled(self, session_id: str, global_enable: bool) -> bool:
        """
        检查会话是否启用 TTS（已废弃，请使用 is_umo_enabled）。
        
        Args:
            session_id: 会话 ID（现在应使用 UMO）
            global_enable: 当前全局开关状态
            
        Returns:
            如果会话启用 TTS 返回 True
        """
        return self.is_umo_enabled(session_id, global_enable)
    
    def is_umo_enabled(self, umo: str, global_enable: bool) -> bool:
        """
        检查 UMO（统一消息来源）是否启用 TTS。
        
        Args:
            umo: 统一消息来源标识（可通过 /sid 命令获取）
            global_enable: 当前全局开关状态
            
        Returns:
            如果该 UMO 启用 TTS 返回 True
        """
        if global_enable:
            # 黑名单模式：默认开启，在黑名单中则关闭
            return umo not in self.get_disabled_umos()
        else:
            # 白名单模式：默认关闭，在白名单中则开启
            return umo in self.get_enabled_umos()
    
    def is_text_voice_enabled_for_umo(self, umo: str) -> bool:
        """
        检查 UMO 是否启用文字+语音同显。
        
        Args:
            umo: 统一消息来源标识
            
        Returns:
            如果该 UMO 启用文字+语音同显返回 True
        """
        return umo in self.get_text_voice_umos()
    
    # ---------- 白名单操作（UMO 版本） ----------
    
    def add_to_enabled(self, session_id: str) -> None:
        """添加会话到白名单（同步，已废弃，请使用 add_to_enabled_umos）。"""
        self.add_to_enabled_umos(session_id)

    async def add_to_enabled_async(self, session_id: str) -> None:
        """添加会话到白名单（异步，已废弃，请使用 add_to_enabled_umos_async）。"""
        await self.add_to_enabled_umos_async(session_id)
    
    def add_to_enabled_umos(self, umo: str) -> None:
        """
        添加 UMO 到白名单（同步）。
        
        Args:
            umo: 统一消息来源标识（可通过 /sid 命令获取）
        """
        umos = self.get_enabled_umos()
        if umo not in umos:
            umos.append(umo)
            self.set("enabled_umos", umos, save=True)
    
    async def add_to_enabled_umos_async(self, umo: str) -> None:
        """
        添加 UMO 到白名单（异步）。
        
        Args:
            umo: 统一消息来源标识（可通过 /sid 命令获取）
        """
        umos = self.get_enabled_umos()
        if umo not in umos:
            umos.append(umo)
            await self.set_async("enabled_umos", umos, save=True)
    
    def remove_from_enabled(self, session_id: str) -> None:
        """从白名单移除会话（同步，已废弃，请使用 remove_from_enabled_umos）。"""
        self.remove_from_enabled_umos(session_id)

    async def remove_from_enabled_async(self, session_id: str) -> None:
        """从白名单移除会话（异步，已废弃，请使用 remove_from_enabled_umos_async）。"""
        await self.remove_from_enabled_umos_async(session_id)
    
    def remove_from_enabled_umos(self, umo: str) -> None:
        """
        从白名单移除 UMO（同步）。
        
        Args:
            umo: 统一消息来源标识
        """
        umos = self.get_enabled_umos()
        if umo in umos:
            umos.remove(umo)
            self.set("enabled_umos", umos, save=True)
    
    async def remove_from_enabled_umos_async(self, umo: str) -> None:
        """
        从白名单移除 UMO（异步）。
        
        Args:
            umo: 统一消息来源标识
        """
        umos = self.get_enabled_umos()
        if umo in umos:
            umos.remove(umo)
            await self.set_async("enabled_umos", umos, save=True)
    
    # ---------- 黑名单操作（UMO 版本） ----------
    
    def add_to_disabled(self, session_id: str) -> None:
        """添加会话到黑名单（同步，已废弃，请使用 add_to_disabled_umos）。"""
        self.add_to_disabled_umos(session_id)

    async def add_to_disabled_async(self, session_id: str) -> None:
        """添加会话到黑名单（异步，已废弃，请使用 add_to_disabled_umos_async）。"""
        await self.add_to_disabled_umos_async(session_id)
    
    def add_to_disabled_umos(self, umo: str) -> None:
        """
        添加 UMO 到黑名单（同步）。
        
        Args:
            umo: 统一消息来源标识（可通过 /sid 命令获取）
        """
        umos = self.get_disabled_umos()
        if umo not in umos:
            umos.append(umo)
            self.set("disabled_umos", umos, save=True)
    
    async def add_to_disabled_umos_async(self, umo: str) -> None:
        """
        添加 UMO 到黑名单（异步）。
        
        Args:
            umo: 统一消息来源标识（可通过 /sid 命令获取）
        """
        umos = self.get_disabled_umos()
        if umo not in umos:
            umos.append(umo)
            await self.set_async("disabled_umos", umos, save=True)
    
    def remove_from_disabled(self, session_id: str) -> None:
        """从黑名单移除会话（同步，已废弃，请使用 remove_from_disabled_umos）。"""
        self.remove_from_disabled_umos(session_id)

    async def remove_from_disabled_async(self, session_id: str) -> None:
        """从黑名单移除会话（异步，已废弃，请使用 remove_from_disabled_umos_async）。"""
        await self.remove_from_disabled_umos_async(session_id)
    
    def remove_from_disabled_umos(self, umo: str) -> None:
        """
        从黑名单移除 UMO（同步）。
        
        Args:
            umo: 统一消息来源标识
        """
        umos = self.get_disabled_umos()
        if umo in umos:
            umos.remove(umo)
            self.set("disabled_umos", umos, save=True)
    
    async def remove_from_disabled_umos_async(self, umo: str) -> None:
        """
        从黑名单移除 UMO（异步）。
        
        Args:
            umo: 统一消息来源标识
        """
        umos = self.get_disabled_umos()
        if umo in umos:
            umos.remove(umo)
            await self.set_async("disabled_umos", umos, save=True)
    
    # ---------- 文字+语音同显操作（新增） ----------
    
    def add_to_text_voice_umos(self, umo: str) -> None:
        """
        添加 UMO 到文字+语音同显列表（同步）。
        
        Args:
            umo: 统一消息来源标识（可通过 /sid 命令获取）
        """
        umos = self.get_text_voice_umos()
        if umo not in umos:
            umos.append(umo)
            self.set("text_voice_umos", umos, save=True)
    
    async def add_to_text_voice_umos_async(self, umo: str) -> None:
        """
        添加 UMO 到文字+语音同显列表（异步）。
        
        Args:
            umo: 统一消息来源标识（可通过 /sid 命令获取）
        """
        umos = self.get_text_voice_umos()
        if umo not in umos:
            umos.append(umo)
            await self.set_async("text_voice_umos", umos, save=True)
    
    def remove_from_text_voice_umos(self, umo: str) -> None:
        """
        从文字+语音同显列表移除 UMO（同步）。
        
        Args:
            umo: 统一消息来源标识
        """
        umos = self.get_text_voice_umos()
        if umo in umos:
            umos.remove(umo)
            self.set("text_voice_umos", umos, save=True)
    
    async def remove_from_text_voice_umos_async(self, umo: str) -> None:
        """
        从文字+语音同显列表移除 UMO（异步）。
        
        Args:
            umo: 统一消息来源标识
        """
        umos = self.get_text_voice_umos()
        if umo in umos:
            umos.remove(umo)
            await self.set_async("text_voice_umos", umos, save=True)
            
    # ==================== 配置修改 ====================
    
    def set_global_enable(self, enable: bool) -> None:
        self.set("global_enable", enable, save=True)

    async def set_global_enable_async(self, enable: bool) -> None:
        await self.set_async("global_enable", enable, save=True)
        
    def set_prob(self, prob: float) -> None:
        self.set("prob", prob, save=True)

    async def set_prob_async(self, prob: float) -> None:
        await self.set_async("prob", prob, save=True)
        
    def set_text_limit(self, limit: int) -> None:
        self.set("text_limit", limit, save=True)

    async def set_text_limit_async(self, limit: int) -> None:
        await self.set_async("text_limit", limit, save=True)
        
    def set_cooldown(self, cooldown: int) -> None:
        self.set("cooldown", cooldown, save=True)

    async def set_cooldown_async(self, cooldown: int) -> None:
        await self.set_async("cooldown", cooldown, save=True)
        
    def set_allow_mixed(self, allow: bool) -> None:
        self.set("allow_mixed", allow, save=True)

    async def set_allow_mixed_async(self, allow: bool) -> None:
        await self.set_async("allow_mixed", allow, save=True)
        
    def set_show_references(self, show: bool) -> None:
        self.set("show_references", show, save=True)

    async def set_show_references_async(self, show: bool) -> None:
        await self.set_async("show_references", show, save=True)
        
    def set_api_gain(self, gain: float) -> None:
        api = self.get("api", {}) or {}
        api["gain"] = gain
        self.set("api", api, save=True)

    async def set_api_gain_async(self, gain: float) -> None:
        api = self.get("api", {}) or {}
        api["gain"] = gain
        await self.set_async("api", api, save=True)
        
    def set_marker_enable(self, enable: bool) -> None:
        emo_cfg = self.get("emotion", {}) or {}
        marker_cfg = (emo_cfg.get("marker") or {}) if isinstance(emo_cfg, dict) else {}
        marker_cfg["enable"] = enable
        emo_cfg["marker"] = marker_cfg
        self.set("emotion", emo_cfg, save=True)

    async def set_marker_enable_async(self, enable: bool) -> None:
        emo_cfg = self.get("emotion", {}) or {}
        marker_cfg = (emo_cfg.get("marker") or {}) if isinstance(emo_cfg, dict) else {}
        marker_cfg["enable"] = enable
        emo_cfg["marker"] = marker_cfg
        await self.set_async("emotion", emo_cfg, save=True)

    def get_text_voice_default(self) -> bool:
        """获取文字+语音同时输出的默认值。"""
        return bool(self.get("text_voice_default", False))
    
    # ==================== 分段 TTS 配置 ====================
    
    def get_segmented_tts_config(self) -> Dict[str, Any]:
        """获取分段 TTS 配置。"""
        return self.get("segmented_tts", {}) or {}
    
    def is_segmented_tts_enabled(self) -> bool:
        """检查是否启用分段 TTS。"""
        cfg = self.get_segmented_tts_config()
        return bool(cfg.get("enable", False))
    
    def get_segmented_tts_interval_mode(self) -> str:
        """
        获取分段间隔模式。
        
        Returns:
            "fixed" 或 "adaptive"
        """
        cfg = self.get_segmented_tts_config()
        mode = cfg.get("interval_mode", "fixed")
        if mode not in ("fixed", "adaptive"):
            return "fixed"
        return mode
    
    def get_segmented_tts_fixed_interval(self) -> float:
        """
        获取固定间隔时间（秒）。
        
        Returns:
            固定间隔时间，默认 1.5 秒
        """
        cfg = self.get_segmented_tts_config()
        return float(cfg.get("fixed_interval", 1.5))
    
    def get_segmented_tts_adaptive_buffer(self) -> float:
        """
        获取自适应模式下的缓冲时间（秒）。
        
        Returns:
            缓冲时间，默认 0.5 秒
        """
        cfg = self.get_segmented_tts_config()
        return float(cfg.get("adaptive_buffer", 0.5))
    
    def get_segmented_tts_max_segments(self) -> int:
        """
        获取最大分段数量。
        
        Returns:
            最大分段数，默认 10
        """
        cfg = self.get_segmented_tts_config()
        return int(cfg.get("max_segments", 10))
    
    def get_segmented_tts_min_segment_chars(self) -> int:
        """
        获取触发分段的最小字符数。
        
        Returns:
            最小字符数，默认 50
        """
        cfg = self.get_segmented_tts_config()
        return int(cfg.get("min_segment_chars", 50))
    
    def get_segmented_tts_split_pattern(self) -> str:
        """
        获取分段正则表达式。
        
        Returns:
            分段正则，默认按句号、问号、感叹号、换行分割
        """
        cfg = self.get_segmented_tts_config()
        return str(cfg.get("split_pattern", r"[。？！?!\n…]+"))
    
    def set_segmented_tts_enabled(self, enable: bool) -> None:
        """设置分段 TTS 启用状态（同步）。"""
        cfg = self.get_segmented_tts_config()
        cfg["enable"] = enable
        self.set("segmented_tts", cfg, save=True)
    
    async def set_segmented_tts_enabled_async(self, enable: bool) -> None:
        """设置分段 TTS 启用状态（异步）。"""
        cfg = self.get_segmented_tts_config()
        cfg["enable"] = enable
        await self.set_async("segmented_tts", cfg, save=True)
    
    def set_segmented_tts_interval_mode(self, mode: str) -> None:
        """设置分段间隔模式（同步）。"""
        if mode not in ("fixed", "adaptive"):
            mode = "fixed"
        cfg = self.get_segmented_tts_config()
        cfg["interval_mode"] = mode
        self.set("segmented_tts", cfg, save=True)
    
    async def set_segmented_tts_interval_mode_async(self, mode: str) -> None:
        """设置分段间隔模式（异步）。"""
        if mode not in ("fixed", "adaptive"):
            mode = "fixed"
        cfg = self.get_segmented_tts_config()
        cfg["interval_mode"] = mode
        await self.set_async("segmented_tts", cfg, save=True)
    
    def set_segmented_tts_fixed_interval(self, interval: float) -> None:
        """设置固定间隔时间（同步）。"""
        cfg = self.get_segmented_tts_config()
        cfg["fixed_interval"] = max(0.5, float(interval))
        self.set("segmented_tts", cfg, save=True)
    
    async def set_segmented_tts_fixed_interval_async(self, interval: float) -> None:
        """设置固定间隔时间（异步）。"""
        cfg = self.get_segmented_tts_config()
        cfg["fixed_interval"] = max(0.5, float(interval))
        await self.set_async("segmented_tts", cfg, save=True)