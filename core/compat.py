# -*- coding: utf-8 -*-
"""
TTS Emotion Router - Compatibility

AstrBot 兼容性处理模块，确保插件在不同版本的 AstrBot 上正常运行。
"""

from __future__ import annotations

import sys
import logging
import importlib
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from .constants import PLUGIN_DIR, VENDORED_ROOT, VENDORED_ASTRBOT

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


def _import_host_first() -> None:
    """优先尝试导入宿主 AstrBot（临时移除插件路径）。"""
    if VENDORED_ASTRBOT.exists() and "astrbot" not in sys.modules:
        root_str = str(PLUGIN_DIR.resolve())
        _orig = list(sys.path)
        try:
            # 临时移除插件路径，优先导入宿主 AstrBot
            sys.path = [
                p for p in sys.path
                if not (isinstance(p, str) and p.startswith(root_str))
            ]
            importlib.import_module("astrbot")
        finally:
            sys.path = _orig


def _is_compatible() -> bool:
    """检测当前 astrbot 是否兼容（能否导入必要模块）。"""
    try:
        importlib.import_module("astrbot.api.event.filter")
        importlib.import_module("astrbot.core.platform")
        return True
    except Exception as e:
        logger.debug(f"AstrBot compatibility check failed (this is normal if using older/newer version needing fallback): {e}")
        return False


def _force_vendored() -> None:
    """强制切换到插件自带的 AstrBot。"""
    try:
        sys.modules.pop("astrbot", None)
        importlib.invalidate_caches()
        # 确保优先搜索插件自带 AstrBot
        if str(VENDORED_ROOT) not in sys.path:
            sys.path.insert(0, str(VENDORED_ROOT))
        importlib.import_module("astrbot")
        logger.info(
            "TTSEmotionRouter: forced to vendored AstrBot: %s",
            (VENDORED_ASTRBOT / "__init__.py").as_posix()
        )
    except Exception as e:
        logger.error(f"Failed to force vendored AstrBot: {e}", exc_info=True)


def ensure_compatible_astrbot() -> None:
    """
    确保 astrbot API 兼容。
    
    若宿主 astrbot 不满足需要，则回退到插件自带的 AstrBot 处理。
    """
    # 1) 优先尝试宿主
    try:
        _import_host_first()
    except Exception as e:
        logger.warning(f"Failed to import host AstrBot first: {e}", exc_info=True)
    
    # 2) 若不兼容，则强制改用内置 AstrBot
    if not _is_compatible() and VENDORED_ASTRBOT.exists():
        _force_vendored()


def log_astrbot_source() -> None:
    """记录 astrbot 实际来源，便于远端排查导入问题。"""
    try:
        import astrbot as _ab_mod
        logger.info(
            "TTSEmotionRouter: using astrbot from %s",
            getattr(_ab_mod, "__file__", None)
        )
    except Exception as e:
        logger.warning(f"Failed to log AstrBot source: {e}")


# ==================== 兼容性导入辅助 ====================


def import_astr_message_event() -> Any:
    """
    兼容不同 AstrBot 版本的 AstrMessageEvent 导入。
    
    Returns:
        AstrMessageEvent 类
    """
    try:
        # 优先常规路径
        from astrbot.api.event import AstrMessageEvent
        logger.debug("Successfully imported AstrMessageEvent from astrbot.api.event")
        return AstrMessageEvent
    except Exception:
        logger.debug("AstrMessageEvent not found in api.event, falling back to core.platform")
        # 旧版本回退
        from astrbot.core.platform import AstrMessageEvent
        return AstrMessageEvent


def import_filter() -> Any:
    """
    统一获取 filter 装饰器集合。
    
    Returns:
        filter 模块或兼容代理对象
    """
    try:
        # 新版通常支持 from astrbot.api.event import filter
        from astrbot.api.event import filter as _filter
        return _filter
    except Exception:
        logger.debug("Failed to import filter from astrbot.api.event")
    
    try:
        # 另一些版本可 import 子模块
        _filter = importlib.import_module("astrbot.api.event.filter")
        return _filter
    except Exception:
        logger.debug("Failed to import astrbot.api.event.filter module")
    
    # 最后回退：用 register 构造一个拥有同名方法的轻量代理
    try:
        logger.info("Attempting fallback filter implementation using astrbot.core.star.register")
        import astrbot.core.star.register as _reg

        class _FilterCompat:
            """filter 装饰器兼容代理类"""
            
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

            # 兼容某些版本名为 on_after_message_sent
            def on_after_message_sent(self, *a, **k):
                return _reg.register_after_message_sent(*a, **k)

        return _FilterCompat()
    except Exception as _e:
        logger.critical("All filter import strategies failed!", exc_info=True)
        raise _e


def import_message_components() -> tuple:
    """
    兼容不同 AstrBot 版本的消息组件导入。
    
    Returns:
        (Record, Plain) 元组
    """
    try:
        # 优先使用 core 版本的组件类型以匹配 RespondStage 校验逻辑
        from astrbot.core.message.components import Record, Plain
        return Record, Plain
    except Exception:
        logger.debug("Components Record/Plain not found in core.message.components, falling back to api.message_components")
        # 旧版本回退
        from astrbot.api.message_components import Record, Plain
        return Record, Plain


def import_context_and_star() -> tuple:
    """
    导入 Context 和 Star 基类。
    
    Returns:
        (Context, Star, register) 元组
    """
    from astrbot.api.star import Context, Star, register
    return Context, Star, register


def import_astrbot_config() -> Any:
    """
    导入 AstrBotConfig 类。
    
    Returns:
        AstrBotConfig 类
    """
    from astrbot.core.config.astrbot_config import AstrBotConfig
    return AstrBotConfig


def import_llm_response() -> Any:
    """
    导入 LLMResponse 类。
    
    Returns:
        LLMResponse 类
    """
    from astrbot.api.provider import LLMResponse
    return LLMResponse


def import_result_content_type() -> Any:
    """
    导入 ResultContentType 枚举。
    
    Returns:
        ResultContentType 枚举
    """
    from astrbot.core.message.message_event_result import ResultContentType
    return ResultContentType


def import_message_chain() -> Any:
    """
    导入 MessageChain 类。
    
    Returns:
        MessageChain 类
    """
    from astrbot.core.message.message_event_result import MessageChain
    return MessageChain


# ==================== 初始化执行 ====================

def initialize_compat() -> None:
    """
    初始化兼容性处理。
    
    在插件加载时调用，确保正确的 astrbot 模块被加载。
    """
    try:
        ensure_compatible_astrbot()
    except Exception as e:
        logger.error(f"Failed to initialize compatibility module: {e}", exc_info=True)
    
    log_astrbot_source()