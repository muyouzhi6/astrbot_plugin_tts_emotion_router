# -*- coding: utf-8 -*-
"""
TTS Emotion Router - Hooks Module

LLM 和消息处理钩子的逻辑实现。
"""

from __future__ import annotations

import logging
import time
import asyncio

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from .constants import EMOTIONS, TEMP_DIR
from .session import SessionState

if TYPE_CHECKING:
    from .marker import EmotionMarkerProcessor
    from .tts_processor import TTSProcessor, TTSConditionChecker


class LLMHooksHandler:
    """
    LLM 钩子处理器。
    
    处理 LLM 请求和响应的钩子逻辑。
    """
    
    def __init__(
        self,
        marker_processor: "EmotionMarkerProcessor",
        session_state: Dict[str, SessionState],
        sess_id_func,
    ):
        """
        初始化 LLM 钩子处理器。
        
        Args:
            marker_processor: 情绪标记处理器
            session_state: 会话状态字典
            sess_id_func: 获取会话 ID 的函数
        """
        self.marker_processor = marker_processor
        self._session_state = session_state
        self._sess_id = sess_id_func
    
    def handle_llm_request(self, request: Any) -> None:
        """
        处理 LLM 请求，注入情绪标记指令。
        
        Args:
            request: LLM 请求对象
        """
        logger.debug("LLMHooksHandler: Handling LLM request")
        try:
            sp = getattr(request, "system_prompt", "") or ""
            pp = getattr(request, "prompt", "") or ""
            
            if not self.marker_processor.is_marker_present(sp, pp):
                instr = self.marker_processor.build_injection_instruction()
                try:
                    request.system_prompt = (instr + "\n" + sp).strip()
                except Exception:
                    logger.warning("Failed to inject emotion marker instruction", exc_info=True)
            
            try:
                ctxs = getattr(request, "contexts", None)
                clen = len(ctxs) if isinstance(ctxs, list) else 0
                plen = len(getattr(request, "prompt", "") or "")
                logger.debug(f"LLMHooksHandler.handle_llm_request: injected=True, contexts={clen}, prompt_len={plen}")
            except Exception:
                logger.debug("Failed to log request details", exc_info=True)
        except Exception:
            logger.error("Error in handle_llm_request", exc_info=True)
    
    def handle_llm_response(
        self,
        event: Any,
        response: Any,
        Plain: type,
    ) -> Optional[str]:
        logger.debug("LLMHooksHandler: Handling LLM response")
        """
        处理 LLM 响应，解析并清理情绪标记。
        
        Args:
            event: 消息事件
            response: LLM 响应对象
            Plain: Plain 消息组件类
            
        Returns:
            缓存的文本内容
        """
        label: Optional[str] = None
        cached_text: Optional[str] = None
        
        # 1) 从 completion_text 提取并清理
        try:
            text = getattr(response, "completion_text", None)
            if isinstance(text, str) and text.strip():
                t0 = self.marker_processor.normalize_text(text)
                cleaned, l1 = self.marker_processor.strip_head_many(t0)
                if l1 in EMOTIONS:
                    label = l1
                response.completion_text = cleaned
                try:
                    setattr(response, "_completion_text", cleaned)
                except Exception:
                    logger.debug("Failed to set _completion_text on response", exc_info=True)
                cached_text = cleaned or cached_text
        except Exception:
            logger.error("Error processing completion_text in handle_llm_response", exc_info=True)
        
        # 2) 从 result_chain 首个 Plain 再尝试一次
        try:
            rc = getattr(response, "result_chain", None)
            if rc and hasattr(rc, "chain") and rc.chain:
                new_chain = []
                cleaned_once = False
                for comp in rc.chain:
                    if (
                        not cleaned_once
                        and isinstance(comp, Plain)
                        and getattr(comp, "text", None)
                    ):
                        t0 = self.marker_processor.normalize_text(comp.text)
                        t, l2 = self.marker_processor.strip_head_many(t0)
                        if l2 in EMOTIONS and label is None:
                            label = l2
                        if t:
                            new_chain.append(Plain(text=t))
                            try:
                                if t and not getattr(response, "_completion_text", None):
                                    setattr(response, "_completion_text", t)
                            except Exception:
                                logger.debug("Failed to set _completion_text from result chain", exc_info=True)
                            cached_text = t or cached_text
                        cleaned_once = True
                    else:
                        new_chain.append(comp)
                rc.chain = new_chain
        except Exception:
            logger.error("Error processing result_chain in handle_llm_response", exc_info=True)
        
        # 3) 记录到 session
        try:
            sid = self._sess_id(event)
            st = self._session_state.setdefault(sid, SessionState())
            if label in EMOTIONS:
                st.pending_emotion = label
            if cached_text and cached_text.strip():
                st.set_assistant_text(cached_text.strip())
        except Exception:
            logger.error("Error updating session state in handle_llm_response", exc_info=True)
        
        return cached_text


class TTSHooksHandler:
    """
    TTS 钩子处理器。
    
    处理 TTS 生成的核心逻辑。
    """
    
    def __init__(
        self,
        tts_processor: "TTSProcessor",
        condition_checker: "TTSConditionChecker",
        marker_processor: "EmotionMarkerProcessor",
        session_state: Dict[str, SessionState],
        inflight_sigs: Set[str],
        sess_id_func,
        is_session_enabled_func,
        config,
        extractor,
    ):
        """
        初始化 TTS 钩子处理器。
        """
        self.tts_processor = tts_processor
        self.condition_checker = condition_checker
        self.marker_processor = marker_processor
        self._session_state = session_state
        self._inflight_sigs = inflight_sigs
        self._sess_id = sess_id_func
        self._is_session_enabled = is_session_enabled_func
        self.config = config
        self.extractor = extractor
    
    def clean_result_chain(self, result: Any, Plain: type) -> None:
        """清理结果链中的情绪标记。"""
        try:
            new_chain = []
            for comp in result.chain:
                if isinstance(comp, Plain) and getattr(comp, "text", None):
                    t0 = self.marker_processor.normalize_text(comp.text)
                    t, _ = self.marker_processor.strip_head_many(t0)
                    t = self.marker_processor.strip_all_visible_markers(t)
                    if t:
                        new_chain.append(Plain(text=t))
                else:
                    new_chain.append(comp)
            result.chain = new_chain
        except Exception:
            logger.error("Error cleaning result chain", exc_info=True)
    
    def check_should_process(
        self,
        event: Any,
        result: Any,
        Plain: type,
        allow_mixed: bool,
    ) -> tuple[bool, str, Optional[SessionState]]:
        """
        检查是否应该处理 TTS。
        
        Returns:
            (should_process, session_id, session_state)
        """
        sid = self._sess_id(event)
        
        # 检查会话是否启用
        if not self._is_session_enabled(sid):
            logger.info("TTS skip: session disabled (%s)", sid)
            return False, sid, None
        
        # 检查混合内容
        has_non_plain = any(not isinstance(c, Plain) for c in result.chain)
        if has_non_plain:
            st = self._session_state.get(sid)
            session_text_voice = st.text_voice_enabled if st else None
            effective_allow_mixed = session_text_voice if session_text_voice is not None else allow_mixed
            if not effective_allow_mixed:
                logger.info("TTS skip: mixed content not allowed")
                return False, sid, None
        
        st = self._session_state.setdefault(sid, SessionState())
        return True, sid, st
    
    def extract_text(self, result: Any, Plain: type) -> tuple[str, List[str]]:
        """
        从结果链中提取文本。
        
        Returns:
            (combined_text, text_parts)
        """
        text_parts = [
            c.text.strip()
            for c in result.chain
            if isinstance(c, Plain) and c.text.strip()
        ]
        if not text_parts:
            return "", []
        return " ".join(text_parts), text_parts
    
    def check_conditions(
        self,
        text: str,
        session_state: SessionState,
        sid: str,
    ) -> tuple[bool, Optional[str]]:
        """
        检查 TTS 条件。
        
        Returns:
            (should_continue, skip_reason)
        """
        # 冷却检查
        cooldown_ok, remaining = self.condition_checker.check_cooldown(session_state.last_ts)
        if not cooldown_ok:
            return False, f"cooldown ({remaining:.1f}s remaining)"
        
        # 长度检查
        if (
            getattr(self.condition_checker, "text_min_limit", 0) > 0
            and len(text) < self.condition_checker.text_min_limit
        ):
            return (
                False,
                f"text too short ({len(text)} < {self.condition_checker.text_min_limit})",
            )
        if (
            getattr(self.condition_checker, "text_limit", 0) > 0
            and len(text) > self.condition_checker.text_limit
        ):
            return False, f"text too long ({len(text)} > {self.condition_checker.text_limit})"
        
        # 概率检查
        prob_ok, roll = self.condition_checker.check_probability()
        if not prob_ok:
            return False, f"probability check failed ({roll:.2f} > {self.condition_checker.prob})"
        
        # 去重检查
        sig = f"{sid}:{hash(text[:50])}"
        if sig in self._inflight_sigs:
            return False, "duplicate request in flight"
        
        return True, None
    
    async def generate_tts(
        self,
        text: str,
        session_state: SessionState,
    ) -> tuple[Optional[Path], str, Optional[str]]:
        """
        生成 TTS 音频。
        
        Returns:
            (audio_path, emotion, voice_key)
        """
        # 确定情绪
        emotion = self.tts_processor.determine_emotion(session_state, text)
        
        # 选择音色
        voice_key, voice_uri = self.tts_processor.pick_voice_for_emotion(emotion)
        if not voice_uri:
            logger.warning("TTS skip: no voice available for emotion: %s", emotion)
            return None, emotion, None
        
        logger.info("TTS: emotion=%s, voice=%s", emotion, voice_key)
        
        # 获取语速
        speed = self.tts_processor.get_speed_for_emotion(emotion)
        
        # 生成音频
        logger.debug(f"Generating audio for text: {text[:50]}...")
        audio_path = await self.tts_processor.generate_audio(text, voice_uri, speed)
        
        return audio_path, emotion, voice_key
