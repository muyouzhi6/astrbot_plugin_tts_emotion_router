# -*- coding: utf-8 -*-
"""
TTS Emotion Router - Session State

会话状态管理模块，定义会话级别的状态数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import time


@dataclass
class SessionState:
    """
    会话状态数据类。

    用于跟踪每个会话（群组/用户）的 TTS 相关状态。

    Attributes:
        last_ts: 最后一次 TTS 生成的时间戳（用于冷却计算）
        pending_emotion: 基于隐藏标记的待用情绪
        last_emotion: 最后使用的情绪
        last_voice: 最后使用的音色
        last_tts_content: 最后生成的 TTS 内容（防重复）
        last_tts_time: 最后 TTS 生成时间
        last_assistant_text: 最近一次助手可读文本（用于兜底入库）
        last_assistant_text_time: 最近一次助手文本时间
        assistant_text: 当前待保存的助手文本（用于历史记录）
        text_voice_enabled: 文字+语音同时输出设置
            - None: 跟随全局设置
            - True: 会话级开启
            - False: 会话级关闭
        suppress_next_llm_plain_text: 是否抑制下一次 LLM 纯文本输出（一次性）
        suppress_next_llm_plain_text_until: 抑制失效时间戳（0 表示无 TTL）
    """
    last_ts: float = 0.0
    pending_emotion: Optional[str] = None
    pending_style_overrides: Optional[Dict[str, str]] = None
    pending_voice: Optional[str] = None
    pending_director_prompt: Optional[str] = None
    pending_performance_tags: Optional[List[str]] = None
    pending_direct_speak_text: Optional[str] = None
    last_emotion: Optional[str] = None
    last_voice: Optional[str] = None
    last_tts_content: Optional[str] = None
    last_tts_time: float = 0.0
    last_assistant_text: Optional[str] = None
    last_assistant_text_time: float = 0.0
    last_spoken_assistant_text: Optional[str] = None
    last_spoken_assistant_time: float = 0.0
    last_spoken_assistant_conversation_id: Optional[str] = None
    pending_spoken_text: Optional[str] = None
    pending_spoken_conversation_id: Optional[str] = None
    pending_history_text: Optional[str] = None
    pending_history_conversation_id: Optional[str] = None
    text_voice_enabled: Optional[bool] = None
    suppress_next_llm_plain_text: bool = False
    suppress_next_llm_plain_text_until: float = 0.0

    def update_tts_time(self) -> None:
        """更新最后 TTS 生成时间戳。"""
        now = time.time()
        self.last_ts = now
        self.last_tts_time = now

    def set_tts_content(self, content: str) -> None:
        """设置最后的 TTS 内容。"""
        self.last_tts_content = content
        self.update_tts_time()

    def set_assistant_text(self, text: str) -> None:
        """设置最近的助手文本。"""
        self.last_assistant_text = text.strip() if text else None
        self.last_assistant_text_time = time.time()

    def set_spoken_assistant_text(
        self,
        text: str,
        conversation_id: Optional[str] = None,
    ) -> None:
        """设置最近一次已实际发出的语音原话。"""
        cleaned = text.strip() if text else ""
        if not cleaned:
            self.last_spoken_assistant_text = None
            self.last_spoken_assistant_time = 0.0
            self.last_spoken_assistant_conversation_id = None
            return
        self.set_assistant_text(cleaned)
        self.last_spoken_assistant_text = cleaned
        self.last_spoken_assistant_time = self.last_assistant_text_time
        self.last_spoken_assistant_conversation_id = (
            str(conversation_id).strip() if conversation_id else None
        )

    def queue_pending_history(
        self,
        text: str,
        conversation_id: Optional[str],
    ) -> None:
        cleaned = text.strip() if text else ""
        self.pending_history_text = cleaned or None
        self.pending_history_conversation_id = (
            str(conversation_id).strip() if conversation_id else None
        )

    def queue_pending_spoken(
        self,
        text: str,
        conversation_id: Optional[str],
    ) -> None:
        cleaned = text.strip() if text else ""
        self.pending_spoken_text = cleaned or None
        self.pending_spoken_conversation_id = (
            str(conversation_id).strip() if conversation_id else None
        )

    def consume_pending_spoken(self) -> tuple[Optional[str], Optional[str]]:
        text = self.pending_spoken_text
        conversation_id = self.pending_spoken_conversation_id
        self.pending_spoken_text = None
        self.pending_spoken_conversation_id = None
        return text, conversation_id

    def clear_pending_spoken(self) -> None:
        self.pending_spoken_text = None
        self.pending_spoken_conversation_id = None

    def consume_pending_history(self) -> tuple[Optional[str], Optional[str]]:
        text = self.pending_history_text
        conversation_id = self.pending_history_conversation_id
        self.pending_history_text = None
        self.pending_history_conversation_id = None
        return text, conversation_id

    def clear_pending_history(self) -> None:
        self.pending_history_text = None
        self.pending_history_conversation_id = None

    def mark_next_llm_plain_text_suppressed(self, ttl_seconds: Optional[float] = None) -> None:
        """
        标记下一次 LLM 纯文本输出应被抑制。

        Args:
            ttl_seconds: 可选失效时间（秒）；<=0 或 None 表示不设置 TTL
        """
        self.suppress_next_llm_plain_text = True
        if ttl_seconds and ttl_seconds > 0:
            self.suppress_next_llm_plain_text_until = time.time() + float(ttl_seconds)
        else:
            self.suppress_next_llm_plain_text_until = 0.0

    def clear_next_llm_plain_text_suppression(self) -> bool:
        """
        清理下一次 LLM 纯文本输出抑制状态。

        Returns:
            如果状态有变更则返回 True
        """
        changed = self.suppress_next_llm_plain_text or self.suppress_next_llm_plain_text_until > 0
        self.suppress_next_llm_plain_text = False
        self.suppress_next_llm_plain_text_until = 0.0
        return changed

    def clear_next_llm_plain_text_suppression_if_expired(self, now: Optional[float] = None) -> bool:
        """
        若抑制状态已过期则清理。

        Args:
            now: 当前时间戳；为空则使用 time.time()

        Returns:
            若发生清理返回 True
        """
        if not self.suppress_next_llm_plain_text:
            return False
        if self.suppress_next_llm_plain_text_until <= 0:
            return False

        now_ts = time.time() if now is None else now
        if now_ts < self.suppress_next_llm_plain_text_until:
            return False
        return self.clear_next_llm_plain_text_suppression()

    def consume_next_llm_plain_text_suppression(self, now: Optional[float] = None) -> bool:
        """
        消费一次下一次 LLM 纯文本输出抑制状态。

        Args:
            now: 当前时间戳；为空则使用 time.time()

        Returns:
            若成功消费返回 True
        """
        if not self.suppress_next_llm_plain_text:
            return False
        if self.clear_next_llm_plain_text_suppression_if_expired(now):
            return False
        self.clear_next_llm_plain_text_suppression()
        return True

    def consume_pending_emotion(self) -> Optional[str]:
        """消费并返回待用情绪。"""
        emotion = self.pending_emotion
        self.pending_emotion = None
        return emotion

    def set_pending_style_overrides(self, overrides: Dict[str, str]) -> None:
        cleaned = {str(k): str(v).strip() for k, v in (overrides or {}).items() if str(v).strip()}
        self.pending_style_overrides = cleaned or None

    def consume_pending_style_overrides(self) -> Optional[Dict[str, str]]:
        overrides = self.pending_style_overrides
        self.pending_style_overrides = None
        return overrides

    def set_pending_voice(self, voice: str) -> None:
        cleaned = str(voice or "").strip()
        self.pending_voice = cleaned or None

    def consume_pending_voice(self) -> Optional[str]:
        voice = self.pending_voice
        self.pending_voice = None
        return voice

    def set_pending_director_prompt(self, prompt: str) -> None:
        cleaned = str(prompt or "").strip()
        self.pending_director_prompt = cleaned or None

    def consume_pending_director_prompt(self) -> Optional[str]:
        prompt = self.pending_director_prompt
        self.pending_director_prompt = None
        return prompt

    def set_pending_performance_tags(self, tags: Sequence[str]) -> None:
        cleaned: List[str] = []
        seen = set()
        for tag in tags or []:
            value = str(tag or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            cleaned.append(value)
        self.pending_performance_tags = cleaned or None

    def consume_pending_performance_tags(self) -> List[str]:
        tags = self.pending_performance_tags or []
        self.pending_performance_tags = None
        return list(tags)

    def set_pending_direct_speak_text(self, text: str) -> None:
        cleaned = str(text or "").strip()
        self.pending_direct_speak_text = cleaned or None

    def consume_pending_direct_speak_text(self) -> Optional[str]:
        text = self.pending_direct_speak_text
        self.pending_direct_speak_text = None
        return text

    def is_cooldown_expired(self, cooldown: int) -> bool:
        """
        检查冷却时间是否已过。

        Args:
            cooldown: 冷却时间（秒）

        Returns:
            如果冷却时间已过返回 True
        """
        if cooldown <= 0:
            return True
        return (time.time() - self.last_ts) >= cooldown

    def get_remaining_cooldown(self, cooldown: int) -> float:
        """
        获取剩余冷却时间。

        Args:
            cooldown: 冷却时间设置（秒）

        Returns:
            剩余冷却时间（秒），如果已过冷却期返回 0
        """
        if cooldown <= 0:
            return 0.0
        elapsed = time.time() - self.last_ts
        remaining = cooldown - elapsed
        return max(0.0, remaining)


class SessionManager:
    """
    会话状态管理器。

    管理所有会话的状态，提供线程安全的访问接口。
    """

    def __init__(self):
        """初始化会话管理器。"""
        self._sessions: Dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState:
        """
        获取或创建会话状态。

        Args:
            session_id: 会话 ID

        Returns:
            对应的会话状态对象
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState()
        return self._sessions[session_id]

    def get_or_none(self, session_id: str) -> Optional[SessionState]:
        """
        获取会话状态（不创建）。

        Args:
            session_id: 会话 ID

        Returns:
            会话状态对象，如果不存在返回 None
        """
        return self._sessions.get(session_id)

    def remove(self, session_id: str) -> bool:
        """
        移除会话状态。

        Args:
            session_id: 会话 ID

        Returns:
            如果会话存在并被移除返回 True
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def clear(self) -> None:
        """清空所有会话状态。"""
        self._sessions.clear()

    @property
    def count(self) -> int:
        """获取当前会话数量。"""
        return len(self._sessions)

    def __contains__(self, session_id: str) -> bool:
        """检查会话是否存在。"""
        return session_id in self._sessions
