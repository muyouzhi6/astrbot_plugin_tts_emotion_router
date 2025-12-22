# -*- coding: utf-8 -*-
"""
TTS Emotion Router - Session State

会话状态管理模块，定义会话级别的状态数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict
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
    """
    last_ts: float = 0.0
    pending_emotion: Optional[str] = None
    last_emotion: Optional[str] = None
    last_voice: Optional[str] = None
    last_tts_content: Optional[str] = None
    last_tts_time: float = 0.0
    last_assistant_text: Optional[str] = None
    last_assistant_text_time: float = 0.0
    assistant_text: Optional[str] = None
    text_voice_enabled: Optional[bool] = None
    
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
    
    def consume_pending_emotion(self) -> Optional[str]:
        """
        消费并返回待用情绪。
        
        Returns:
            待用情绪字符串，如果没有则返回 None
        """
        emotion = self.pending_emotion
        self.pending_emotion = None
        return emotion
    
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