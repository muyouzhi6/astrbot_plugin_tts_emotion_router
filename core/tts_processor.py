# -*- coding: utf-8 -*-
"""
TTS Emotion Router - TTS Processor

核心 TTS 处理逻辑模块。
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from .session import SessionState
    from ..utils.extract import ProcessedText

from .constants import (
    TEMP_DIR,
    EMOTIONS,
    EMOTION_PREFERENCE_MAP,
    AUDIO_MIN_VALID_SIZE,
    AUDIO_VALID_EXTENSIONS,
)
from ..utils.audio import validate_audio_file

logger = logging.getLogger(__name__)


@dataclass
class TTSCheckResult:
    """TTS 条件检查结果"""
    passed: bool
    reason: str = ""
    remaining_cooldown: float = 0.0


@dataclass
class TTSProcessingResult:
    """TTS 处理结果"""
    audio_path: Optional[Path] = None
    emotion: str = "neutral"
    voice: Optional[str] = None
    speed: float = 1.0
    text: str = ""
    success: bool = False
    error: str = ""


class TTSConditionChecker:
    """
    TTS 条件检查器。
    
    检查是否满足 TTS 生成的各种条件。
    """
    
    def __init__(
        self,
        prob: float = 1.0,
        text_limit: int = 0,
        text_min_limit: int = 0,
        cooldown: int = 0,
        allow_mixed: bool = False,
    ):
        """
        初始化条件检查器。
        
        Args:
            prob: TTS 触发概率
            text_limit: 文本长度上限
            text_min_limit: 文本长度下限（低于此值不触发TTS）
            cooldown: 冷却时间
            allow_mixed: 是否允许混合内容
        """
        self.prob = prob
        self.text_limit = text_limit
        self.text_min_limit = text_min_limit
        self.cooldown = cooldown
        self.allow_mixed = allow_mixed
    
    def check_all(
        self,
        text: str,
        session_state: SessionState,
        has_non_plain_elements: bool = False,
        *,
        enable_probability: bool = True,
    ) -> TTSCheckResult:
        """
        执行所有检查。
        
        Args:
            text: 待合成文本
            session_state: 会话状态
            has_non_plain_elements: 消息链中是否包含非纯文本元素
            
        Returns:
            检查结果
        """
        # 1. 混合内容检查
        if has_non_plain_elements:
            # 优先使用会话级设置，如果未设置则使用全局设置
            session_mixed = session_state.text_voice_enabled
            effective_mixed = session_mixed if session_mixed is not None else self.allow_mixed
            
            if not effective_mixed:
                return TTSCheckResult(False, "mixed content not allowed")
        
        # 2. 文本长度下限检查
        if self.text_min_limit > 0 and len(text) < self.text_min_limit:
            return TTSCheckResult(False, f"text too short ({len(text)} < {self.text_min_limit})")
        
        # 3. 文本长度上限检查
        if self.text_limit > 0 and len(text) > self.text_limit:
            return TTSCheckResult(False, f"text too long ({len(text)} > {self.text_limit})")
        
        # 4. 冷却时间检查
        is_cd_ok, remaining = self.check_cooldown(session_state.last_ts)
        if not is_cd_ok:
            return TTSCheckResult(False, f"cooldown ({remaining:.1f}s)", remaining)
            
        # 5. 概率检查
        if enable_probability:
            is_prob_ok, roll = self.check_probability()
            if not is_prob_ok:
                return TTSCheckResult(False, f"probability check failed ({roll:.2f} > {self.prob})")
            
        return TTSCheckResult(True)
    
    def check_probability(self) -> Tuple[bool, float]:
        """检查概率条件。"""
        roll = random.random()
        return roll <= self.prob, roll
    
    def check_cooldown(self, last_ts: float) -> Tuple[bool, float]:
        """检查冷却时间。"""
        if self.cooldown <= 0:
            return True, 0.0
        
        now = time.time()
        elapsed = now - last_ts
        if elapsed >= self.cooldown:
            return True, 0.0
        
        return False, self.cooldown - elapsed


class TTSProcessor:
    """
    TTS 处理器类。
    
    封装核心的 TTS 生成逻辑，包括：
    - 音色选择
    - 音频验证
    - TTS 生成调用
    """
    
    def __init__(
        self,
        tts_client: Any,
        voice_map: Dict[str, str],
        speed_map: Dict[str, float],
        heuristic_classifier: Any,
    ):
        """
        初始化 TTS 处理器。
        
        Args:
            tts_client: TTS 客户端实例
            voice_map: 情绪-音色映射
            speed_map: 情绪-语速映射
            heuristic_classifier: 启发式分类器
        """
        self.tts = tts_client
        self.voice_map = voice_map
        self.speed_map = speed_map
        self.heuristic_cls = heuristic_classifier
    
    async def process(
        self,
        text: str,
        session_state: SessionState,
    ) -> TTSProcessingResult:
        """
        处理 TTS 请求的主入口。
        
        Args:
            text: 待合成文本
            session_state: 会话状态
            
        Returns:
            处理结果
        """
        result = TTSProcessingResult(text=text)
        
        try:
            # 1. 确定情绪
            emotion = self.determine_emotion(session_state, text)
            result.emotion = emotion
            
            # 2. 选择音色
            voice_key, voice_uri = self.pick_voice_for_emotion(emotion)
            if not voice_uri:
                result.success = False
                result.error = f"no voice available for emotion: {emotion}"
                logger.warning(f"TTS skip: {result.error}")
                return result
            
            result.voice = voice_key
            
            # 3. 确定语速
            speed = self.get_speed_for_emotion(emotion)
            result.speed = speed
            
            logger.info(f"TTS: emotion={emotion}, voice={voice_key}, speed={speed}")
            
            # 4. 生成音频
            audio_path = await self.generate_audio(
                text,
                voice_uri,
                speed,
                emotion=emotion,
            )
            
            if audio_path:
                result.success = True
                result.audio_path = audio_path
                # 更新会话状态
                self._update_session_state(session_state, result)
            else:
                result.success = False
                result.error = "audio generation failed"
                
        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error(f"TTS process failed: {e}", exc_info=True)
            
        return result
    
    def _update_session_state(self, st: SessionState, result: TTSProcessingResult) -> None:
        """更新会话状态"""
        now = time.time()
        st.last_ts = now
        st.last_emotion = result.emotion
        st.last_voice = result.voice
        # 注意：不在这里设置 assistant_text，因为那属于发送逻辑
    
    def pick_voice_for_emotion(self, emotion: str) -> Tuple[Optional[str], Optional[str]]:
        """根据情绪选择音色。"""
        vm = self.voice_map or {}
        
        # exact match
        v = vm.get(emotion)
        if v:
            return emotion, v
        
        # neutral fallback
        v = vm.get("neutral")
        if v:
            return "neutral", v
        
        # preference mapping
        for key in [EMOTION_PREFERENCE_MAP.get(emotion), "happy", "angry"]:
            if key and vm.get(key):
                return key, vm[key]
        
        # any available
        for k, v in vm.items():
            if v:
                return k, v
        
        return None, None
    
    def determine_emotion(
        self,
        session_state: SessionState,
        text: str
    ) -> str:
        """确定要使用的情绪。"""
        # 优先使用 pending emotion
        if session_state.pending_emotion and session_state.pending_emotion in EMOTIONS:
            emotion = session_state.pending_emotion
            session_state.pending_emotion = None
            logger.info("Using pending emotion: %s", emotion)
            return emotion
        
        # 启发式分类
        emotion = self.heuristic_cls.classify(text)
        logger.info("Heuristic classified emotion: %s", emotion)
        return emotion
    
    def get_speed_for_emotion(self, emotion: str) -> float:
        """获取情绪对应的语速。"""
        return self.speed_map.get(emotion, self.speed_map.get("neutral", 1.0))
    
    async def generate_audio(
        self,
        text: str,
        voice_uri: str,
        speed: float,
        *,
        emotion: Optional[str] = None,
    ) -> Optional[Path]:
        """生成 TTS 音频。"""
        try:
            audio_path = await self.tts.synth(
                text,
                voice_uri,
                TEMP_DIR,
                speed=speed,
                emotion=emotion,
            )
            
            if not audio_path:
                logger.error("TTS returned empty path")
                return None
            
            audio_path = Path(audio_path)
            if not await self.validate_audio_file(audio_path):
                logger.error("Audio file validation failed")
                return None
            
            return audio_path
            
        except Exception as e:
            logger.error("TTS generation failed", exc_info=True)
            return None

    async def validate_audio_file(self, audio_path: Path) -> bool:
        """验证音频文件是否有效（异步）。"""
        return await validate_audio_file(audio_path)
            
    def normalize_audio_path(self, audio_path: Path) -> str:
        """规范化音频文件路径。"""
        try:
            import os
            abs_path = audio_path.resolve()
            normalized = os.path.normpath(str(abs_path))
            if os.name == 'nt':
                return normalized
            else:
                return normalized.replace('\\', '/')
        except Exception as e:
            logger.error(f"Path normalization failed: {audio_path}", exc_info=True)
            return str(audio_path)


class TTSResultBuilder:
    """
    TTS 结果构建器。
    
    负责构建最终的消息结果链。
    """
    
    def __init__(self, plain_class: type, record_class: type):
        """
        初始化结果构建器。
        
        Args:
            plain_class: Plain 消息组件类
            record_class: Record 消息组件类
        """
        self.Plain = plain_class
        self.Record = record_class
    
    def build(
        self,
        original_chain: list,
        audio_path: str,
        send_text: str,
        text_voice_enabled: bool,
    ) -> list:
        """
        构建包含音频的新消息链。
        
        Args:
            original_chain: 原始消息链
            audio_path: 音频文件路径
            send_text: 要发送的文本内容
            text_voice_enabled: 是否同时发送文本
            
        Returns:
            新的消息链
        """
        new_chain = []
        
        # 1. 如果启用了文字+语音，先添加文字
        if text_voice_enabled and send_text.strip():
            new_chain.append(self.Plain(text=send_text))
            
        # 2. 添加音频
        new_chain.append(self.Record(file=audio_path))
        
        # 3. 保留非 Plain/Record 元素（如图片）
        # 这里的逻辑是：我们替换掉了原来的 Plain，增加了 Record，
        # 但如果是 Image 或其他组件，应该保留下来。
        # 注意：original_chain 里的 Plain 已经被我们处理提取为 send_text 了，所以这里只保留非文本非语音组件。
        for comp in original_chain:
            if not isinstance(comp, (self.Plain, self.Record)):
                new_chain.append(comp)
                
        return new_chain
