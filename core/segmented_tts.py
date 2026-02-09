# -*- coding: utf-8 -*-
"""
TTS Emotion Router - Segmented TTS Processor

分段语音处理模块，将长文本分段后逐条发送语音。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine, List, Optional, Tuple

if TYPE_CHECKING:
    from .session import SessionState
    from .tts_processor import TTSProcessor

from .text_splitter import TextSplitter, TextSegment
from .constants import TEMP_DIR

logger = logging.getLogger(__name__)


# 间隔模式常量
INTERVAL_MODE_FIXED = "fixed"
INTERVAL_MODE_ADAPTIVE = "adaptive"


@dataclass
class SegmentTTSResult:
    """单个片段的 TTS 结果"""
    segment: TextSegment
    audio_path: Optional[Path] = None
    duration_seconds: float = 0.0
    success: bool = False
    error: str = ""


@dataclass
class SegmentedTTSResult:
    """分段 TTS 整体结果"""
    segments: List[SegmentTTSResult] = field(default_factory=list)
    emotion: str = "neutral"
    voice: Optional[str] = None
    speed: float = 1.0
    success: bool = False
    error: str = ""
    
    @property
    def successful_segments(self) -> List[SegmentTTSResult]:
        return [s for s in self.segments if s.success]
    
    @property
    def total_duration(self) -> float:
        return sum(s.duration_seconds for s in self.segments if s.success)


async def get_audio_duration(audio_path: Path) -> float:
    """
    获取音频文件时长（秒）。
    
    使用 ffprobe 获取音频时长。如果失败则返回估算值。
    """
    try:
        import subprocess
        
        def _get_duration():
            try:
                result = subprocess.run(
                    [
                        "ffprobe", "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        str(audio_path)
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    return float(result.stdout.strip())
            except Exception as e:
                logger.warning(f"ffprobe failed: {e}")
            return 0.0
        
        duration = await asyncio.to_thread(_get_duration)
        if duration > 0:
            return duration
        
        # 备选：根据文件大小估算（假设 mp3 128kbps）
        # 128kbps = 16KB/s
        try:
            file_size = audio_path.stat().st_size
            estimated = file_size / 16000
            logger.info(f"Estimated duration from file size: {estimated:.2f}s")
            return estimated
        except Exception:
            pass
        
        return 3.0  # 默认假设 3 秒
        
    except Exception as e:
        logger.warning(f"get_audio_duration failed: {e}")
        return 3.0


class SegmentedTTSProcessor:
    """
    分段 TTS 处理器。
    
    将长文本分段后逐条生成语音并发送。
    支持固定间隔和自适应间隔两种模式。
    """
    
    def __init__(
        self,
        tts_processor: TTSProcessor,
        splitter: Optional[TextSplitter] = None,
        interval_mode: str = INTERVAL_MODE_FIXED,
        fixed_interval: float = 1.5,
        adaptive_buffer: float = 0.5,
        max_segments: int = 10,
        min_segment_length: int = 5,
    ):
        """
        初始化分段 TTS 处理器。
        
        Args:
            tts_processor: TTS 处理器实例
            splitter: 文本分段器（可选，不提供则自动创建）
            interval_mode: 间隔模式 ("fixed" 或 "adaptive")
            fixed_interval: 固定间隔时间（秒）
            adaptive_buffer: 自适应模式下的额外缓冲时间（秒）
            max_segments: 最大分段数量
            min_segment_length: 每个分段的最少字符数（低于此值会与相邻分段合并）
        """
        self.tts_processor = tts_processor
        self.splitter = splitter or TextSplitter(
            max_segments=max_segments,
            min_segment_length=min_segment_length,
        )
        self.interval_mode = interval_mode
        self.fixed_interval = fixed_interval
        self.adaptive_buffer = adaptive_buffer
        self.max_segments = max_segments
    
    async def process_and_send(
        self,
        text: str,
        session_state: SessionState,
        send_func: Callable[[Path], Coroutine[Any, Any, bool]],
    ) -> SegmentedTTSResult:
        """
        处理文本并逐条发送语音。
        
        Args:
            text: 待处理文本
            session_state: 会话状态
            send_func: 发送函数，接收音频路径，返回是否成功
            
        Returns:
            分段 TTS 结果
        """
        result = SegmentedTTSResult()
        
        try:
            # 1. 分段
            segments = self.splitter.split(text)
            if not segments:
                result.error = "no segments after splitting"
                return result
            
            logger.info(f"SegmentedTTS: processing {len(segments)} segment(s)")
            
            # 2. 确定统一的情绪/音色/语速（基于整段文本）
            emotion = self.tts_processor.determine_emotion(session_state, text)
            voice_key, voice_uri = self.tts_processor.pick_voice_for_emotion(emotion)
            speed = self.tts_processor.get_speed_for_emotion(emotion)
            
            if not voice_uri:
                result.error = f"no voice available for emotion: {emotion}"
                return result
            
            result.emotion = emotion
            result.voice = voice_key
            result.speed = speed
            
            logger.info(f"SegmentedTTS: unified emotion={emotion}, voice={voice_key}, speed={speed}")
            
            # 3. 逐段处理并发送
            for i, segment in enumerate(segments):
                seg_result = SegmentTTSResult(segment=segment)
                
                try:
                    # 生成音频
                    audio_path = await self.tts_processor.generate_audio(
                        segment.text,
                        voice_uri,
                        speed,
                        emotion=emotion,
                    )
                    
                    if not audio_path:
                        seg_result.error = "audio generation failed"
                        result.segments.append(seg_result)
                        continue
                    
                    seg_result.audio_path = audio_path
                    
                    # 获取音频时长
                    duration = await get_audio_duration(audio_path)
                    seg_result.duration_seconds = duration
                    
                    # 发送
                    norm_path = self.tts_processor.normalize_audio_path(audio_path)
                    send_success = await send_func(Path(norm_path))
                    
                    if send_success:
                        seg_result.success = True
                        logger.info(f"SegmentedTTS: segment {i+1}/{len(segments)} sent, duration={duration:.2f}s")
                    else:
                        seg_result.error = "send failed"
                        logger.warning(f"SegmentedTTS: segment {i+1}/{len(segments)} send failed")
                    
                except Exception as e:
                    seg_result.error = str(e)
                    logger.error(f"SegmentedTTS: segment {i+1} processing error: {e}")
                
                result.segments.append(seg_result)
                
                # 如果不是最后一段，等待间隔
                if i < len(segments) - 1:
                    wait_time = self._calculate_interval(seg_result)
                    logger.info(f"SegmentedTTS: waiting {wait_time:.2f}s before next segment")
                    await asyncio.sleep(wait_time)
            
            # 4. 更新会话状态
            if result.successful_segments:
                result.success = True
                session_state.last_ts = time.time()
                session_state.last_emotion = emotion
                session_state.last_voice = voice_key
            
        except Exception as e:
            result.error = str(e)
            logger.error(f"SegmentedTTS: process_and_send failed: {e}", exc_info=True)
        
        return result
    
    async def process_only(
        self,
        text: str,
        session_state: SessionState,
    ) -> SegmentedTTSResult:
        """
        仅处理文本生成音频，不发送。
        
        返回所有分段的音频路径列表，由调用者负责发送。
        
        Args:
            text: 待处理文本
            session_state: 会话状态
            
        Returns:
            分段 TTS 结果（包含所有音频路径）
        """
        result = SegmentedTTSResult()
        
        try:
            # 1. 分段
            segments = self.splitter.split(text)
            if not segments:
                result.error = "no segments after splitting"
                return result
            
            logger.info(f"SegmentedTTS: processing {len(segments)} segment(s)")
            
            # 2. 确定统一的情绪/音色/语速
            emotion = self.tts_processor.determine_emotion(session_state, text)
            voice_key, voice_uri = self.tts_processor.pick_voice_for_emotion(emotion)
            speed = self.tts_processor.get_speed_for_emotion(emotion)
            
            if not voice_uri:
                result.error = f"no voice available for emotion: {emotion}"
                return result
            
            result.emotion = emotion
            result.voice = voice_key
            result.speed = speed
            
            # 3. 逐段生成音频
            for i, segment in enumerate(segments):
                seg_result = SegmentTTSResult(segment=segment)
                
                try:
                    audio_path = await self.tts_processor.generate_audio(
                        segment.text,
                        voice_uri,
                        speed,
                        emotion=emotion,
                    )
                    
                    if audio_path:
                        seg_result.audio_path = audio_path
                        seg_result.duration_seconds = await get_audio_duration(audio_path)
                        seg_result.success = True
                        logger.info(f"SegmentedTTS: segment {i+1}/{len(segments)} generated")
                    else:
                        seg_result.error = "audio generation failed"
                        
                except Exception as e:
                    seg_result.error = str(e)
                    logger.error(f"SegmentedTTS: segment {i+1} generation error: {e}")
                
                result.segments.append(seg_result)
            
            if result.successful_segments:
                result.success = True
            
        except Exception as e:
            result.error = str(e)
            logger.error(f"SegmentedTTS: process_only failed: {e}", exc_info=True)
        
        return result
    
    def _calculate_interval(self, seg_result: SegmentTTSResult) -> float:
        """
        计算到下一段的等待时间。
        
        Args:
            seg_result: 当前段的结果
            
        Returns:
            等待时间（秒）
        """
        if self.interval_mode == INTERVAL_MODE_ADAPTIVE:
            # 自适应模式：基于音频时长
            duration = seg_result.duration_seconds if seg_result.success else 0
            # 等待时间 = 音频时长 + 缓冲时间
            # 这样可以模拟"正在录语音"的效果
            return max(duration + self.adaptive_buffer, 1.0)
        else:
            # 固定间隔模式
            return self.fixed_interval
    
    def should_use_segmented(self, text: str, min_chars: int = 50) -> bool:
        """
        判断是否应该使用分段模式。
        
        Args:
            text: 待处理文本
            min_chars: 最小字符数阈值
            
        Returns:
            是否应该使用分段模式
        """
        if not text or len(text) < min_chars:
            return False
        
        # 估算分段数量
        estimated = self.splitter.estimate_segment_count(text)
        return estimated >= 2
