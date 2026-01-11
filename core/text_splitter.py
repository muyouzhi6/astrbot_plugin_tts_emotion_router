# -*- coding: utf-8 -*-
"""
TTS Emotion Router - Text Splitter

文本分段模块，将长文本按句子分割成多个片段。
参考 astrbot_plugin_splitter 的智能分段逻辑。
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TextSegment:
    """文本片段"""
    text: str
    index: int  # 片段索引（从0开始）
    
    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class TextSplitter:
    """
    文本分段器。
    
    将长文本按照标点符号分割成多个片段，支持智能模式（识别引号括号内的内容不分割）。
    """
    
    # 默认分段正则：句号、问号、感叹号、换行、省略号
    DEFAULT_SPLIT_PATTERN = r"[。？！?!\n…]+"
    
    # 成对符号映射
    PAIR_MAP = {
        '"': '"', '《': '》', '（': '）', '(': ')',
        '[': ']', '{': '}', '\u2018': '\u2019', '【': '】', '<': '>'
    }
    
    # 引号字符（可能自闭合）
    QUOTE_CHARS = {'"', "'", "`"}
    
    def __init__(
        self,
        split_pattern: str = DEFAULT_SPLIT_PATTERN,
        smart_mode: bool = True,
        max_segments: int = 10,
        min_segment_length: int = 2,
    ):
        """
        初始化文本分段器。
        
        Args:
            split_pattern: 分段正则表达式
            smart_mode: 是否启用智能模式（识别引号括号内不分割）
            max_segments: 最大分段数量
            min_segment_length: 最小片段长度（过短的会合并）
        """
        self.split_pattern = split_pattern
        self.smart_mode = smart_mode
        self.max_segments = max_segments
        self.min_segment_length = min_segment_length
        self._compiled_pattern = re.compile(split_pattern)
    
    def split(self, text: str) -> List[TextSegment]:
        """
        分割文本。
        
        Args:
            text: 待分割文本
            
        Returns:
            TextSegment 列表
        """
        if not text or not text.strip():
            return []
        
        # 执行分割
        if self.smart_mode:
            raw_segments = self._split_smart(text)
        else:
            raw_segments = self._split_simple(text)
        
        # 过滤空片段
        raw_segments = [s for s in raw_segments if s.strip()]
        
        if not raw_segments:
            return []
        
        # 合并过短的片段
        merged = self._merge_short_segments(raw_segments)
        
        # 限制最大分段数
        if len(merged) > self.max_segments and self.max_segments > 0:
            logger.info(f"TextSplitter: segments ({len(merged)}) exceeds max ({self.max_segments}), merging tail")
            final = merged[:self.max_segments - 1]
            # 合并剩余的
            tail = "".join(merged[self.max_segments - 1:])
            final.append(tail)
            merged = final
        
        # 构建结果
        result = [
            TextSegment(text=seg.strip(), index=i)
            for i, seg in enumerate(merged)
            if seg.strip()
        ]
        
        logger.info(f"TextSplitter: split into {len(result)} segment(s)")
        return result
    
    def _split_simple(self, text: str) -> List[str]:
        """简单分割：直接按正则分割。"""
        parts = re.split(f"({self.split_pattern})", text)
        segments = []
        current = ""
        
        for part in parts:
            if not part:
                continue
            if re.fullmatch(self.split_pattern, part):
                # 这是分隔符，追加到当前片段
                current += part
                segments.append(current)
                current = ""
            else:
                # 普通文本
                current += part
        
        if current:
            segments.append(current)
        
        return segments
    
    def _split_smart(self, text: str) -> List[str]:
        """
        智能分割：识别引号和括号内的内容不分割。
        
        参考 splitter 插件的实现。
        """
        stack: List[str] = []
        segments: List[str] = []
        current_chunk = ""
        
        i = 0
        n = len(text)
        
        while i < n:
            char = text[i]
            is_opener = char in self.PAIR_MAP
            
            # 处理引号字符（可能开也可能闭）
            if char in self.QUOTE_CHARS:
                if stack and stack[-1] == char:
                    stack.pop()
                else:
                    stack.append(char)
                current_chunk += char
                i += 1
                continue
            
            # 如果在嵌套结构中
            if stack:
                expected_closer = self.PAIR_MAP.get(stack[-1])
                if char == expected_closer:
                    stack.pop()
                elif is_opener:
                    stack.append(char)
                current_chunk += char
                i += 1
                continue
            
            # 不在嵌套中，检查是否是开始符号
            if is_opener:
                stack.append(char)
                current_chunk += char
                i += 1
                continue
            
            # 检查是否匹配分隔符
            match = self._compiled_pattern.match(text, pos=i)
            if match:
                delimiter = match.group()
                current_chunk += delimiter
                segments.append(current_chunk)
                current_chunk = ""
                i += len(delimiter)
            else:
                current_chunk += char
                i += 1
        
        # 添加剩余内容
        if current_chunk:
            segments.append(current_chunk)
        
        return segments
    
    def _merge_short_segments(self, segments: List[str]) -> List[str]:
        """合并过短的片段到相邻片段。"""
        if not segments or self.min_segment_length <= 0:
            return segments
        
        result: List[str] = []
        buffer = ""
        
        for seg in segments:
            combined = buffer + seg
            if len(combined.strip()) < self.min_segment_length:
                # 太短，继续缓冲
                buffer = combined
            else:
                result.append(combined)
                buffer = ""
        
        # 处理尾部
        if buffer:
            if result:
                result[-1] += buffer
            else:
                result.append(buffer)
        
        return result
    
    def estimate_segment_count(self, text: str) -> int:
        """
        估算分段数量（不实际分割）。
        
        用于快速判断是否需要分段。
        """
        if not text:
            return 0
        matches = self._compiled_pattern.findall(text)
        return min(len(matches) + 1, self.max_segments)
