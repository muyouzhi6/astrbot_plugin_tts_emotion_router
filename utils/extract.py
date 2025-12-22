# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import List, Pattern
from dataclasses import dataclass


@dataclass
class ProcessedText:
    clean_text: str
    speak_text: str
    links: List[str]
    codes: List[str]
    has_links_or_code: bool


class CodeAndLinkExtractor:
    """提取文本中的代码块和链接，并生成用于发送和语音合成的文本"""

    combined_re: Pattern

    def __init__(self):
        # 正则表达式合并，并使用命名捕获组
        # 增强代码块匹配容错性：允许任意字符，非贪婪匹配
        code_block_pattern = r'```[\s\S]*?```'
        # 放宽行内代码匹配正则：匹配反引号包裹的非换行内容
        inline_code_pattern = r'`[^`\n]+`'
        # URL 匹配增强：
        # 1. 支持 http/https
        # 2. 支持 www. 开头的域名（自动补全 http）
        # 3. 排除末尾的标点符号
        url_pattern = r'(?:https?://|www\.)[^\s<>"{}|\\^`\[\]\u4e00-\u9fa5]+'
        
        self.combined_re = re.compile(
            '|'.join([
                f'(?P<CODE>{code_block_pattern}|{inline_code_pattern})',
                f'(?P<LINK>{url_pattern})',
            ]),
            re.DOTALL | re.IGNORECASE
        )

    def process_text(self, text: str) -> ProcessedText:
        """
        处理输入文本，分离出用于发送的文本和用于语音合成的文本。
        - 发送文本 (send_text) 包含原始代码和Markdown格式的链接。
        - 语音文本 (speak_text) 将代码和链接替换为占位符（如 "代码" 或 "链接"）。
        """
        clean_text_parts = []
        speak_text_parts = []
        extracted_links = []
        extracted_codes = []
        last_end: int = 0
        matches_found: bool = False

        for match in self.combined_re.finditer(text):
            matches_found = True
            # 添加匹配前的普通文本
            plain_text = text[last_end:match.start()]
            clean_text_parts.append(plain_text)
            speak_text_parts.append(plain_text)

            # 根据匹配的类型处理
            group_name = match.lastgroup
            matched_content = match.group(0)

            if group_name == 'LINK':
                # 对于链接，提取链接，speak_text 使用占位符
                extracted_links.append(matched_content)
                speak_text_parts.append(" 链接 ")
            elif group_name == 'CODE':
                # 对于代码，send_text 使用原始代码，speak_text 使用占位符
                clean_text_parts.append(matched_content)
                extracted_codes.append(matched_content)
                speak_text_parts.append(" 代码 ")
            
            last_end = match.end()

        # 添加最后一个匹配项之后的剩余文本
        remaining_text = text[last_end:]
        clean_text_parts.append(remaining_text)
        speak_text_parts.append(remaining_text)

        # 组合最终的字符串
        clean_text = ''.join(clean_text_parts)
        speak_text = ''.join(speak_text_parts)

        return ProcessedText(
            clean_text=clean_text,
            speak_text=speak_text,
            links=extracted_links,
            codes=extracted_codes,
            has_links_or_code=matches_found
        )


# 全局实例
extractor = CodeAndLinkExtractor()