# -*- coding: utf-8 -*-
"""
代码和链接提取器 - 增强版

功能：
1. 代码块提取：支持 ```lang ... ``` 格式，提取语言标识
2. 行内代码智能判定：过滤模型名、版本号、简单文件名等误判
3. 链接增强匹配：支持 Markdown 链接、裸域名、常见 TLD
4. 去重和清理：自动去重、过滤无效内容
"""
from __future__ import annotations

import re
from typing import List, Pattern, Set, Tuple, Optional
from dataclasses import dataclass, field


# ==================== 常量定义 ====================

# 常见顶级域名（用于裸域名匹配）
COMMON_TLDS = frozenset({
    'com', 'org', 'net', 'io', 'dev', 'ai', 'app', 'cn', 'co',
    'edu', 'gov', 'info', 'me', 'tv', 'cc', 'xyz', 'top', 'tech',
    'cloud', 'online', 'site', 'blog', 'wiki', 'docs', 'api',
    # 国家/地区域名
    'uk', 'de', 'fr', 'jp', 'kr', 'ru', 'au', 'ca', 'in', 'br',
})

# 不应被视为行内代码的模式（模型名、版本号等）
INLINE_CODE_EXCLUSIONS = [
    # 模型名称模式
    r'^[A-Z][a-zA-Z]*-\d+(\.\d+)?[A-Za-z]*$',  # GPT-4, GPT-4o, Claude-3
    r'^[A-Z][a-zA-Z]*\d+(\.\d+)?[A-Za-z]*$',   # GPT4, Gemini1.5
    r'^[a-z]+-[a-z]+-\d+[a-z]*$',               # gpt-4-turbo, claude-3-opus
    # 版本号
    r'^v?\d+\.\d+(\.\d+)?(-[a-zA-Z0-9]+)?$',   # v1.0.0, 2.3.4-beta
    r'^v\d+$',                                   # v3, v4
    # 简单文件名（无路径，常见扩展名）
    r'^[a-zA-Z0-9_-]+\.(txt|md|json|yaml|yml|toml|ini|cfg|conf)$',
    # 纯数字或简单标识符
    r'^\d+$',                                   # 纯数字
    r'^[A-Z]{2,5}$',                            # 纯大写缩写 API, SDK
    # 常见缩写词
    r'^(API|SDK|CLI|GUI|URL|URI|HTTP|HTTPS|SSH|FTP|DNS|TCP|UDP|IP)$',
]

# 真正的代码特征模式（应被识别为代码）
INLINE_CODE_INCLUSIONS = [
    # 函数调用
    r'^[a-zA-Z_][a-zA-Z0-9_]*\(.*\)$',         # func(), print("hello")
    # 变量赋值
    r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*.+$',      # x = 1
    # 方法链
    r'^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)+$',  # obj.method.call
    # 包含特殊代码字符
    r'.*[{}\[\]();].*',                         # 包含括号等
    # 命令行
    r'^(pip|npm|yarn|cargo|go|apt|brew|choco)\s+.+$',  # 包管理命令
    # 路径（带目录分隔符）
    r'^[./\\].*[/\\].*$',                       # ./path/to/file
    # 环境变量
    r'^\$[A-Z_][A-Z0-9_]*$',                   # $HOME, $PATH
    r'^%[A-Z_][A-Z0-9_]*%$',                   # %PATH% (Windows)
]

# 域名白名单（常见技术网站，用于裸域名识别）
DOMAIN_WHITELIST = frozenset({
    'github.com', 'gitlab.com', 'bitbucket.org',
    'stackoverflow.com', 'stackexchange.com',
    'npmjs.com', 'pypi.org', 'crates.io', 'rubygems.org',
    'docker.com', 'hub.docker.com',
    'medium.com', 'dev.to', 'hashnode.dev',
    'google.com', 'youtube.com', 'twitter.com', 'x.com',
    'discord.com', 'discord.gg', 'telegram.org', 't.me',
    'reddit.com', 'v2ex.com', 'zhihu.com', 'juejin.cn',
    'mozilla.org', 'w3.org', 'w3schools.com',
    'python.org', 'rust-lang.org', 'golang.org', 'nodejs.org',
    'microsoft.com', 'azure.com', 'aws.amazon.com',
    'openai.com', 'anthropic.com', 'huggingface.co',
})


@dataclass
class CodeBlockInfo:
    """代码块信息"""
    content: str           # 完整内容（包含 ``` 标记）
    code: str              # 纯代码内容
    language: Optional[str] = None  # 语言标识（如 python, js）


@dataclass
class ProcessedText:
    """处理后的文本结果"""
    clean_text: str                           # 清理后的文本（保留代码块）
    speak_text: str                           # 用于 TTS 的文本（代码/链接替换为占位符）
    links: List[str] = field(default_factory=list)              # 提取的链接列表
    codes: List[str] = field(default_factory=list)              # 提取的代码列表
    code_blocks: List[CodeBlockInfo] = field(default_factory=list)  # 代码块详细信息
    has_links_or_code: bool = False           # 是否包含链接或代码


class CodeAndLinkExtractor:
    """
    增强版代码和链接提取器
    
    功能特性：
    1. 代码块语言识别：提取 ```python 中的语言标识
    2. 智能行内代码判定：过滤模型名、版本号、简单文件名
    3. 增强链接匹配：支持 Markdown 链接 [text](url)、裸域名
    4. 自动去重：链接和代码自动去重
    5. 清理优化：过滤无效匹配、标准化输出
    """

    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """编译所有正则表达式"""
        
        # 1. 代码块模式（带语言标识捕获）
        # 匹配 ```lang\ncode\n``` 或 ```\ncode\n```
        self._code_block_re = re.compile(
            r'```(\w*)\n?([\s\S]*?)```',
            re.DOTALL
        )
        
        # 2. 行内代码模式
        self._inline_code_re = re.compile(r'`([^`\n]+)`')
        
        # 3. Markdown 链接模式 [text](url)
        self._md_link_re = re.compile(
            r'\[([^\]]*)\]\(([^)]+)\)',
            re.IGNORECASE
        )
        
        # 4. 标准 URL 模式（http/https/ftp）
        self._url_re = re.compile(
            r'(?:https?|ftp)://[^\s<>"\'`\[\]{}|\\^（）【】\u4e00-\u9fa5]+',
            re.IGNORECASE
        )
        
        # 5. 裸域名模式（无协议前缀）
        # 匹配 domain.tld/path 或 subdomain.domain.tld/path
        self._bare_domain_re = re.compile(
            r'\b([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+('
            + '|'.join(COMMON_TLDS)
            + r')(/[^\s<>"\'`\[\]{}|\\^（）【】\u4e00-\u9fa5]*)?',
            re.IGNORECASE
        )
        
        # 6. 编译排除和包含模式
        self._exclusion_patterns = [re.compile(p, re.IGNORECASE) for p in INLINE_CODE_EXCLUSIONS]
        self._inclusion_patterns = [re.compile(p, re.IGNORECASE) for p in INLINE_CODE_INCLUSIONS]
        
        # 7. 综合匹配模式（用于单次遍历）
        # 优先级：代码块 > Markdown 链接 > 行内代码 > URL > 裸域名
        self._combined_re = re.compile(
            '|'.join([
                r'(?P<CODEBLOCK>```\w*\n?[\s\S]*?```)',
                r'(?P<MDLINK>\[[^\]]*\]\([^)]+\))',
                r'(?P<INLINE>`[^`\n]+`)',
                r'(?P<URL>(?:https?|ftp)://[^\s<>"\'`\[\]{}|\\^（）【】\u4e00-\u9fa5]+)',
                r'(?P<BARE>(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+(?:' + '|'.join(COMMON_TLDS) + r')(?:/[^\s<>"\'`\[\]{}|\\^（）【】\u4e00-\u9fa5]*)?)',
            ]),
            re.DOTALL | re.IGNORECASE
        )

    def _is_valid_inline_code(self, content: str) -> bool:
        """
        判断行内代码是否为有效代码（非模型名、版本号等）
        
        Args:
            content: 反引号内的内容（不含反引号）
            
        Returns:
            True 如果是有效代码，False 如果应该忽略
        """
        content = content.strip()
        
        # 空内容忽略
        if not content:
            return False
        
        # 太短的内容（单个字符）忽略
        if len(content) <= 1:
            return False
        
        # 检查是否匹配包含模式（强制识别为代码）
        for pattern in self._inclusion_patterns:
            if pattern.match(content):
                return True
        
        # 检查是否匹配排除模式（不识别为代码）
        for pattern in self._exclusion_patterns:
            if pattern.match(content):
                return False
        
        # 默认：如果包含代码特征字符，认为是代码
        code_chars = {'(', ')', '{', '}', '[', ']', ';', '=', '+', '-', '*', '/', '<', '>', '!', '&', '|'}
        if any(c in content for c in code_chars):
            return True
        
        # 如果包含多个点号（可能是方法链或包名）
        if content.count('.') >= 2:
            return True
        
        # 如果包含下划线（常见变量命名）
        if '_' in content and not content.startswith('_'):
            return True
        
        # 其他情况默认为代码
        return True

    def _is_valid_bare_domain(self, domain: str) -> bool:
        """
        判断裸域名是否为有效链接
        
        Args:
            domain: 匹配到的裸域名
            
        Returns:
            True 如果是有效链接
        """
        domain_lower = domain.lower().rstrip('/')
        
        # 在白名单中
        base_domain = domain_lower.split('/')[0]
        if base_domain in DOMAIN_WHITELIST:
            return True
        
        # 检查是否有路径（更可能是真正的链接）
        if '/' in domain:
            return True
        
        # 检查是否是子域名
        parts = base_domain.split('.')
        if len(parts) >= 3:  # 如 docs.python.org
            return True
        
        # 简单的两段域名，需要更严格判断
        if len(parts) == 2:
            # 检查 TLD 是否常见
            tld = parts[-1]
            if tld in {'com', 'org', 'net', 'io', 'dev', 'ai', 'cn'}:
                return True
        
        return False

    def _clean_link(self, url: str) -> str:
        """
        清理链接，移除尾部标点符号
        
        Args:
            url: 原始链接
            
        Returns:
            清理后的链接
        """
        # 移除尾部常见标点
        while url and url[-1] in '.,;:!?。，；：！？)）]】>》':
            url = url[:-1]
        return url

    def _extract_code_block_info(self, match_content: str) -> CodeBlockInfo:
        """
        从代码块匹配中提取详细信息
        
        Args:
            match_content: 完整的代码块内容（包含 ``` 标记）
            
        Returns:
            CodeBlockInfo 对象
        """
        m = self._code_block_re.match(match_content)
        if m:
            language = m.group(1).strip() or None
            code = m.group(2).strip()
            return CodeBlockInfo(
                content=match_content,
                code=code,
                language=language
            )
        return CodeBlockInfo(
            content=match_content,
            code=match_content.strip('`').strip()
        )

    def process_text(self, text: str) -> ProcessedText:
        """
        处理输入文本，分离出用于发送的文本和用于语音合成的文本。
        
        Args:
            text: 输入文本
            
        Returns:
            ProcessedText 对象，包含：
            - clean_text: 清理后的文本（保留代码块，移除情绪标记等）
            - speak_text: 用于 TTS 的文本（代码/链接替换为占位符）
            - links: 提取的链接列表（去重）
            - codes: 提取的代码列表
            - code_blocks: 代码块详细信息
            - has_links_or_code: 是否包含链接或代码
        """
        if not text:
            return ProcessedText(
                clean_text="",
                speak_text="",
                links=[],
                codes=[],
                code_blocks=[],
                has_links_or_code=False
            )
        
        clean_text_parts: List[str] = []
        speak_text_parts: List[str] = []
        extracted_links: List[str] = []
        extracted_codes: List[str] = []
        code_blocks: List[CodeBlockInfo] = []
        seen_links: Set[str] = set()
        last_end: int = 0
        matches_found: bool = False
        
        for match in self._combined_re.finditer(text):
            group_name = match.lastgroup
            matched_content = match.group(0)
            
            # 添加匹配前的普通文本
            plain_text = text[last_end:match.start()]
            clean_text_parts.append(plain_text)
            speak_text_parts.append(plain_text)
            
            if group_name == 'CODEBLOCK':
                # 代码块：保留在 clean_text，TTS 替换为占位符
                matches_found = True
                clean_text_parts.append(matched_content)
                extracted_codes.append(matched_content)
                code_blocks.append(self._extract_code_block_info(matched_content))
                speak_text_parts.append(" 代码 ")
                
            elif group_name == 'MDLINK':
                # Markdown 链接 [text](url)
                md_match = self._md_link_re.match(matched_content)
                if md_match:
                    link_text = md_match.group(1)
                    link_url = self._clean_link(md_match.group(2))
                    
                    if link_url and link_url not in seen_links:
                        matches_found = True
                        seen_links.add(link_url)
                        extracted_links.append(link_url)
                        # clean_text 保留显示文本
                        clean_text_parts.append(link_text if link_text else "链接")
                        speak_text_parts.append(f" {link_text if link_text else '链接'} ")
                    else:
                        # 重复链接，只保留文本
                        clean_text_parts.append(link_text if link_text else "")
                        speak_text_parts.append(link_text if link_text else "")
                        
            elif group_name == 'INLINE':
                # 行内代码
                inner_content = matched_content[1:-1]  # 去掉反引号
                
                if self._is_valid_inline_code(inner_content):
                    matches_found = True
                    clean_text_parts.append(matched_content)
                    extracted_codes.append(matched_content)
                    speak_text_parts.append(f" {inner_content} ")  # 尝试朗读内容
                else:
                    # 不是有效代码，保留原样
                    clean_text_parts.append(inner_content)
                    speak_text_parts.append(inner_content)
                    
            elif group_name == 'URL':
                # 标准 URL
                link = self._clean_link(matched_content)
                if link and link not in seen_links:
                    matches_found = True
                    seen_links.add(link)
                    extracted_links.append(link)
                    # clean_text 保留原始链接，speak_text 替换为占位符
                    clean_text_parts.append(link)
                    speak_text_parts.append(" 链接 ")
                else:
                    # 重复链接，保留原文但不重复提取
                    clean_text_parts.append(matched_content)

            elif group_name == 'BARE':
                # 裸域名
                if self._is_valid_bare_domain(matched_content):
                    link = self._clean_link(matched_content)
                    if link and link not in seen_links:
                        matches_found = True
                        seen_links.add(link)
                        extracted_links.append(link)
                        # clean_text 保留原始链接，speak_text 替换为占位符
                        clean_text_parts.append(link)
                        speak_text_parts.append(" 链接 ")
                    else:
                        # 重复链接，保留原文
                        clean_text_parts.append(matched_content)
                else:
                    # 不是有效域名，保留原样
                    clean_text_parts.append(matched_content)
                    speak_text_parts.append(matched_content)
            
            last_end = match.end()
        
        # 添加最后一个匹配项之后的剩余文本
        remaining_text = text[last_end:]
        clean_text_parts.append(remaining_text)
        speak_text_parts.append(remaining_text)
        
        # 组合最终的字符串
        clean_text = ''.join(clean_text_parts)
        speak_text = ''.join(speak_text_parts)
        
        # 清理多余空格
        speak_text = re.sub(r'\s+', ' ', speak_text).strip()
        
        return ProcessedText(
            clean_text=clean_text,
            speak_text=speak_text,
            links=extracted_links,
            codes=extracted_codes,
            code_blocks=code_blocks,
            has_links_or_code=matches_found
        )


# 全局实例
extractor = CodeAndLinkExtractor()