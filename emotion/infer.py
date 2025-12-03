from typing import List, Optional, Dict
import re

EMOTIONS = ["neutral", "happy", "sad", "angry"]

# 极简启发式情绪分类器，避免引入大模型依赖；后续可替换为 onnx 推理
POS_WORDS = {"开心", "高兴", "喜欢", "太棒了", "哈哈", "lol", ":)", "😀"}
NEG_WORDS = {"难过", "伤心", "失望", "糟糕", "无语", "唉", "sad", ":(", "😢"}
ANG_WORDS = {"气死", "愤怒", "生气", "nm", "tmd", "淦", "怒", "怒了", "😡"}

URL_RE = re.compile(r"https?://|www\.")
# 代码块检测
CODE_BLOCK_RE = re.compile(r'```[a-zA-Z0-9_+-]*\n.*?\n```', re.DOTALL)
INLINE_CODE_RE = re.compile(r'`([^`\n]+)`')


def is_informational(text: str) -> bool:
    # 包含链接/代码/文件提示等，视为信息性，倾向 neutral
    has_url = bool(URL_RE.search(text or ""))
    has_code_block = bool(CODE_BLOCK_RE.search(text or ""))
    # 对于行内代码，只检测包含复杂内容的（不是单个模型名）
    has_inline_code = False
    for match in INLINE_CODE_RE.finditer(text or ""):
        code_content = match.group(1)
        # 如果包含空格、换行符或多个符号，很可能是真正的代码
        if (' ' in code_content or
            '\n' in code_content or
            code_content.count('.') > 1 or
            code_content.count('/') > 1 or
            len(code_content) > 20):
            has_inline_code = True
            break
    
    return has_url or has_code_block or has_inline_code


def classify(text: str, context: Optional[List[str]] = None) -> str:
    # 如果是信息类文本，直接返回 neutral
    if is_informational(text or ""):
        return "neutral"

    t = (text or "").lower()
    score: Dict[str, float] = {"happy": 0.0, "sad": 0.0, "angry": 0.0}

    # 简单计数词典命中
    for w in POS_WORDS:
        if w.lower() in t:
            score["happy"] += 1.0
    for w in NEG_WORDS:
        if w.lower() in t:
            score["sad"] += 1.0
    for w in ANG_WORDS:
        if w.lower() in t:
            score["angry"] += 1.0

    # 感叹号、全大写等作为情绪增强
    if text and "!" in text:
        score["angry"] += 0.5  # 降低感叹号的权重，避免误判
    if (
        text
        and text.strip()
        and text == text.upper()
        and any(c.isalpha() for c in text)
    ):
        score["angry"] += 1.0

    # 上下文弱加权
    if context:
        # 过滤非字符串类型的上下文
        valid_context = [c for c in context if isinstance(c, str)]
        if valid_context:
            ctx = "\n".join(valid_context[-3:]).lower()
            for w in POS_WORDS:
                if w.lower() in ctx:
                    score["happy"] += 0.2
            for w in NEG_WORDS:
                if w.lower() in ctx:
                    score["sad"] += 0.2
            for w in ANG_WORDS:
                if w.lower() in ctx:
                    score["angry"] += 0.2

    # 选最大，否则中性
    label = max(score.keys(), key=lambda k: score[k])
    if score[label] <= 0.5: # 提高阈值，微弱情绪倾向于 neutral
        return "neutral"
    return label
