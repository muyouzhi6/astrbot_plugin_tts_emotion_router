from typing import List, Optional, Dict, Pattern, Set
import re

from ..core.constants import DEFAULT_EMOTION_KEYWORDS_LIST, EMOTION_KEYWORDS, EMOTIONS

# 极简启发式情绪分类器，避免引入大模型依赖；后续可替换为 onnx 推理
DEFAULT_KEYWORDS: Dict[str, Set[str]] = {
    k: set(v) for k, v in DEFAULT_EMOTION_KEYWORDS_LIST.items()
}
# Do not try to derive keyword strings from regex.pattern: escaped Chinese
# sequences such as "\u59d4\u5c48" would become literal backslash text.
# The compiled EMOTION_KEYWORDS regexes below provide the full 18-emotion
# coverage; DEFAULT_KEYWORDS remains the user-editable simple keyword layer.

URL_RE: Pattern = re.compile(r"https?://|www\.")
SING_INTENT_RE: Pattern = re.compile(
    r"(\b(?:sing|singing)\b|唱歌|唱一?首(?:歌)?|唱首歌|唱一句|"
    r"来一?首(?:歌)?|给.+唱|唱给|献唱|清唱|哼一?段|哼首|开嗓|大海啊大海)",
    re.I,
)
WEAK_SING_KEYWORDS = {"唱", "歌", "歌曲", "歌词"}


def _is_strong_sing_keyword(keyword: str, text: str, lowered: str) -> bool:
    word = str(keyword or "").strip()
    if not word or word in WEAK_SING_KEYWORDS:
        return False
    if word.lower() in {"sing", "singing"}:
        return bool(re.search(rf"\b{re.escape(word)}\b", lowered, re.I))
    return word in text
# 代码块检测
CODE_BLOCK_RE: Pattern = re.compile(r'```[a-zA-Z0-9_+-]*\n.*?\n```', re.DOTALL)
INLINE_CODE_RE: Pattern = re.compile(r'`([^`\n]+)`')


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


def classify(text: str, context: Optional[List[str]] = None, keywords: Optional[Dict[str, Set[str]]] = None) -> str:
    # 如果是信息类文本，直接返回 neutral
    if is_informational(text or ""):
        return "neutral"

    t = (text or "").lower()

    # 唱歌是用户的明确功能意图，必须高于普通情绪识别。
    if SING_INTENT_RE.search(text or ""):
        return "sing"

    # 使用传入的关键词或默认关键词。支持配置里的扩展情绪（尤其 sing）。
    kw_map = keywords if keywords else DEFAULT_KEYWORDS

    # 唱歌是明确意图，不应该被感叹号或其他情绪抢走。
    for w in kw_map.get("sing", set()):
        if _is_strong_sing_keyword(w, text or "", t):
            return "sing"

    score: Dict[str, float] = {emo: 0.0 for emo in EMOTIONS if emo not in ("neutral", "sing")}
    for emo in kw_map.keys():
        if emo not in ("neutral", "sing"):
            score.setdefault(emo, 0.0)

    # 简单计数词典命中
    for emo, words in kw_map.items():
        if emo == "sing":
            continue
        if emo in score:
            for w in words:
                if w and w.lower() in t:
                    score[emo] += 1.0

    # 内置正则覆盖 18 种情绪。
    for emo, pattern in EMOTION_KEYWORDS.items():
        if emo in ("sing", "neutral"):
            continue
        if emo in score and pattern.search(text or ""):
            score[emo] += 1.0

    # 感叹号、全大写等作为情绪增强
    if text and "!" in text:
        score["angry"] += 0.5  # 降低感叹号的权重，避免误判
    if (
        text
        and text.strip()
        and text == text.upper()
        and any(("a" <= c.lower() <= "z") for c in text)
    ):
        score["angry"] += 1.0

    # 上下文弱加权
    if context:
        # 过滤非字符串类型的上下文
        valid_context = [c for c in context if isinstance(c, str)]
        if valid_context:
            ctx = "\n".join(valid_context[-3:]).lower()
            # 使用相同的关键词映射进行上下文加权
            for emo, words in kw_map.items():
                if emo == "sing":
                    continue
                if emo in score:
                    for w in words:
                        if w and w.lower() in ctx:
                            score[emo] += 0.2

    # 选最大，否则中性
    label = max(score.keys(), key=lambda k: score[k])
    if score[label] <= 0.5: # 提高阈值，微弱情绪倾向于 neutral
        return "neutral"
    return label
