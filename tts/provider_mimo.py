# -*- coding: utf-8 -*-
"""MiMo TTS Provider - 适配小米 MiMo TTS API"""

import base64
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Sequence

import aiohttp
import asyncio

from ..utils.audio import validate_audio_file

logger = logging.getLogger(__name__)

# MiMo TTS 预置音色
MIMO_VOICES = {
    "mimo_default": "mimo_default",
    "冰糖": "冰糖",
    "茉莉": "茉莉",
    "苏打": "苏打",
    "白桦": "白桦",
    "Mia": "Mia",
    "Chloe": "Chloe",
    "Milo": "Milo",
    "Dean": "Dean",
}

# MiMo TTS 情绪标签（英文 key -> 中文标签）
MIMO_EMOTION_TAGS = {
    # 基础情绪
    "neutral": "平静",
    "happy": "开心",
    "sad": "悲伤",
    "angry": "愤怒",
    "surprise": "惊讶",
    "fear": "恐惧",
    "excited": "兴奋",
    "wronged": "委屈",
    "cold": "冷漠",
    # 复合情绪
    "melancholy": "怅然",
    "gratified": "欣慰",
    "helpless": "无奈",
    "guilty": "愧疚",
    "relieved": "释然",
    "jealous": "嫉妒",
    "weary": "厌倦",
    "anxious": "忐忑",
    "affectionate": "动情",
    # 整体语调
    "gentle": "温柔",
    "cool": "高冷",
    "lively": "活泼",
    "serious": "严肃",
    "lazy": "慵懒",
    "playful": "俏皮",
    "deep": "深沉",
    "capable": "干练",
    "sharp": "凌厉",
    # 音色定位
    "magnetic": "磁性",
    "mellow": "醇厚",
    "clear": "清亮",
    "ethereal": "空灵",
    "young": "稚嫩",
    "old": "苍老",
    "sweet": "甜美",
    "hoarse": "沙哑",
    "elegant": "醇雅",
    # 人设腔调
    "jiazisheng": "夹子音",
    "yujie": "御姐音",
    "zhengtai": "正太音",
    "dashu": "大叔音",
    "taiwan": "台湾腔",
    # 方言
    "dongbei": "东北话",
    "sichuan": "四川话",
    "henan": "河南话",
    "cantonese": "粤语",
    # 角色扮演
    "sunwukong": "孙悟空",
    "lindaiyu": "林黛玉",
    # 特殊标签
    "sing": "唱歌",
}

# 反向映射（中文标签 -> 英文 key）
MIMO_EMOTION_TAGS_REVERSE = {v: k for k, v in MIMO_EMOTION_TAGS.items()}

MIMO_STYLE_FIELD_NAMES = (
    "overall_tone",
    "timbre_positioning",
    "persona_accent",
    "dialect",
    "role_play",
)


STRICT_SING_CACHE_VERSION = "strict-sing-v2"


SINGING_CHATTER_RE = re.compile(
    r"(好嘞|好的|安排|来首|应景|献上|听好|听着|给你唱|我来唱|今天第几首|"
    r"门票|结一下|小子|老铁|家人们|下面|接下来|一首|唱(?:首|一首|一句|歌)?)"
)


def _is_sing_tag(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"sing", "singing", "唱歌"}


class MiMoTTS:
    """MiMo TTS Provider - 适配小米 MiMo TTS API

    MiMo TTS 使用 /v1/chat/completions 端点，而不是 /audio/speech。
    情绪和唱歌功能通过括号格式控制，如：(唱歌)文本、(开心)文本
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str = "mimo-v2.5-tts",
        voice: str = "mimo_default",
        fmt: str = "mp3",
        speed: float = 1.0,
        max_retries: int = 2,
        timeout: int = 30,
        *,
        style_prompt: str = "",
        dialect: str = "",
        seed_text: str = "",
        sing_voice: str = "",
        overall_tone: str = "",
        timbre_positioning: str = "",
        persona_accent: str = "",
        role_play: str = "",
        director_enable: bool = False,
        director_role: str = "",
        director_scene: str = "",
        director_instruction: str = "",
        director_context: str = "",
    ):
        self.api_url = (api_url or "https://api.xiaomimimo.com/v1").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.voice = voice
        self.format = fmt
        self.speed = speed
        self.max_retries = max_retries
        self.timeout = timeout
        self.style_prompt = style_prompt
        self.overall_tone = overall_tone
        self.timbre_positioning = timbre_positioning
        self.persona_accent = persona_accent
        self.dialect = dialect
        self.role_play = role_play
        self.seed_text = seed_text
        self.director_enable = bool(director_enable)
        self.director_role = director_role
        self.director_scene = director_scene
        self.director_instruction = director_instruction
        self.director_context = director_context
        # Empty sing_voice means "follow the main/default voice".  Use a
        # concrete main voice such as 冰糖/茉莉 if you need stable timbre; the
        # MiMo backend alias "mimo_default" may vary by deployment cluster.
        self.sing_voice = sing_voice or ""
        self._session: Optional[aiohttp.ClientSession] = None

    async def close(self):
        """关闭 HTTP 会话"""
        if self._session:
            await self._session.close()
            self._session = None

    def _build_style_prefix(
        self,
        emotion_tag: Optional[str] = None,
        style_overrides: Optional[Dict[str, str]] = None,
        performance_tags: Optional[Sequence[str]] = None,
    ) -> str:
        """构建情绪标签前缀

        根据小米官方文档，情绪标签使用括号格式：
        - (唱歌)文本
        - (开心)文本
        - （唱歌）文本（全角括号也可以）

        Args:
            emotion_tag: 情绪标签（如"唱歌"、"开心"等）

        Returns:
            括号格式的情绪标签字符串，如果没有情绪标签则返回空字符串
        """
        style_parts: list[str] = []

        # 优先使用传入的情绪标签
        if emotion_tag:
            style_parts.append(emotion_tag)

        overrides = style_overrides or {}

        for field in MIMO_STYLE_FIELD_NAMES:
            value = str(overrides.get(field) or getattr(self, field, "") or "").strip()
            if value:
                style_parts.append(value)

        # 兼容旧配置：额外自由风格标签仍可叠加在最后。
        if self.style_prompt.strip():
            style_parts.append(self.style_prompt.strip())

        # Explicit MiMo performance tags from the current user/LLM turn.
        # They are already whitelist-filtered by the caller and are never inferred here.
        for tag in performance_tags or []:
            value = str(tag or "").strip()
            if value and value not in style_parts:
                style_parts.append(value)

        style_content = "、".join(style_parts).strip()
        if not style_content:
            return ""

        # 使用括号格式（小米官方文档推荐）
        return f"（{style_content}）"

    def _build_user_prompt(self, temporary_director_prompt: Optional[str] = None) -> Optional[str]:
        """构建不会被朗读的 user prompt。

        MiMo 支持在 user message 中放自然语言风格指令/对话历史。
        这里将旧的 seed_text 与新的导演模式字段合并，最终只生成一条
        user message，避免把控制指令混入 assistant 待朗读文本。
        """
        parts: list[str] = []
        seed_text = self.seed_text.strip()
        if seed_text:
            parts.append(seed_text)

        if self.director_enable:
            director_parts: list[str] = []
            role = self.director_role.strip()
            scene = self.director_scene.strip()
            instruction = self.director_instruction.strip()
            context = self.director_context.strip()
            if role:
                director_parts.append(f"角色：{role}")
            if scene:
                director_parts.append(f"场景：{scene}")
            if instruction:
                director_parts.append(f"指导：{instruction}")
            if context:
                director_parts.append(f"上下文：{context}")
            if director_parts:
                parts.append(
                    "请按以下导演模式进行语音表演，这些要求不要朗读出来：\n"
                    + "\n".join(director_parts)
                )

        temp_prompt = str(temporary_director_prompt or "").strip()
        if temp_prompt:
            parts.append("本次临时语音指导：" + temp_prompt)

        return "\n\n".join(parts).strip() or None

    def _prepare_strict_sing_text(self, text: str) -> str:
        """把用户/LLM 的唱歌式回复压成更像歌词的文本。

        MiMo 的“唱歌”标签更容易在歌词体上生效；如果把“好嘞、给你来首、
        门票记得结一下”这类对白一起交给模型，它经常会按台词朗读。
        这里不尝试创作新内容，只做轻量清洗和分行。
        """
        raw = (text or "").strip()
        if not raw:
            return "啦啦啦，啦啦啦\n唱一段小小的歌"

        # 去掉常见引号/标签/前置说明。
        cleaned = re.sub(r"^\s*[《「『“\"']|[》」』”\"']\s*$", "", raw)
        cleaned = re.sub(r"^\s*(?:唱(?:首|一首|一句|歌)?[:：，,\s]*)+", "", cleaned)
        cleaned = SINGING_CHATTER_RE.sub("", cleaned)

        # 去掉明显的闲聊句，保留更像歌词的内容。
        parts = re.split(r"([。！？!?；;，,\n])", cleaned)
        merged: list[str] = []
        for idx in range(0, len(parts), 2):
            sent = (parts[idx] or "").strip()
            punct = parts[idx + 1] if idx + 1 < len(parts) else ""
            if not sent:
                continue
            if re.search(r"(哈哈|呵呵|记得|别忘|怎么样|可以吗|要不要|不是|就是|这个|那个)", sent):
                continue
            merged.append(sent + (punct if punct and punct != "," else ""))

        lyric = "".join(merged).strip(" ，,。；;")
        if not lyric:
            lyric = raw.strip()

        # 分行增强“歌词体”。太短时补一个哼唱尾句，减少纯朗读概率。
        lyric = re.sub(r"[。！？!?；;]+", "\n", lyric)
        lyric = re.sub(r"[，,、]+", "\n", lyric)
        lines = []
        for line in lyric.splitlines():
            line = line.strip(" ，,。；;")
            line = re.sub(r"^[的了啊呀呢嘛吧]+", "", line)
            line = re.sub(r"[的了啊呀呢嘛吧]+$", "", line)
            if line and line not in {"的", "了", "啊", "呀", "呢", "嘛", "吧"}:
                lines.append(line)
        if len("".join(lines)) < 14:
            lines.append("啦啦啦，轻轻唱")
        return "\n".join(lines[:8])

    def _build_strict_sing_user_prompt(self) -> str:
        return (
            "请严格按“唱歌/旋律哼唱”的方式生成音频，不要朗读，不要说话，"
            "不要播报说明。下一条 assistant 内容是歌词，请把每一行都唱出来。"
        )

    def _build_payload(
        self,
        text: str,
        emotion_tag: Optional[str] = None,
        actual_voice: Optional[str] = None,
        *,
        strict_sing: bool = False,
        style_overrides: Optional[Dict[str, str]] = None,
        temporary_director_prompt: Optional[str] = None,
        performance_tags: Optional[Sequence[str]] = None,
    ) -> dict:
        """构建 MiMo TTS API 请求 payload

        Args:
            text: 要合成的文本
            emotion_tag: 情绪标签（如"唱歌"、"开心"等）

        Returns:
            API 请求 payload
        """
        messages: list[dict[str, str]] = []

        # 添加 user prompt（seed text）
        user_prompt = self._build_strict_sing_user_prompt() if strict_sing else self._build_user_prompt(temporary_director_prompt)
        if user_prompt:
            messages.append(
                {
                    "role": "user",
                    "content": user_prompt,
                }
            )

        # 构建 assistant content（带 style 标签）
        synth_text = self._prepare_strict_sing_text(text) if strict_sing else text
        style_prefix = self._build_style_prefix(
            emotion_tag,
            style_overrides=style_overrides,
            performance_tags=performance_tags,
        )
        assistant_content = f"{style_prefix}{synth_text}" if style_prefix else synth_text
        if strict_sing:
            logger.info("MiMoTTS: strict singing content=%r", assistant_content[:200])
        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
            }
        )

        # 构建 audio 参数
        audio_params: dict = {"format": self.format}
        # voicedesign 模型不支持 audio.voice 参数；voiceclone 使用 audio.voice
        # 传入 data:{MIME_TYPE};base64,... 音频样本。
        if "voicedesign" not in self.model:
            audio_params["voice"] = actual_voice or self.voice

        return {
            "model": self.model,
            "messages": messages,
            "audio": audio_params,
        }

    async def synth(
        self,
        text: str,
        voice: str,
        out_dir: Path,
        speed: Optional[float] = None,
        *,
        emotion: Optional[str] = None,
        style_overrides: Optional[Dict[str, str]] = None,
        director_prompt: Optional[str] = None,
        performance_tags: Optional[Sequence[str]] = None,
    ) -> Optional[Path]:
        """合成语音

        Args:
            text: 要合成的文本
            voice: 音色或情绪标签
            out_dir: 输出目录
            speed: 语速（MiMo TTS 暂不支持）
            emotion: 情绪标签

        Returns:
            生成的音频文件路径，失败返回 None
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        # 解析 voice/emotion 参数：
        # - emotion 是插件识别出的情绪 key（如 sing/happy），必须优先用于 MiMo 风格标签
        # - voice 可能是实际音色，也可能是 voice_map 里的中文标签（如 唱歌/开心）
        emotion_tag = None
        actual_voice = self.voice

        if emotion in MIMO_EMOTION_TAGS:
            emotion_tag = MIMO_EMOTION_TAGS[emotion]
        elif emotion in MIMO_EMOTION_TAGS_REVERSE:
            emotion_tag = emotion
        elif emotion and emotion not in ("neutral", "mimo_default"):
            emotion_tag = emotion

        if voice in MIMO_VOICES:
            actual_voice = MIMO_VOICES[voice]
        elif voice in MIMO_EMOTION_TAGS and not emotion_tag:
            # 是英文情绪标签
            emotion_tag = MIMO_EMOTION_TAGS[voice]
        elif voice in MIMO_EMOTION_TAGS_REVERSE and not emotion_tag:
            # 是中文情绪标签（如"唱歌"、"开心"）
            emotion_tag = voice
        elif voice and voice not in ("mimo_default",) and not emotion_tag:
            # 尝试作为情绪标签处理
            emotion_tag = voice

        supports_singing = self.model == "mimo-v2.5-tts"
        strict_sing = supports_singing and (
            _is_sing_tag(emotion) or _is_sing_tag(voice) or _is_sing_tag(emotion_tag)
        )
        if not supports_singing and (_is_sing_tag(emotion) or _is_sing_tag(voice) or _is_sing_tag(emotion_tag)):
            logger.warning("MiMoTTS: model %s does not support singing; ignoring sing tag", self.model)
            emotion_tag = None
        if strict_sing:
            emotion_tag = "唱歌"
            if self.sing_voice:
                if self.sing_voice in MIMO_VOICES:
                    actual_voice = MIMO_VOICES[self.sing_voice]
                else:
                    actual_voice = self.sing_voice

        # 缓存 key
        key = hashlib.sha256(
            json.dumps(
                {
                    "t": text,
                    "v": actual_voice,
                    "m": self.model,
                    "e": emotion_tag,
                    "f": self.format,
                    "strict_sing": STRICT_SING_CACHE_VERSION if strict_sing else "",
                    "style_overrides": style_overrides or {},
                    "director_prompt": director_prompt or "",
                    "performance_tags": list(performance_tags or []),
                },
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()[:16]
        out_path = out_dir / f"{key}.{self.format}"
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path

        if not self.api_url or not self.api_key:
            logger.error("MiMoTTS: 缺少 api_url 或 api_key")
            return None

        # 构建请求
        url = f"{self.api_url}/chat/completions"
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = self._build_payload(
            text,
            emotion_tag,
            actual_voice=actual_voice,
            strict_sing=strict_sing,
            style_overrides=style_overrides,
            temporary_director_prompt=director_prompt,
            performance_tags=performance_tags,
        )
        if strict_sing:
            logger.info("MiMoTTS: strict singing mode enabled")

        last_err = None
        backoff = 1.0

        # 懒加载 session
        if self._session is None or self._session.closed:
            client_timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=client_timeout)

        for attempt in range(1, self.max_retries + 2):
            try:
                async with self._session.post(
                    url, headers=headers, json=payload
                ) as r:
                    if 200 <= r.status < 300:
                        # MiMo TTS 返回 JSON，音频在 message.audio.data 中
                        data = await r.json()
                        choices = data.get("choices") or []
                        first_choice = choices[0] if choices else {}
                        message = first_choice.get("message", {})
                        audio_data = message.get("audio", {}).get("data")

                        if not audio_data:
                            logger.error(f"MiMoTTS: 返回无音频数据: {data}")
                            last_err = {"error": "No audio data in response"}
                            break

                        # 解码 base64 音频并写入文件
                        audio_bytes = base64.b64decode(audio_data)

                        def _write_file():
                            with open(out_path, "wb") as f:
                                f.write(audio_bytes)
                        await asyncio.to_thread(_write_file)

                        # 验证生成的文件
                        if not await validate_audio_file(out_path, expected_format=self.format):
                            logger.error(f"MiMoTTS: 生成的文件验证失败: {out_path}")
                            last_err = {"error": "Generated audio file validation failed"}
                            break

                        logger.info(f"MiMoTTS: 成功生成音频文件: {out_path} ({out_path.stat().st_size}字节)")
                        return out_path

                    # 非 2xx
                    err_detail = None
                    try:
                        err_detail = await r.json()
                    except Exception:
                        text_content = await r.text()
                        err_detail = {"error": text_content[:200]}

                    logger.warning(
                        f"MiMoTTS: 请求失败({r.status}) attempt={attempt}, detail={err_detail}"
                    )
                    last_err = err_detail
                    if r.status in (429,) or 500 <= r.status < 600:
                        if attempt <= self.max_retries:
                            await asyncio.sleep(backoff)
                            backoff = min(backoff * 2, 8)
                            continue
                    break
            except Exception as e:
                logger.warning(f"MiMoTTS: 网络异常 attempt={attempt}, err={e}")
                last_err = str(e)
                if attempt <= self.max_retries:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 8)
                    continue
                break

        # 失败清理
        try:
            def _cleanup():
                if out_path.exists() and out_path.stat().st_size == 0:
                    out_path.unlink()
            await asyncio.to_thread(_cleanup)
        except Exception:
            pass
        logger.error(f"MiMoTTS: 合成失败，已放弃。last_error={last_err}")
        return None
