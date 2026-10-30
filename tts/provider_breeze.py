# -*- coding: utf-8 -*-
"""Breeze Blue TTS Provider - 适配 Breeze Developer API"""

import asyncio
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import aiohttp

from ..utils.audio import validate_audio_file
from .cache import audio_cache
from .errors import http_failure, set_failure

logger = logging.getLogger(__name__)


# Breeze API 的 instructions 与文本同语言；本插件面向中文对话。
BREEZE_EMOTION_INSTRUCTIONS = {
    "happy": "用开心、明亮的语气说这句话，语速轻快。",
    "sad": "用悲伤、低落的语气说这句话，语气放缓。",
    "angry": "用愤怒、压抑着怒火的语气说这句话，咬字有力。",
    "fear": "用害怕、紧张的语气说这句话，声音发虚。",
    "surprise": "用惊讶、意外的语气说这句话，语调上扬。",
    "excited": "用兴奋、充满活力的语气说这句话，情绪饱满。",
    "wronged": "用委屈、快要哭出来的语气说这句话。",
    "cold": "用冷淡、疏离的语气说这句话，缺乏起伏。",
    "melancholy": "用怅然、若有所失的语气说这句话，略带叹息。",
    "gratified": "用欣慰、温柔的语气说这句话。",
    "helpless": "用无奈、疲惫的语气说这句话，带着苦笑。",
    "guilty": "用愧疚、心虚的语气说这句话，声音放低。",
    "relieved": "用释然、放松的语气说这句话，像放下了心事。",
    "jealous": "用嫉妒、酸涩的语气说这句话。",
    "weary": "用厌倦、提不起精神的语气说这句话。",
    "anxious": "用忐忑、不安的语气说这句话，语速略快。",
    "affectionate": "用动情、温柔而投入的语气说这句话。",
}

MAX_INSTRUCTIONS_CHARS = 1000


class BreezeTTS:
    """Breeze Blue 同步 HTTP TTS。

    Breeze 使用 ``POST /v1/text-to-speech/{voice_id}`` 返回二进制音频；
    表现力通过 ``instructions`` 与 ``voice_settings`` 控制。
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str = "breeze-tts-2",
        voice_id: str = "",
        fmt: str = "mp3",
        speed: float = 1.0,
        max_retries: int = 2,
        timeout: int = 30,
        *,
        volume: float = 1.0,
        guidance_scale: float = 1.0,
        instructions: str = "",
        language_code: str = "",
        use_speaker_boost: bool = False,
        emotion_instructions_enable: bool = True,
        sample_rate: int = 24000,
    ):
        self.api_url = (api_url or "https://api.breeze.blue/v1").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.voice = voice_id or ""
        self.voice_id = voice_id or ""
        self.format = (fmt or "mp3").strip().lower()
        self.speed = speed
        self.volume = volume
        self.guidance_scale = guidance_scale
        self.instructions = instructions or ""
        self.language_code = (language_code or "").strip()
        self.use_speaker_boost = bool(use_speaker_boost)
        self.emotion_instructions_enable = bool(emotion_instructions_enable)
        self.sample_rate = sample_rate
        self.max_retries = max_retries
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    def _build_instructions(self, emotion: Optional[str]) -> str:
        parts = []
        base = self.instructions.strip()
        if base:
            parts.append(base)
        if self.emotion_instructions_enable:
            key = str(emotion or "").strip().lower()
            emotion_instruction = BREEZE_EMOTION_INSTRUCTIONS.get(key, "")
            if emotion_instruction:
                parts.append(emotion_instruction)
        return "\n".join(parts)[:MAX_INSTRUCTIONS_CHARS].strip()

    def _build_payload(
        self,
        text: str,
        *,
        speed: Optional[float],
        emotion: Optional[str],
    ) -> dict:
        payload: dict = {"text": str(text or "")}
        if self.model:
            payload["model_id"] = self.model
        if self.language_code:
            payload["language_code"] = self.language_code

        instructions = self._build_instructions(emotion)
        if instructions:
            payload["instructions"] = instructions

        settings: dict = {}
        effective_speed = float(speed) if speed is not None else float(self.speed)
        if abs(effective_speed - 1.0) > 1e-6:
            settings["speed"] = effective_speed
        if abs(float(self.volume) - 1.0) > 1e-6:
            settings["volume"] = float(self.volume)
        if abs(float(self.guidance_scale) - 1.0) > 1e-6:
            settings["guidance_scale"] = float(self.guidance_scale)
        if self.use_speaker_boost:
            settings["use_speaker_boost"] = True
        if settings:
            payload["voice_settings"] = settings
        return payload

    @staticmethod
    def _error_code(payload) -> str:
        return str(payload.get("code", "")) if isinstance(payload, dict) else ""

    @staticmethod
    def _retry_delay(response, fallback: float) -> float:
        headers = getattr(response, "headers", {}) or {}
        raw = headers.get("Retry-After")
        try:
            return min(max(float(raw), 0.5), 30.0)
        except (TypeError, ValueError):
            return fallback

    @audio_cache
    async def synth(
        self,
        text: str,
        voice: str,
        out_dir: Path,
        speed: Optional[float] = None,
        *,
        emotion: Optional[str] = None,
    ) -> Optional[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)

        if not self.api_url or not self.api_key:
            logger.error("BreezeTTS: 缺少 api_url 或 api_key")
            return None

        effective_voice = str(voice or self.voice_id or "").strip()
        if not effective_voice:
            set_failure("voice", "Breeze 未配置音色，请填写默认音色或情绪路由音色映射")
            logger.error("BreezeTTS: 缺少 voice_id")
            return None

        url = f"{self.api_url}/text-to-speech/{quote(effective_voice, safe='')}"
        params = {"output_format": self.format}
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = self._build_payload(text, speed=speed, emotion=emotion)

        out_path = out_dir / f"breeze.{self.format}"

        if self._session is None or self._session.closed:
            client_timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=client_timeout)

        backoff = 1.0

        for attempt in range(1, self.max_retries + 2):
            try:
                async with self._session.post(
                    url,
                    params=params,
                    headers=headers,
                    json=payload,
                ) as response:
                    if 200 <= response.status < 300:
                        content_type = (response.headers.get("Content-Type") or "").lower()
                        if content_type and not (
                            content_type.startswith("audio/")
                            or content_type.startswith("application/octet-stream")
                        ):
                            try:
                                err = await response.json(content_type=None)
                            except Exception:
                                err = {"detail": (await response.text())[:200]}
                            logger.error(
                                "BreezeTTS: 非音频响应 status=%s code=%s",
                                response.status,
                                self._error_code(err),
                            )
                            set_failure("request", "Breeze 返回非音频响应，请检查音色、模型和输出格式")
                            break

                        content = await response.read()
                        if not content:
                            set_failure("audio", "Breeze 返回空音频，请重试")
                            continue

                        await asyncio.to_thread(out_path.write_bytes, content)
                        if not await validate_audio_file(
                            out_path,
                            expected_format=self.format,
                            strict_format=True,
                        ):
                            set_failure("audio", "供应商音频损坏，请重试或检查模型与音频格式")
                            continue

                        logger.info(
                            "BreezeTTS: 生成音频 %s (%d bytes)",
                            out_path,
                            out_path.stat().st_size,
                        )
                        return out_path

                    try:
                        err = await response.json(content_type=None)
                    except Exception:
                        err = {"detail": (await response.text())[:200]}

                    logger.warning(
                        "BreezeTTS: 请求失败 status=%s code=%s attempt=%s",
                        response.status,
                        self._error_code(err),
                        attempt,
                    )
                    http_failure(response.status)
                    if response.status == 402 and self._error_code(err) == "API_KEY_CREDIT_LIMIT_EXCEEDED":
                        set_failure("balance", "Breeze API Key 额度已用尽，请提高或重置 Key 预算，或充值账户")
                    if response.status in (429, 500, 502, 503) and attempt <= self.max_retries:
                        await asyncio.sleep(self._retry_delay(response, backoff))
                        backoff = min(backoff * 2, 8)
                        continue
                    break
            except Exception as exc:
                logger.warning(
                    "BreezeTTS: 请求或音频处理失败 attempt=%s type=%s",
                    attempt,
                    type(exc).__name__,
                )
                set_failure("network", "Breeze 请求或音频处理异常，请检查网络、服务地址和 ffmpeg", True)
                if attempt <= self.max_retries:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 8)
                    continue
                break

        await asyncio.to_thread(out_path.unlink, missing_ok=True)
        logger.error("BreezeTTS: 合成失败")
        return None
