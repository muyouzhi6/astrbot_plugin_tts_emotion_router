import asyncio
import base64
import copy
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

import aiohttp

from ..utils.audio import validate_audio_file


logger = logging.getLogger(__name__)


class MiniMaxTTS:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        *,
        fmt: str = "mp3",
        speed: float = 1.0,
        voice_id: str = "",
        vol: float = 1.0,
        pitch: int = 0,
        default_emotion: str = "neutral",
        sample_rate: int = 32000,
        bitrate: int = 128000,
        channel: int = 1,
        output_format: str = "hex",
        language_boost: str = "",
        proxy: str = "",
        voice_modify: Optional[dict] = None,
        timber_weights: Optional[list] = None,
        subtitle_enable: bool = False,
        pronunciation_dict: Optional[dict] = None,
        aigc_watermark: bool = False,
        max_retries: int = 2,
        timeout: int = 30,
    ):
        self.api_url = api_url.strip() or "https://api.minimaxi.com/v1/t2a_v2"
        self.api_key = api_key.strip()
        self.model = model or "speech-2.8-hd"
        self.format = (fmt or "mp3").lower()
        self.speed = float(speed)
        self.voice_id = voice_id or ""
        self.vol = float(vol)
        self.pitch = int(pitch)
        self.default_emotion = default_emotion or "neutral"
        self.sample_rate = int(sample_rate)
        self.bitrate = int(bitrate)
        self.channel = int(channel)
        self.output_format = str(output_format or "hex").strip().lower() or "hex"
        self.language_boost = str(language_boost or "").strip()
        self.proxy = str(proxy or "").strip() or None
        self.voice_modify = copy.deepcopy(voice_modify or {})
        self.timber_weights = copy.deepcopy(timber_weights or [])
        self.subtitle_enable = bool(subtitle_enable)
        self.pronunciation_dict = copy.deepcopy(pronunciation_dict or {})
        self.aigc_watermark = bool(aigc_watermark)
        self.max_retries = max(0, int(max_retries))
        self.timeout = max(5, int(timeout))
        self.transport_mode = "sync_http"
        self.last_response_meta: Optional[dict[str, Any]] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            client_timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=client_timeout)

    @staticmethod
    def _looks_like_hex(s: str) -> bool:
        if len(s) < 4 or len(s) % 2 != 0:
            return False
        try:
            int(s[:16], 16)
            return all(c in "0123456789abcdefABCDEF" for c in s[:64])
        except Exception:
            return False

    @staticmethod
    async def _write_bytes(path: Path, content: bytes) -> None:
        def _write():
            with open(path, "wb") as f:
                f.write(content)

        await asyncio.to_thread(_write)

    async def _download_to_path(self, url: str, out_path: Path) -> bool:
        if not url:
            return False
        await self._ensure_session()
        try:
            assert self._session is not None
            async with self._session.get(url, proxy=self.proxy) as response:
                if response.status != 200:
                    return False
                content = await response.read()
                if not content:
                    return False
                await self._write_bytes(out_path, content)
                return True
        except Exception:
            return False

    def _build_sync_http_payload(
        self,
        text: str,
        *,
        voice: str,
        speed: float,
        emotion: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "text": text,
            "stream": False,
            "voice_setting": {
                "voice_id": voice,
                "speed": speed,
                "vol": self.vol,
                "pitch": self.pitch,
                "emotion": emotion,
            },
            "audio_setting": {
                "sample_rate": self.sample_rate,
                "bitrate": self.bitrate,
                "format": self.format,
                "channel": self.channel,
            },
            "output_format": self.output_format,
            "subtitle_enable": self.subtitle_enable,
            "aigc_watermark": self.aigc_watermark,
        }

        if self.language_boost:
            payload["language_boost"] = self.language_boost
        if self.voice_modify:
            payload["voice_modify"] = copy.deepcopy(self.voice_modify)
        if self.timber_weights:
            payload["timber_weights"] = copy.deepcopy(self.timber_weights)
        if self.pronunciation_dict:
            payload["pronunciation_dict"] = copy.deepcopy(self.pronunciation_dict)

        return payload

    def _extract_response_meta(self, data: dict[str, Any]) -> dict[str, Any]:
        body = data.get("data", {}) or {}
        extra_info = body.get("extra_info")
        if not isinstance(extra_info, dict):
            extra_info = data.get("extra_info") if isinstance(data.get("extra_info"), dict) else {}

        return {
            "status_code": (data.get("base_resp") or {}).get("status_code"),
            "status_msg": (data.get("base_resp") or {}).get("status_msg"),
            "usage_characters": extra_info.get("usage_characters"),
            "audio_length": extra_info.get("audio_length"),
            "invisible_character_ratio": extra_info.get("invisible_character_ratio"),
        }

    def _log_response_meta(self, meta: dict[str, Any]) -> None:
        compact_meta = {key: value for key, value in meta.items() if value not in (None, "", [], {})}
        if compact_meta:
            logger.info("MiniMaxTTS response meta: %s", compact_meta)

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

        if not self.api_key:
            logger.error("MiniMaxTTS: missing api key")
            return None

        effective_speed = float(speed) if speed is not None else float(self.speed)
        effective_voice = voice or self.voice_id
        effective_emotion = (emotion or self.default_emotion or "neutral").lower()

        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "text": text,
                    "voice": effective_voice,
                    "speed": effective_speed,
                    "emotion": effective_emotion,
                    "model": self.model,
                    "fmt": self.format,
                    "sr": self.sample_rate,
                    "br": self.bitrate,
                    "ch": self.channel,
                    "vol": self.vol,
                    "pitch": self.pitch,
                    "output_format": self.output_format,
                    "language_boost": self.language_boost,
                    "proxy": self.proxy,
                    "voice_modify": self.voice_modify,
                    "timber_weights": self.timber_weights,
                    "pronunciation_dict": self.pronunciation_dict,
                    "subtitle_enable": self.subtitle_enable,
                    "aigc_watermark": self.aigc_watermark,
                },
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()[:16]

        out_path = out_dir / f"{cache_key}.{self.format}"
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path

        if self.transport_mode != "sync_http":
            logger.error("MiniMaxTTS transport not implemented: %s", self.transport_mode)
            return None

        payload = self._build_sync_http_payload(
            text,
            voice=effective_voice,
            speed=effective_speed,
            emotion=effective_emotion,
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        await self._ensure_session()
        last_error = None
        backoff = 1.0

        for attempt in range(1, self.max_retries + 2):
            try:
                assert self._session is not None
                async with self._session.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    proxy=self.proxy,
                ) as resp:
                    content_type = (resp.headers.get("Content-Type") or "").lower()
                    if 200 <= resp.status < 300:
                        if content_type.startswith("audio/"):
                            raw = await resp.read()
                            if not raw:
                                last_error = "empty audio response"
                                break
                            await self._write_bytes(out_path, raw)
                        else:
                            data = await resp.json(content_type=None)
                            meta = self._extract_response_meta(data)
                            self.last_response_meta = meta
                            self._log_response_meta(meta)

                            if (data.get("base_resp") or {}).get("status_code", 0) != 0:
                                last_error = (data.get("base_resp") or {}).get("status_msg")
                                break

                            body = data.get("data", {}) or {}
                            audio_text = str(
                                body.get("audio")
                                or body.get("audio_hex")
                                or body.get("audio_base64")
                                or ""
                            ).strip()
                            audio_file = str(body.get("audio_file") or "").strip()

                            if audio_text:
                                if self._looks_like_hex(audio_text):
                                    raw = bytes.fromhex(audio_text)
                                else:
                                    raw = base64.b64decode(audio_text)
                                await self._write_bytes(out_path, raw)
                            elif audio_file:
                                downloaded = await self._download_to_path(audio_file, out_path)
                                if not downloaded:
                                    last_error = "download audio_file failed"
                                    break
                            else:
                                last_error = "no audio data in minimax response"
                                break

                        if not await validate_audio_file(out_path, expected_format=self.format):
                            last_error = "audio file validation failed"
                            break
                        return out_path

                    try:
                        err = await resp.json(content_type=None)
                    except Exception:
                        err = await resp.text()
                    last_error = f"http {resp.status}: {err}"
                    if resp.status in (429,) or 500 <= resp.status < 600:
                        if attempt <= self.max_retries:
                            await asyncio.sleep(backoff)
                            backoff = min(backoff * 2, 8)
                            continue
                    break
            except Exception as e:
                last_error = str(e)
                if attempt <= self.max_retries:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 8)
                    continue
                break

        try:
            if out_path.exists() and out_path.stat().st_size == 0:
                out_path.unlink()
        except Exception:
            pass

        logger.error("MiniMaxTTS synth failed: %s", last_error)
        return None
