import asyncio
import hashlib
import json
import logging
import os
import uuid
import weakref
from pathlib import Path
from typing import Optional

import aiohttp

from ..utils.audio import append_audio_silence, validate_audio_file


COSYVOICE_TAIL_PADDING_MS = 350
CACHE_SCHEMA_VERSION = 2


class SiliconFlowTTS:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        fmt: str = "mp3",
        speed: float = 1.0,
        max_retries: int = 2,
        timeout: int = 30,
        *,
        gain: float = 5.0,
        sample_rate: Optional[int] = None,
    ):
        self.api_url = (api_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.format = fmt
        self.speed = speed
        self.max_retries = max_retries
        self.timeout = timeout
        self.gain = gain
        self.sample_rate = sample_rate
        self._session: Optional[aiohttp.ClientSession] = None
        self._key_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    async def close(self):
        """关闭 HTTP 会话"""
        if self._session:
            await self._session.close()
            self._session = None

    def _is_audio_response(self, content_type: str) -> bool:
        ct = content_type.lower()
        return ct.startswith("audio/") or ct.startswith("application/octet-stream")

    async def synth(
        self,
        text: str,
        voice: str,
        out_dir: Path,
        speed: Optional[float] = None,
        *,
        emotion: Optional[str] = None,
    ) -> Optional[Path]:
        _ = emotion
        out_dir.mkdir(parents=True, exist_ok=True)

        if not self.api_url or not self.api_key:
            logging.error("SiliconFlowTTS: 缺少 api_url 或 api_key")
            return None

        request_text = str(text or "")
        is_cosyvoice = "cosyvoice" in str(self.model or "").lower()
        tail_padding_ms = COSYVOICE_TAIL_PADDING_MS if is_cosyvoice else 0
        if is_cosyvoice and request_text:
            request_text = request_text.strip()
            closing_chars = "\"'”’」』）》】）)]}"
            suffix = ""
            while request_text and request_text[-1] in closing_chars:
                suffix = request_text[-1] + suffix
                request_text = request_text[:-1].rstrip()
            if request_text and request_text[-1] not in "。！？!?；;.…":
                request_text += "。"
            request_text += suffix

        eff_speed = float(speed) if speed is not None else float(self.speed)

        key = hashlib.sha256(
            json.dumps(
                {
                    "cache_schema": CACHE_SCHEMA_VERSION,
                    "t": request_text,
                    "v": voice,
                    "m": self.model,
                    "s": eff_speed,
                    "f": self.format,
                    "g": self.gain,
                    "sr": self.sample_rate,
                    "tail_padding_ms": tail_padding_ms,
                },
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()[:16]
        out_path = out_dir / f"{key}.{self.format}"
        key_lock = self._key_locks.get(key)
        if key_lock is None:
            key_lock = asyncio.Lock()
            self._key_locks[key] = key_lock

        async with key_lock:
            if out_path.exists() and out_path.stat().st_size > 0:
                if await validate_audio_file(
                    out_path,
                    expected_format=self.format,
                    strict_format=True,
                ):
                    return out_path
                await asyncio.to_thread(out_path.unlink, missing_ok=True)

            url = f"{self.api_url}/audio/speech"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "voice": voice,
                "input": request_text,
                "response_format": self.format,
                "speed": eff_speed,
                "gain": self.gain,
                "stream": False,
            }
            if self.sample_rate:
                payload["sample_rate"] = int(self.sample_rate)

            last_err = None
            backoff = 1.0

            if self._session is None or self._session.closed:
                client_timeout = aiohttp.ClientTimeout(total=self.timeout)
                self._session = aiohttp.ClientSession(timeout=client_timeout)

            for attempt in range(1, self.max_retries + 2):
                raw_path = out_dir / (f".{key}.{uuid.uuid4().hex}.raw.{self.format}")
                padded_path = out_dir / (
                    f".{key}.{uuid.uuid4().hex}.padded.{self.format}"
                )
                try:
                    async with self._session.post(
                        url,
                        headers=headers,
                        json=payload,
                    ) as response:
                        if 200 <= response.status < 300:
                            content_type = response.headers.get("Content-Type", "")
                            if not self._is_audio_response(content_type):
                                try:
                                    err = await response.json()
                                except Exception:
                                    text_content = await response.text()
                                    err = {"error": text_content[:200]}
                                logging.error(
                                    "SiliconFlowTTS: non-audio response, code=%s, detail=%s",
                                    response.status,
                                    err,
                                )
                                last_err = err
                                break

                            content = await response.read()
                            await asyncio.to_thread(raw_path.write_bytes, content)
                            if not await validate_audio_file(
                                raw_path,
                                expected_format=self.format,
                                strict_format=True,
                            ):
                                last_err = {
                                    "error": "Generated audio file validation failed"
                                }
                                continue

                            ready_path = raw_path
                            if tail_padding_ms > 0:
                                try:
                                    ready_path = await append_audio_silence(
                                        raw_path,
                                        padded_path,
                                        tail_padding_ms,
                                        audio_format=self.format,
                                        sample_rate=self.sample_rate,
                                    )
                                    if not await validate_audio_file(
                                        ready_path,
                                        expected_format=self.format,
                                        strict_format=True,
                                    ):
                                        last_err = {
                                            "error": "Padded audio file validation failed"
                                        }
                                        continue
                                except (RuntimeError, ValueError) as exc:
                                    logging.warning(
                                        "SiliconFlowTTS: tail padding unavailable; using original audio: %s",
                                        exc,
                                    )
                                    ready_path = raw_path

                            await asyncio.to_thread(os.replace, ready_path, out_path)
                            logging.info(
                                "SiliconFlowTTS: generated audio file: %s (%d bytes, tail_padding_ms=%d)",
                                out_path,
                                out_path.stat().st_size,
                                tail_padding_ms,
                            )
                            return out_path

                        try:
                            err_detail = await response.json()
                        except Exception:
                            text_content = await response.text()
                            err_detail = {"error": text_content[:200]}

                        logging.warning(
                            "SiliconFlowTTS: request failed (%s) attempt=%s, detail=%s",
                            response.status,
                            attempt,
                            err_detail,
                        )
                        last_err = err_detail
                        if response.status in (429,) or 500 <= response.status < 600:
                            if attempt <= self.max_retries:
                                await asyncio.sleep(backoff)
                                backoff = min(backoff * 2, 8)
                                continue
                        break
                except Exception as e:
                    logging.warning(
                        "SiliconFlowTTS: request or audio processing failed attempt=%s, err=%s",
                        attempt,
                        e,
                    )
                    last_err = str(e)
                    if attempt <= self.max_retries:
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 8)
                        continue
                    break
                finally:
                    await asyncio.to_thread(raw_path.unlink, missing_ok=True)
                    await asyncio.to_thread(padded_path.unlink, missing_ok=True)

            await asyncio.to_thread(out_path.unlink, missing_ok=True)
            logging.error(
                "SiliconFlowTTS: synthesis failed after retries, last_error=%s",
                last_err,
            )
            return None
