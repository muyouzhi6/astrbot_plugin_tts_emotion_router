import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import logging
import aiohttp
import asyncio

from ..utils.audio import validate_audio_file


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

    async def close(self):
        """关闭 HTTP 会话"""
        if self._session:
            await self._session.close()
            self._session = None

    def _is_audio_response(self, content_type: str) -> bool:
        ct = content_type.lower()
        return ct.startswith("audio/") or ct.startswith("application/octet-stream")

    async def synth(
        self, text: str, voice: str, out_dir: Path, speed: Optional[float] = None
    ) -> Optional[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)

        if not self.api_url or not self.api_key:
            logging.error("SiliconFlowTTS: 缺少 api_url 或 api_key")
            return None

        # 有效语速：优先使用传入值，其次使用全局默认
        eff_speed = float(speed) if speed is not None else float(self.speed)

        # 缓存 key：文本+voice+model+speed+format+gain+sample_rate
        key = hashlib.sha256(
            json.dumps(
                {
                    "t": text,
                    "v": voice,
                    "m": self.model,
                    "s": eff_speed,
                    "f": self.format,
                    "g": self.gain,
                    "sr": self.sample_rate,
                },
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()[:16]
        out_path = out_dir / f"{key}.{self.format}"
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path

        url = f"{self.api_url}/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "voice": voice,
            "input": text,
            "response_format": self.format,
            "speed": eff_speed,
            "gain": self.gain,
        }
        if self.sample_rate:
            payload["sample_rate"] = int(self.sample_rate)

        last_err = None
        backoff = 1.0
        
        # 懒加载 session
        if self._session is None or self._session.closed:
             client_timeout = aiohttp.ClientTimeout(total=self.timeout)
             self._session = aiohttp.ClientSession(timeout=client_timeout)

        for attempt in range(1, self.max_retries + 2):  # 尝试(重试N次+首次)=N+1 次
            try:
                async with self._session.post(
                    url, headers=headers, json=payload
                ) as r:
                    # 2xx
                    if 200 <= r.status < 300:
                        # Pylance might complain about headers type, but it is MultidictProxy
                        content_type = r.headers.get("Content-Type", "") # type: ignore
                        if not self._is_audio_response(content_type):
                            # 可能是 JSON 错误
                            try:
                                err = await r.json()
                            except Exception:
                                text_content = await r.text()
                                err = {"error": text_content[:200]}
                            logging.error(
                                f"SiliconFlowTTS: 返回非音频内容，code={r.status}, detail={err}"
                            )
                            last_err = err
                            break
                        
                        # 写入文件
                        content = await r.read()
                        
                        def _write_file():
                            with open(out_path, "wb") as f:
                                f.write(content)
                        await asyncio.to_thread(_write_file)
                        
                        # 验证生成的文件 (使用新的异步方法)
                        if not await validate_audio_file(out_path, expected_format=self.format):
                            logging.error(f"SiliconFlowTTS: 生成的文件验证失败: {out_path}")
                            last_err = {"error": "Generated audio file validation failed"}
                            break
                        
                        logging.info(f"SiliconFlowTTS: 成功生成音频文件: {out_path} ({out_path.stat().st_size}字节)")
                        return out_path

                    # 非 2xx
                    err_detail = None
                    try:
                        err_detail = await r.json()
                    except Exception:
                        text_content = await r.text()
                        err_detail = {"error": text_content[:200]}

                    logging.warning(
                        f"SiliconFlowTTS: 请求失败({r.status}) attempt={attempt}, detail={err_detail}"
                    )
                    last_err = err_detail
                    # 429 或 5xx 进行重试
                    if r.status in (429,) or 500 <= r.status < 600:
                        if attempt <= self.max_retries:
                            await asyncio.sleep(backoff)
                            backoff = min(backoff * 2, 8)
                            continue
                    break
            except Exception as e:
                logging.warning(f"SiliconFlowTTS: 网络异常 attempt={attempt}, err={e}")
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
        logging.error(f"SiliconFlowTTS: 合成失败，已放弃。last_error={last_err}")
        return None
