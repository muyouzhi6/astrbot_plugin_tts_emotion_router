import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import logging
import aiohttp
import asyncio


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

    def _is_audio_response(self, content_type: str) -> bool:
        ct = content_type.lower()
        return ct.startswith("audio/") or ct.startswith("application/octet-stream")

    def _validate_generated_file(self, file_path: Path) -> bool:
        """验证生成的音频文件是否有效"""
        try:
            if not file_path.exists():
                logging.error(f"SiliconFlowTTS: 文件不存在: {file_path}")
                return False
            
            file_size = file_path.stat().st_size
            if file_size == 0:
                logging.error(f"SiliconFlowTTS: 文件为空: {file_path}")
                return False
            
            if file_size < 100:  # 小于100字节通常是无效的音频文件
                logging.error(f"SiliconFlowTTS: 文件太小({file_size}字节): {file_path}")
                return False
            
            # 检查文件头部是否符合音频格式
            try:
                with open(file_path, "rb") as f:
                    header = f.read(12)
                
                # 检查常见音频格式的文件头
                if self.format.lower() == "mp3":
                    # MP3文件应该以ID3标签或者MPEG帧同步字开始
                    if header.startswith(b"ID3") or header.startswith(b"\xff\xfb") or header.startswith(b"\xff\xfa"):
                        return True
                    # 也可能直接是MPEG帧
                    if len(header) >= 2 and header[0] == 0xff and (header[1] & 0xe0) == 0xe0:
                        return True
                elif self.format.lower() == "wav":
                    # WAV文件应该以RIFF开始，后跟WAVE
                    if header.startswith(b"RIFF") and b"WAVE" in header:
                        return True
                elif self.format.lower() == "opus":
                    # Opus文件通常在OGG容器中
                    if header.startswith(b"OggS"):
                        return True
                
                # 如果格式检查失败，但文件大小合理，给出警告但允许通过
                logging.warning(f"SiliconFlowTTS: 文件格式验证失败，但继续使用: {file_path}")
                return True
                
            except Exception as e:
                logging.warning(f"SiliconFlowTTS: 文件头验证异常: {e}")
                return True  # 验证异常时允许通过
            
        except Exception as e:
            logging.error(f"SiliconFlowTTS: 文件验证失败: {e}")
            return False

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
        
        client_timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            for attempt in range(1, self.max_retries + 2):  # 尝试(重试N次+首次)=N+1 次
                try:
                    async with session.post(
                        url, headers=headers, json=payload
                    ) as r:
                        # 2xx
                        if 200 <= r.status < 300:
                            content_type = r.headers.get("Content-Type", "")
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
                            with open(out_path, "wb") as f:
                                f.write(content)
                            
                            # 验证生成的文件
                            if not self._validate_generated_file(out_path):
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
            if out_path.exists() and out_path.stat().st_size == 0:
                out_path.unlink()
        except Exception:
            pass
        logging.error(f"SiliconFlowTTS: 合成失败，已放弃。last_error={last_err}")
        return None
