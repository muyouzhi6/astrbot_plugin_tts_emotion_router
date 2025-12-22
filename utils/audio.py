import logging
import asyncio
from pathlib import Path
import time
from typing import List, Optional

from ..core.constants import (
    AUDIO_CLEANUP_TTL_SECONDS,
    AUDIO_MIN_VALID_SIZE,
    AUDIO_VALID_EXTENSIONS,
)

logger = logging.getLogger(__name__)


def ensure_dir(p: Path):
    """
    同步确保目录存在（非阻塞不严重，可保持同步或在初始化调用）。
    如果需要在运行时频繁调用，建议改用 async_ensure_dir。
    """
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


async def async_ensure_dir(p: Path):
    """异步确保目录存在。"""
    await asyncio.to_thread(ensure_dir, p)


async def cleanup_dir(root: Path, ttl_seconds: int = AUDIO_CLEANUP_TTL_SECONDS):
    """异步清理目录。"""
    def _cleanup():
        try:
            if not root.exists():
                return
            now = time.time()
            for f in root.glob("**/*"):
                try:
                    if f.is_file() and (now - f.stat().st_mtime) > ttl_seconds:
                        f.unlink()
                except Exception:
                    pass
        except Exception:
            pass
            
    await asyncio.to_thread(_cleanup)


async def validate_audio_file(audio_path: Path, expected_format: Optional[str] = None) -> bool:
    """
    异步验证音频文件是否有效。
    
    Args:
        audio_path: 音频文件路径
        expected_format: 期望的音频格式 (mp3, wav, opus 等)，如果提供则进行文件头检查
        
    Returns:
        bool: 是否有效
    """
    return await asyncio.to_thread(_validate_audio_file_sync, audio_path, expected_format)


def _validate_audio_file_sync(audio_path: Path, expected_format: Optional[str] = None) -> bool:
    """验证音频文件是否有效（同步实现）。"""
    try:
        if not audio_path.exists():
            logger.error(f"validate_audio_file: file not found: {audio_path}")
            return False
        
        file_size = audio_path.stat().st_size
        if file_size == 0:
            logger.error(f"validate_audio_file: file is empty: {audio_path}")
            return False
        
        if file_size < AUDIO_MIN_VALID_SIZE:
            logger.error(f"validate_audio_file: file too small ({file_size} bytes): {audio_path}")
            return False
        
        # 扩展名检查
        if audio_path.suffix.lower() and audio_path.suffix.lower() not in AUDIO_VALID_EXTENSIONS:
            logger.warning(f"validate_audio_file: unexpected extension: {audio_path}")

        # 文件头检查
        if expected_format:
            try:
                with open(audio_path, "rb") as f:
                    header = f.read(12)
                
                fmt = expected_format.lower()
                if fmt == "mp3":
                    # MP3: ID3 or sync word
                    if not (header.startswith(b"ID3") or header.startswith(b"\xff\xfb") or header.startswith(b"\xff\xfa") or (len(header) >= 2 and header[0] == 0xff and (header[1] & 0xe0) == 0xe0)):
                        logger.warning(f"validate_audio_file: MP3 header check failed for {audio_path}, but proceeding")
                elif fmt == "wav":
                    # WAV: RIFF ... WAVE
                    if not (header.startswith(b"RIFF") and b"WAVE" in header):
                        logger.warning(f"validate_audio_file: WAV header check failed for {audio_path}, but proceeding")
                elif fmt == "opus":
                    # Opus: OggS
                    if not header.startswith(b"OggS"):
                        logger.warning(f"validate_audio_file: Opus header check failed for {audio_path}, but proceeding")
            except Exception as e:
                logger.warning(f"validate_audio_file: header check error: {e}")

        logger.info(f"validate_audio_file: passed: {audio_path} ({file_size} bytes)")
        return True
    except Exception as e:
        logger.error(f"validate_audio_file: validation failed: {audio_path}, error: {e}")
        return False
