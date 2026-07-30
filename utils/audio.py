import asyncio
import logging
from pathlib import Path
import shutil
import time
from typing import Optional
import wave

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


async def append_audio_silence(
    audio_path: Path,
    output_path: Path,
    duration_ms: int,
    *,
    audio_format: str,
    sample_rate: Optional[int] = None,
) -> Path:
    """Append silence without exposing a partially written output file.

    Args:
        audio_path: Source audio path.
        output_path: Destination path for the padded audio.
        duration_ms: Silence duration in milliseconds.
        audio_format: Source and destination audio format.
        sample_rate: PCM sample rate when the format is raw PCM.

    Returns:
        The destination path.

    Raises:
        RuntimeError: If the audio cannot be padded.
        ValueError: If the format or PCM sample rate is invalid.
    """
    if duration_ms <= 0:
        await asyncio.to_thread(shutil.copyfile, audio_path, output_path)
        return output_path

    fmt = str(audio_format or "").strip().lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "wav":

        def _append_pcm_wav() -> None:
            with wave.open(str(audio_path), "rb") as source:
                params = source.getparams()
                if params.comptype != "NONE":
                    raise wave.Error(f"unsupported WAV compression: {params.comptype}")
                frames = source.readframes(params.nframes)

            silence_frames = round(params.framerate * duration_ms / 1000)
            silence = b"\x00" * silence_frames * params.nchannels * params.sampwidth
            with wave.open(str(output_path), "wb") as target:
                target.setparams(params)
                target.writeframes(frames)
                target.writeframes(silence)

        try:
            await asyncio.to_thread(_append_pcm_wav)
            return output_path
        except (EOFError, OSError, wave.Error) as exc:
            logger.warning(
                "append_audio_silence: native WAV padding failed, falling back to ffmpeg: %s",
                exc,
            )

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to append silence to this audio format")

    args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    if fmt == "pcm":
        if not sample_rate or sample_rate <= 0:
            raise ValueError("sample_rate is required for raw PCM padding")
        args.extend(["-f", "s16le", "-ar", str(sample_rate), "-ac", "1"])

    args.extend(
        [
            "-i",
            str(audio_path),
            "-af",
            f"apad=pad_dur={duration_ms / 1000:.3f}",
        ]
    )
    if fmt == "wav":
        args.extend(["-c:a", "pcm_s16le"])
    elif fmt == "mp3":
        args.extend(["-c:a", "libmp3lame"])
    elif fmt == "opus":
        args.extend(["-c:a", "libopus"])
    elif fmt == "pcm":
        args.extend(["-c:a", "pcm_s16le", "-f", "s16le"])
    else:
        raise ValueError(f"unsupported audio format for silence padding: {fmt}")
    args.append(str(output_path))

    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        output_path.unlink(missing_ok=True)
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg silence padding failed: {detail}")

    return output_path


async def validate_audio_file(
    audio_path: Path,
    expected_format: Optional[str] = None,
    *,
    strict_format: bool = False,
) -> bool:
    """Validate an audio file asynchronously.

    Args:
        audio_path: Audio file path.
        expected_format: Expected format such as mp3, wav, opus, or pcm.
        strict_format: Return false when the file header does not match.

    Returns:
        Whether the file passed validation.
    """
    return await asyncio.to_thread(
        _validate_audio_file_sync,
        audio_path,
        expected_format,
        strict_format,
    )


def _validate_audio_file_sync(
    audio_path: Path,
    expected_format: Optional[str] = None,
    strict_format: bool = False,
) -> bool:
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
            logger.error(
                f"validate_audio_file: file too small ({file_size} bytes): {audio_path}"
            )
            return False

        # 扩展名检查
        if (
            audio_path.suffix.lower()
            and audio_path.suffix.lower() not in AUDIO_VALID_EXTENSIONS
        ):
            logger.warning(f"validate_audio_file: unexpected extension: {audio_path}")

        # 文件头检查
        if expected_format:
            try:
                with open(audio_path, "rb") as f:
                    header = f.read(12)

                fmt = expected_format.lower()
                header_valid = True
                if fmt == "mp3":
                    # MP3: ID3 or sync word
                    header_valid = bool(
                        header.startswith(b"ID3")
                        or header.startswith(b"\xff\xfb")
                        or header.startswith(b"\xff\xfa")
                        or (
                            len(header) >= 2
                            and header[0] == 0xFF
                            and (header[1] & 0xE0) == 0xE0
                        )
                    )
                elif fmt == "wav":
                    # WAV: RIFF ... WAVE
                    header_valid = header.startswith(b"RIFF") and b"WAVE" in header
                elif fmt == "opus":
                    # Opus: OggS
                    header_valid = header.startswith(b"OggS")

                if not header_valid:
                    message = (
                        f"validate_audio_file: {fmt.upper()} header check failed for "
                        f"{audio_path}"
                    )
                    if strict_format:
                        logger.error(message)
                        return False
                    logger.warning(f"{message}, but proceeding")

                if fmt == "wav" and strict_format:
                    with wave.open(str(audio_path), "rb") as wav_file:
                        if wav_file.getnframes() <= 0 or wav_file.getframerate() <= 0:
                            logger.error(
                                "validate_audio_file: WAV contains no decodable frames: %s",
                                audio_path,
                            )
                            return False
            except Exception as e:
                if strict_format:
                    logger.error(f"validate_audio_file: format validation error: {e}")
                    return False
                logger.warning(f"validate_audio_file: header check error: {e}")

        logger.info(f"validate_audio_file: passed: {audio_path} ({file_size} bytes)")
        return True
    except Exception as e:
        logger.error(
            f"validate_audio_file: validation failed: {audio_path}, error: {e}"
        )
        return False
