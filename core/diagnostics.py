"""Safe, cache-independent provider checks."""
import asyncio
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit


def configuration_errors(cfg, *, require_default_voice=True):
    errors = []
    for field, label in (("key", "API Key"), ("model", "模型"),
                         ("default_voice", "默认音色")):
        if field == "default_voice" and not require_default_voice:
            continue
        if not str(cfg.get(field) or "").strip():
            errors.append(f"TTS 引擎 → {cfg['provider']} → {label} 未填写")
    try:
        url = urlsplit(str(cfg.get("url") or ""))
        valid_url = (url.scheme in ("http", "https") and bool(url.hostname)
                     and not url.username and not url.password)
        url.port
    except ValueError:
        valid_url = False
    if not valid_url:
        errors.append("服务地址须为不含账号密码的 HTTP/HTTPS 地址")
    for field, lower, upper in (("timeout", 1, 300), ("max_retries", 0, 5),
                                ("speed", 0.5, 2)):
        try:
            value = float(cfg[field])
            valid = math.isfinite(value) and lower <= value <= upper
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            errors.append(f"{field} 须在 {lower}～{upper} 之间")
    if cfg.get("format") not in ("mp3", "wav", "opus", "pcm", "flac"):
        errors.append("音频格式须为 mp3、wav、opus、pcm 或 flac")
    if cfg.get("provider") == "breeze":
        for field, lower, upper in (("volume", 0.01, 2), ("guidance_scale", 1, 10)):
            try:
                value = float(cfg[field])
                valid = math.isfinite(value) and lower <= value <= upper
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                errors.append(f"tts_engine.breeze.{field} 须在 {lower}～{upper} 之间")
        language_code = str(cfg.get("language_code") or "").strip()
        if language_code and not re.fullmatch(r"[A-Za-z]{2}", language_code):
            errors.append("tts_engine.breeze.language_code 须为两位 ISO 639-1 语言代码，或留空")
    return errors


class DecoderUnavailable(ValueError):
    """Local decoder failure; audio corruption has not been established."""


def check_decoder():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise DecoderUnavailable("请在 AstrBot 运行环境安装 ffmpeg")
    try:
        result = subprocess.run([ffmpeg, "-version"], capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        raise DecoderUnavailable("ffmpeg 无法运行，请检查安装、执行权限和系统负载") from None
    if result.returncode:
        raise DecoderUnavailable("ffmpeg 启动失败，请修复 AstrBot 运行环境中的 ffmpeg 安装")
    return ffmpeg


def decode_duration(path, fmt, sample_rate=0, channels=1):
    """Decode the entire stream to PCM; never estimate duration from file size."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise DecoderUnavailable("请在 AstrBot 运行环境安装 ffmpeg")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("音频文件不存在或为空")
    args = [ffmpeg, "-v", "error", "-xerror"]
    if fmt == "pcm":
        if sample_rate <= 0:
            raise DecoderUnavailable("PCM 须配置正确的采样率")
        args += ["-f", "s16le", "-ar", str(sample_rate), "-ac", str(channels)]
    args += ["-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
             "-f", "s16le", "pipe:1"]
    try:
        result = subprocess.run(args, capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise DecoderUnavailable("音频解码超时，已保留缓存；请检查系统负载后重试") from None
    except OSError:
        raise DecoderUnavailable("ffmpeg 无法运行，请检查安装和执行权限") from None
    if result.returncode < 0:
        raise DecoderUnavailable("ffmpeg 进程被终止，请检查系统资源后重试")
    if result.returncode or not result.stdout:
        raise ValueError("音频无法完整解码，请检查服务商格式与采样率")
    return len(result.stdout) / 32000


async def check_provider(cfg, factory):
    from ..tts.errors import failure
    failure.set(None)
    name = cfg["provider"]
    errors = configuration_errors(cfg)
    if errors:
        return [f"❌ {name} 配置：" + "；".join(errors),
                f"❌ {name} 连通与合成：未执行，请先补齐配置"]
    lines = [f"✅ {name} 配置完整"]
    client = None
    try:
        await asyncio.to_thread(check_decoder)
        client = factory(cfg)
        # Each run must reach the provider rather than reusing a cached sample.
        with tempfile.TemporaryDirectory(prefix="tts-check-") as directory:
            path = await client.synth("你好，请听清最后一个字：好。", cfg["default_voice"], Path(directory))
            if path is None:
                detail = failure.get()
                reason = detail.message if detail else "请求失败，请检查凭据、余额、模型和音色；CosyVoice 还需 ffmpeg"
                return lines + [f"❌ {name} 连通与合成：{reason}"]
            lines.append(f"✅ {name} 连通与真实合成")
            duration = await asyncio.to_thread(decode_duration, Path(path), cfg["format"],
                                              cfg.get("sample_rate", 0), cfg.get("channel", 1))
            lines += [f"✅ {name} 文件存在且非空",
                      f"✅ {name} 完整解码可播放，时长 {duration:.3f} 秒"]
    except DecoderUnavailable as exc:
        lines.append(f"❌ {name} 音频解码：{exc}")
    except ValueError:
        lines.append(f"❌ {name} 配置或音频校验失败，请检查模型、音色、格式、采样率及 ffmpeg 安装")
    except Exception:
        lines.append(f"❌ {name} 请求失败，请检查网络、服务地址和凭据")
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception:
                lines.append(f"❌ {name} 连接释放失败，请重载插件后重试")
    return lines
